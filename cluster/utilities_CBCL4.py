import numpy as np
import pandas as pd
import os
import os.path as op
from pathlib import Path
from scipy.stats import norm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    make_scorer,
    precision_recall_fscore_support,
)
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
import warnings


class CSVLoader:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)

    def list_csv_files(self):
        """
        Lists all CSV files in the specified folder.

        Parameters:
            folder_path (str): The path to the folder.

        Returns:
            list: A list of file paths to CSV files.
        """
        folder = Path(self.folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"The folder {self.folder_path} does not exist.")

        # Use glob to find all CSV files
        csv_files = [file.name for file in folder.glob("*.csv")]
        return csv_files


class GetData:
    """
    A class to manage data retrieval for outcome, predictor, and optional files.

    Attributes:
    ----------
    OutComePath : str
        Path to the directory containing the outcome file.
    OutComeFile : str
        Name of the outcome file (CSV format).
    PredictorPath : str
        Path to the directory containing the predictor file.
    PredictorFile : str
        Name of the predictor file (CSV format).
    OptionalPath : str
        Path to the directory containing the optional file.
    OptionalFile : str
        Name of the optional file (CSV format).
    """

    @staticmethod
    def _ensure_trailing_slash(path):
        """
        Ensures the given path ends with a slash ('/' or '\\'), appending one if it's missing.
        """
        return (
            path if path and path.endswith(os.sep) else path + os.sep if path else None
        )

    def __init__(
        self,
        OutComePath=None,
        OutComeFile=None,
        PredictorPath=None,
        PredictorFile=None,
        OptionalPath=None,
        OptionalFile=None,
    ):
        """
        Initializes the GetData class with paths and filenames for outcome, predictor, and optional data.
        """
        self.OutComePath = self._ensure_trailing_slash(OutComePath)
        self.OutComeFile = OutComeFile
        self.PredictorPath = self._ensure_trailing_slash(PredictorPath)
        self.PredictorFile = PredictorFile
        self.OptionalPath = self._ensure_trailing_slash(OptionalPath)
        self.OptionalFile = OptionalFile
        self.year_n0 = int(PredictorFile[-5])
        self.year_n1 = self.year_n0 + 1

    def get_outcome(self, year_n1=None):
        """
        Reads the outcome data from the specified file and returns it as a DataFrame.
        Raises a FileNotFoundError if the file does not exist.
        """
        if not self.OutComePath or not self.OutComeFile:
            raise ValueError("OutComePath and OutComeFile must be specified.")

        filepath = op.join(self.OutComePath, self.OutComeFile)
        if not op.exists(filepath):
            raise FileNotFoundError(f"Outcome file not found at: {filepath}")

        if year_n1 is not None:
            self.year_n1 = year_n1
            self.year_n0 = year_n1 - 1

        cbcl_delta = pd.read_csv(filepath)
        df = cbcl_delta[
            [
                "src_subject_id",
                f"cbclTscore_int-{self.year_n0}_{self.year_n0}",
                f"cbclTscore_ext-{self.year_n0}_{self.year_n0}",
                f"cbclTscore_int-{self.year_n1}_{self.year_n1}",
                f"cbclTscore_ext-{self.year_n1}_{self.year_n1}",
                f"cbclDelta_int-{self.year_n0}_{self.year_n1}",
                f"cbclDelta_ext-{self.year_n0}_{self.year_n1}",
                f"cbclDelta_bthMax-{self.year_n0}_{self.year_n1}",
            ]
        ]

        return df

    def get_predictors(self):
        """
        Reads the predictor data from the specified file and returns it as a DataFrame.
        Raises a FileNotFoundError if the file does not exist.
        """
        if not self.PredictorPath or not self.PredictorFile:
            raise ValueError("PredictorPath and PredictorFile must be specified.")

        filepath = op.join(self.PredictorPath, self.PredictorFile)
        if not op.exists(filepath):
            raise FileNotFoundError(f"Predictor file not found at: {filepath}")

        data_df = pd.read_csv(filepath)
        columns_to_drop = ["cbcl_scr_dsm5_adhd_t", "cbcl_scr_syn_attention_t"]
        data_df = data_df.drop(
            columns=[col for col in columns_to_drop if col in data_df.columns],
            inplace=False,
        )
        return data_df, self.PredictorFile[:-4], self.year_n0

    def get_optional_data(self):
        """
        Reads the optional file if both the path and file are provided.
        Identifies common columns between the optional file and the outcome data.
        """
        if self.OptionalPath and self.OptionalFile:
            filepath = op.join(self.OptionalPath, self.OptionalFile)
            if not op.exists(filepath):
                raise FileNotFoundError(f"Optional file not found at: {filepath}")

            optional_df = pd.read_csv(filepath, encoding="latin-1")
            optional_df["item"] = optional_df["item"].fillna(0)

            predictors_df, _, _ = self.get_predictors()

            # Example logic to find common columns
            columns_df1 = list(optional_df[optional_df["item"] == 0]["varname"])
            columns_df2 = list(predictors_df.columns)

            # Find the intersection of column sets
            common_columns = list(set(columns_df1).intersection(columns_df2)) + [
                "src_subject_id"
            ]
            # common_columns.append("src_subject_id")

            return optional_df, common_columns
        else:
            missing = "OptionalPath" if not self.OptionalPath else "OptionalFile"
            raise ValueError(f"{missing} must be provided to use this method.")


class RCI:
    def __init__(self, sd=10, rel=0.8):
        """
        Initialize the RCI class with default standard deviation and reliability.

        Parameters:
        - sd (float): Standard deviation of the measurement.
        - rel (float): Reliability of the measurement.
        """
        self.sd = sd
        self.rel = rel

    def calculate_threshold(self, p=0.05, one_tailed=True):
        """
        Calculate the Reliable Change Index (RCI) threshold.

        Parameters:
        - p (float): Significance level (default 0.05).
        - one_tailed (bool): Whether the test is one-tailed (default True).

        Returns:
        - float: RCI threshold.
        """
        q = norm.ppf(1 - p) if one_tailed else norm.ppf(1 - p / 2)
        var = self.sd**2
        rci_threshold = q * np.sqrt(2 * (1 - self.rel) * var)
        return rci_threshold


class CBCLPreProcessing:
    def __init__(self, cbcl_delta, data_df, year_n0, rci):
        """
        Initialize the CBCLClassifier.

        Parameters:
        - cbcl_delta (pd.DataFrame): The CBCL delta data.
        - data_df (pd.DataFrame): Additional data for processing (e.g., sleep data).
        - year (int): The base year for the analysis.
        - lowrc_threshold (float): The RCI threshold for classification.
        """
        self.cbcl_delta = cbcl_delta
        self.data_df = data_df
        self.year_n1 = year_n0 + 1
        self.year_n0 = year_n0
        self.rciThreshold = rci

    @staticmethod
    def assign_class_INT_EXT_only(row, mode, rci_threshold, year_n0, year_n1):
        if (row[f"cbclTscore_{mode}-{year_n1}_{year_n1}"] <= 60) & (
            row[f"cbclDelta_{mode}-{year_n0}_{year_n1}"] <= rci_threshold
        ):
            return "stable control"
        elif (row[f"cbclTscore_{mode}-{year_n1}_{year_n1}"] <= 60) & (
            row[f"cbclDelta_{mode}-{year_n0}_{year_n1}"] >= rci_threshold
        ):
            return "sub-clinical worsening"
        elif (row[f"cbclTscore_{mode}-{year_n1}_{year_n1}"] >= 60) & (
            row[f"cbclDelta_{mode}-{year_n0}_{year_n1}"] <= rci_threshold
        ):
            return "stable clinical"
        elif (row[f"cbclTscore_{mode}-{year_n1}_{year_n1}"] >= 60) & (
            row[f"cbclDelta_{mode}-{year_n0}_{year_n1}"] >= rci_threshold
        ):
            return "clinical"

    @staticmethod
    def assign_class_INT_EXT_both(row, rci_threshold, year_n0, year_n1):
        if (
            (row[f"cbclTscore_int-{year_n1}_{year_n1}"] <= 60)
            & (row[f"cbclTscore_ext-{year_n1}_{year_n1}"] <= 60)
        ) & (row[f"cbclDelta_bthMax-{year_n0}_{year_n1}"] <= rci_threshold):
            return "stable control"
        elif (
            (row[f"cbclTscore_int-{year_n1}_{year_n1}"] <= 60)
            | (row[f"cbclTscore_ext-{year_n1}_{year_n1}"] <= 60)
        ) & (row[f"cbclDelta_bthMax-{year_n0}_{year_n1}"] >= rci_threshold):
            return "sub-clinical worsening"
        elif (
            (row[f"cbclTscore_int-{year_n1}_{year_n1}"] >= 60)
            | (row[f"cbclTscore_ext-{year_n1}_{year_n1}"] >= 60)
        ) & (row[f"cbclDelta_bthMax-{year_n0}_{year_n1}"] <= rci_threshold):
            return "stable clinical"
        elif (
            (row[f"cbclTscore_int-{year_n1}_{year_n1}"] >= 60)
            | (row[f"cbclTscore_ext-{year_n1}_{year_n1}"] >= 60)
        ) & (row[f"cbclDelta_bthMax-{year_n0}_{year_n1}"] >= rci_threshold):
            return "clinical"

    def create_labels(self, mode):
        """
        Create labels for classification based on the mode and thresholds.

        Parameters:
        - mode (str): Mode to classify ('int', 'ext', or 'both').
        - rci_threshold (float): The RCI threshold for classification.
        - include_previous_year (bool): Whether to include previous year data.

        Returns:
        - pd.DataFrame: A DataFrame with generated labels.
        """
        df = self.cbcl_delta.copy()
        df = df.reset_index(drop="index")  # Reset index for proper row-wise operations

        if mode in ["int", "ext"]:
            df["class_label"] = df.apply(
                lambda row: self.assign_class_INT_EXT_only(
                    row, mode, self.rciThreshold, self.year_n0, self.year_n1
                ),
                axis=1,
            )
        elif mode == "both":
            df["class_label"] = df.apply(
                lambda row: self.assign_class_INT_EXT_both(
                    row, self.rciThreshold, self.year_n0, self.year_n1
                ),
                axis=1,
            )

        # Create binary labels
        df["class_label_binary"] = df["class_label"].map(
            {
                "stable control": 0,
                "sub-clinical worsening": 1,
                "stable clinical": 2,
                "clinical": 3,
            }
        )
        return df

    def preprocess(
        self,
        mode: str,
        # year_n0: int,
        item=None,
        common_columns=None,
        include_previous_year=True,
    ):
        """

        Parameters:
        - mode (str): Mode to classify ('int', 'ext', or 'both').
        - year_n0: year predictors
        - year_n1: year predicting
        - item: include item level or not. Item is true, drop all items marked with a 1
        - include_previous_year (bool): Whether to include previous year data.

        Returns:
        - X: A DataFrame of predictors
        - y: A DataFrame of outcomes
        """
        # year_n1 = year_n0 + 1

        # Remove columns that are marked with a 1
        if item:
            self.data_df = self.data_df[common_columns]

        cbcl_df = self.create_labels(mode)
        df = pd.merge(self.data_df, cbcl_df, on="src_subject_id", how="left")

        # Threshold for missing values (80%)
        missing_percentage = df.isnull().mean(axis=1)
        df_cleaned = df[~(missing_percentage >= 0.8)]

        # Drop rows if y has NaNs
        df_cleaned = df_cleaned.dropna(subset=["class_label_binary"])

        # Define features and target variable based on `mode` and `include_previous_year`
        if include_previous_year is True and mode == "int":
            drop_columns = [
                "src_subject_id",
                # f'cbclTscore_int-{year_n0}_{year_n0}',
                f"cbclTscore_ext-{self.year_n0}_{self.year_n0}",
                f"cbclTscore_int-{self.year_n1}_{self.year_n1}",
                f"cbclTscore_ext-{self.year_n1}_{self.year_n1}",
                f"cbclDelta_int-{self.year_n0}_{self.year_n1}",
                f"cbclDelta_ext-{self.year_n0}_{self.year_n1}",
                f"cbclDelta_bthMax-{self.year_n0}_{self.year_n1}",
                "class_label",
                "class_label_binary",
            ]

        elif include_previous_year is True and mode == "ext":
            drop_columns = [
                "src_subject_id",
                f"cbclTscore_int-{self.year_n0}_{self.year_n0}",
                # f'cbclTscore_ext-{year_n0}_{year_n0}',
                f"cbclTscore_int-{self.year_n1}_{self.year_n1}",
                f"cbclTscore_ext-{self.year_n1}_{self.year_n1}",
                f"cbclDelta_int-{self.year_n0}_{self.year_n1}",
                f"cbclDelta_ext-{self.year_n0}_{self.year_n1}",
                f"cbclDelta_bthMax-{self.year_n0}_{self.year_n1}",
                "class_label",
                "class_label_binary",
            ]

        elif include_previous_year is True and mode == "both":
            drop_columns = [
                "src_subject_id",
                # f'cbclTscore_int-{year_n0}_{year_n0}',
                # f'cbclTscore_ext-{year_n0}_{year_n0}',
                f"cbclTscore_int-{self.year_n1}_{self.year_n1}",
                f"cbclTscore_ext-{self.year_n1}_{self.year_n1}",
                f"cbclDelta_int-{self.year_n0}_{self.year_n1}",
                f"cbclDelta_ext-{self.year_n0}_{self.year_n1}",
                f"cbclDelta_bthMax-{self.year_n0}_{self.year_n1}",
                "class_label",
                "class_label_binary",
            ]

        else:
            drop_columns = [
                "src_subject_id",
                f"cbclTscore_int-{self.year_n0}_{self.year_n0}",
                f"cbclTscore_ext-{self.year_n0}_{self.year_n0}",
                f"cbclTscore_int-{self.year_n1}_{self.year_n1}",
                f"cbclTscore_ext-{self.year_n1}_{self.year_n1}",
                f"cbclDelta_int-{self.year_n0}_{self.year_n1}",
                f"cbclDelta_ext-{self.year_n0}_{self.year_n1}",
                f"cbclDelta_bthMax-{self.year_n0}_{self.year_n1}",
                "class_label",
                "class_label_binary",
            ]

        X = df_cleaned.drop(columns=drop_columns)
        y = df_cleaned["class_label_binary"]
        df_info = {
            "item": item,
            "include_previous_year": include_previous_year,
            "Mode": mode,
            "year_n0": self.year_n0,
            "year_n1": self.year_n1,
            "rci": self.rciThreshold,
        }

        return X, y, df_info


class DataPreprocessor:
    def __init__(self, X, y, downsampling=False, num_bootstraps=None):
        """
        Initialize the DataPreprocessor with data and settings.

        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target labels.
        - downsampling (bool): Whether to apply downsampling for class balancing.
        """
        self.X = X
        self.y = y
        self.downsampling = downsampling
        self.num_bootstraps = num_bootstraps

        # These will be used to store splits
        self.scaler = StandardScaler()

    def split_data(self):
        """
        Split data into training, validation, and test sets.

        Returns:
        - X_train, X_val, X_test: Feature matrices for training, validation, and test sets.
        - y_train, y_val, y_test: Target labels for training, validation, and test sets.
        """
        # indices = np.arange(self.X.shape[0])

        # Initial split into train+validation and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=0.25, stratify=self.y, random_state=42
        )

        # Further split train+validation into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            stratify=y_train_val,
            random_state=42,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def apply_downsampling(self, X_train, y_train):
        """
        Perform downsampling to balance classes in the training set.

        Parameters:
        - X_train (array-like): Training feature matrix.
        - y_train (array-like): Training target labels.

        Returns:
        - X_train_balanced, y_train_balanced: Balanced training data.
        """
        value_counts = pd.Series(y_train).value_counts()
        largest_class = value_counts.idxmax()
        second_largest_class = value_counts.nlargest(2).index[1]
        target_count = value_counts[second_largest_class]

        # Separate the largest class
        X_train_largest = X_train[y_train == largest_class]
        y_train_largest = y_train[y_train == largest_class]

        # Downsample the largest class
        X_train_largest_downsampled, y_train_largest_downsampled = resample(
            X_train_largest,
            y_train_largest,
            replace=False,
            n_samples=target_count,
            random_state=42,
        )

        # Combine with the rest of the classes
        X_train_balanced = np.vstack(
            (X_train_largest_downsampled, X_train[y_train != largest_class])
        )
        y_train_balanced = np.hstack(
            (y_train_largest_downsampled, y_train[y_train != largest_class])
        )

        return pd.DataFrame(X_train_balanced, columns=X_train.columns), y_train_balanced

    def impute_and_scale(self, X_train, X_val, X_test):
        """
        Apply KNN imputation and standard scaling to the data.

        Parameters:
        - X_train, X_val, X_test: Feature matrices to process.

        Returns:
        - X_train_scaled, X_val_scaled, X_test_scaled: Processed feature matrices.
        """
        knn_imputer = KNNImputer(n_neighbors=7)

        # Impute missing values
        X_train = knn_imputer.fit_transform(X_train)
        X_val = knn_imputer.transform(X_val)
        X_test = knn_imputer.transform(X_test)

        # Standard scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled

    def bootstrap_data(self, X_train, X_val):
        """
        Generate bootstrap samples for training and validation data.

        Parameters:
        - X_train (array-like): Training feature matrix.
        - X_val (array-like): Validation feature matrix.

        Returns:
        - train_bootstrap_indices: List of bootstrap indices for training.
        - val_bootstrap_indices: List of bootstrap indices for validation.
        """
        # Generate indices for the data
        train_indices = np.arange(len(X_train))
        val_indices = np.arange(len(X_val))

        # Generate bootstrap samples for train and validation
        train_bootstrap_indices = [
            resample(train_indices, replace=True, random_state=i)
            for i in range(self.num_bootstraps)
        ]
        val_bootstrap_indices = [
            resample(val_indices, replace=True, random_state=i + 100)
            for i in range(self.num_bootstraps)
        ]

        return train_bootstrap_indices, val_bootstrap_indices

    def process_data(self):
        """
        Complete data processing workflow.

        Returns:
        - X_train, X_val, X_test: Processed feature matrices for training, validation, and test sets.
        - y_train, y_val, y_test: Target labels for training, validation, and test sets.
        """
        # Step 1: Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()

        # Step 2: Apply downsampling if needed
        if self.downsampling:
            X_train, y_train = self.apply_downsampling(X_train, y_train)

        # Step 3: Impute and scale the data
        X_train, X_val, X_test = self.impute_and_scale(X_train, X_val, X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test


class ModelTrainer2:
    def __init__(
        self,
        X,
        X_train,
        y_train,
        X_val,
        y_val,
        csvname: str,
        df_info,
        model=None,
        downsample=True,
    ):
        """
        Initialize the ModelTrainer with a model and settings.

        Parameters:
        - model: A scikit-learn classifier. Defaults to RandomForestClassifier.
        - num_classes (int): Number of classes in the target variable.
        """
        # self.model = (
        #    model
        #    if model
        #   else RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        # )
        self.model = model
        self.results = {}
        self.probas_dict_full = {}
        self.probas_dict_rfecv = {}
        self.downsampling = downsample
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X = X
        self.csvname = csvname
        self.df_info = df_info
        self.random_state = 42

    def get_modelname(self):
        # Get the class name of the model
        self.model_name = type(self.model).__name__
        # Optionally, clean up the name if needed (e.g., remove 'Classifier')
        self.model_name = self.model_name.replace("Classifier", "")

        return self.model_name

    def train(self):
        """
        Train the model on the provided training data.
        """
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluate the model and calculate metrics.

        Parameters:
        - X_train: Training features.
        - y_train: Training labels.
        - X_val: Validation features.
        - y_val: Validation labels.

        Returns:
        - accuracy: Accuracy score.
        - roc_auc: ROC-AUC score(s).
        """
        # Make predictions
        y_pred = self.model.predict(self.X_val)
        y_pred_proba = self.model.predict_proba(self.X_val)

        accuracy = accuracy_score(self.y_val, y_pred)
        roc_auc = None

        # Collect probabilities for analysis
        probas_dict_full = {"y_val": self.y_val, "y_pred": y_pred}
        for class_idx in range(y_pred_proba.shape[1]):
            probas_dict_full[f"y_pred_proba_{class_idx}"] = y_pred_proba[:, class_idx]
        self.probas_dict_full["Probabilities"] = pd.DataFrame(probas_dict_full)

        # Calculate ROC-AUC
        try:
            if y_pred_proba.shape[1] > 2:
                roc_auc = roc_auc_score(
                    self.y_val, y_pred_proba, multi_class="ovr", average=None
                )
            else:
                roc_auc = roc_auc_score(self.y_val, y_pred_proba[:, 1])
        except AttributeError:
            pass  # Some models may not support predict_proba

        # Calculate AUC-PR for each class
        num_classes = y_pred_proba.shape[1]
        auc_pr_list = []
        for class_idx in range(num_classes):
            precision, recall, _ = precision_recall_curve(
                self.y_val == class_idx, y_pred_proba[:, class_idx]
            )
            auc_pr = auc(recall, precision)
            auc_pr_list.append(auc_pr)
        auc_pr_mean = np.mean(auc_pr_list)

        # Confusion matrix and per-class metrics
        cm = confusion_matrix(self.y_val, y_pred)
        # cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sensitivity, specificity, ppv, npv = [], [], [], []
        total = np.sum(cm)

        for i in range(num_classes):
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = total - (TP + FN + FP)

            sensitivity.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
            specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
            ppv.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
            npv.append(TN / (TN + FN) if (TN + FN) > 0 else 0)

        avg_sensitivity = np.mean(sensitivity)
        avg_specificity = np.mean(specificity)
        avg_ppv = np.mean(ppv)
        avg_npv = np.mean(npv)
        class_counts = [np.sum(self.y_train == i) for i in range(num_classes)]

        # Collect results for this model
        results_full = {
            "Downsampling": self.downsampling,
            **{f"nClass{i}": class_counts[i] for i in range(num_classes)},
            "Accuracy": accuracy,
            "AUC_PR": auc_pr_mean,
            "Avg Sensitivity": avg_sensitivity,
            "Avg Specificity": avg_specificity,
            "Avg PPV": avg_ppv,
            "Avg NPV": avg_npv,
            **{f"Class {i} Sensitivity": sensitivity[i] for i in range(num_classes)},
            **{f"Class {i} Specificity": specificity[i] for i in range(num_classes)},
            **{f"Class {i} PPV": ppv[i] for i in range(num_classes)},
            **{f"Class {i} NPV": npv[i] for i in range(num_classes)},
        }

        if roc_auc is not None:
            if num_classes > 2:
                results_full.update(
                    **{f"Class {i} AUC_ROC": roc_auc[i] for i in range(num_classes)}
                )
            else:
                results_full["AUC-ROC"] = roc_auc

        results_full["csv"] = self.csvname
        results_full["Model"] = self.get_modelname()
        results_full["Mode"] = self.df_info["Mode"]
        results_full["Include Previous Year"] = self.df_info["include_previous_year"]
        results_full["Year_n0"] = self.df_info["year_n0"]
        results_full["Year_n1"] = self.df_info["year_n1"]
        results_full["rci"] = self.df_info["rci"]
        results_full["item"] = self.df_info["item"]
        results_full["nCBCL_cols"] = self.X.filter(regex="cbcl").shape[
            1
        ]  # to confirm number of correct dropped columns
        results_full["nFeat"] = self.X.shape[1]  # number of features being used
        results_full["nSubjs"] = self.X.shape[
            0
        ]  # number of subjects after removing nans

        # Store results in class attributes
        self.results = results_full

        return pd.DataFrame([self.results]), self.probas_dict_full["Probabilities"]


class Collapsing:
    """
    Class to process and evaluate model results using a collapsing framework.

    Attributes:
        results: DataFrame containing evaluation results.
        probabilities: Dictionary of model probabilities.
        labels: List of label groupings for collapsing. Looks like ['(0,1)','(2,3)']
        rci: Random index for filtering results.
        mode: Evaluation mode, either "int" or other.
        include_previous_year: Whether to include previous year's data.
        downsample: Whether downsampling is applied.
        item: Item-level evaluation flag.
    """

    def __init__(
        self,
        results,
        probabilities,
        labels,
        rci,
        mode,
        include_previous_year,
        downsample,
        item,
    ):
        self.results = results
        self.probabilities = probabilities
        self.labels = labels
        self.rci = rci
        self.mode = mode
        self.include_previous_year = include_previous_year
        self.downsample = downsample
        self.item = item

    def _get_model_index(self):
        """
        Filter results DataFrame and get the model index for the probabilities dictionary.
        """
        filtered_df = self.results[
            (self.results["Mode"] == self.mode)
            & (self.results["Include Previous Year"] == self.include_previous_year)
            & (self.results["Downsampling"] == self.downsample)
            & (self.results["rci"] == self.rci)
            & (self.results["item"] == self.item)
        ]

        if filtered_df.empty:
            raise ValueError("No matching results found for the given parameters.")

        return filtered_df.index[0]

    def _update_probabilities(self, df):
        """
        Update probabilities for collapsing labels.
        """
        binarize = [
            [int(x) for x in label.strip("()").split(",")] for label in self.labels
        ]
        df["y_val_updated"] = self.probabilities["y_val"].apply(
            lambda x: 0 if x in binarize[0] else 1
        )
        df["y_pred_updated"] = self.probabilities["y_pred"].apply(
            lambda x: 0 if x in binarize[0] else 1
        )

        for i, group in enumerate(binarize):
            df[self.labels[i]] = sum(
                self.probabilities[f"y_pred_proba_{class_idx}"] for class_idx in group
            )

        df["y_pred_reassigned"] = (
            df[self.labels]
            .idxmax(axis=1)
            .apply(lambda label: 0 if label == self.labels[0] else 1)
        )

        df["y_pred_final"] = df.apply(
            lambda row: row["y_pred_reassigned"]
            if row[self.labels[0]] > 0.5 or row[self.labels[1]] > 0.5
            else row["y_pred_updated"],
            axis=1,
        )
        return df

    def _calculate_metrics(self, y_val, y_pred, y_pred_proba):
        """
        Calculate evaluation metrics for each class and overall AUC.
        """
        metrics = {
            "Accuracy": [],
            "Sensitivity": [],
            "Specificity": [],
            "PPV": [],
            "NPV": [],
            "AUC_ROC": [],
            "PR_AUC": [],
            "PPV_05": [],
            "PPV_50": [],
            "HC": [],
        }

        for class_idx in range(y_pred_proba.shape[1]):
            y_val_binary = (y_val == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            y_pred_proba_class = y_pred_proba[:, class_idx]

            cm = confusion_matrix(y_val_binary, y_pred_binary)
            TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            acc = accuracy_score(y_val_binary, y_pred_binary)
            sens = TP / (TP + FN) if (TP + FN) > 0 else 0
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0
            ppv_value = TP / (TP + FP) if (TP + FP) > 0 else 0
            npv_value = TN / (TN + FN) if (TN + FN) > 0 else 0
            auc = roc_auc_score(y_val_binary, y_pred_proba_class)
            precision, recall, _ = precision_recall_curve(
                y_val_binary, y_pred_proba_class
            )
            avg_precision = average_precision_score(y_val_binary, y_pred_proba_class)

            # Prevalence
            prevalence = 0.05
            ppv_05 = (sens * prevalence) / (
                sens * prevalence + (1 - spec) * (1 - prevalence)
            )

            prevalence = 0.50
            ppv_50 = (sens * prevalence) / (
                sens * prevalence + (1 - spec) * (1 - prevalence)
            )

        # High Confidence
        high_threshold = 0.8
        low_threshold = 0.2

        # Max pred_proba
        if y_pred_proba.shape[1] == 2:
            max_probs = np.max(y_pred_proba, axis=1)

            # Find indices of high confident predictions
            high_confidence_indices = np.where(
                (max_probs >= high_threshold) | (max_probs <= low_threshold)
            )[0]

            # Percentage of high confidence predictions by total predictions
            HC = len(high_confidence_indices) / len(y_pred_proba_class)

            # Filter for high-confidence predictions
            high_confidence_mask = (max_probs >= high_threshold) | (
                max_probs <= low_threshold
            )

            y_true_high_conf = y_val_binary[high_confidence_mask]
            y_pred_high_conf = y_pred_binary[high_confidence_mask]

            cm = confusion_matrix(y_true_high_conf, y_pred_high_conf, labels=[0, 1])
            TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            sens_HC = TP / (TP + FN) if (TP + FN) > 0 else 0
            spec_HC = TN / (TN + FP) if (TN + FP) > 0 else 0

            prevalence = 0.05
            ppv_05_HC = (sens_HC * prevalence) / (
                sens_HC * prevalence + (1 - spec_HC) * (1 - prevalence)
            )

            accuracy_HC = accuracy_score(y_true_high_conf, y_pred_high_conf)
            # specificity_sensitivity_test = precision_recall_fscore_support(y_true_high_conf, y_pred_high_conf, average=None)[1]
            precision_HC, sensitivity_HC, fscore_HC, support_HC = (
                precision_recall_fscore_support(
                    y_true_high_conf, y_pred_high_conf, average="binary", beta=1.0
                )
            )  # Recall = Sensitivity

            ## Ratio of correct to incorrect high confidnence predictions
            # Determine correct and incorrect predictions
            correct_predictions = y_true_high_conf == y_pred_high_conf

            # Calculate number of correct and incorrect high-confidence predictions
            num_correct = np.sum(correct_predictions)
            num_incorrect = np.sum(~correct_predictions)

            # Calculate the ratio of high-confidence correct to incorrect predictions
            ratio_correct_to_incorrect = (
                num_correct / num_incorrect if num_incorrect != 0 else 1
            )  # "All Correct"#float('inf')

        else:
            print("More than two classes?")

        # Just the last class or class 1
        metrics["Accuracy"] = acc
        metrics["Sensitivity"] = sens
        metrics["Specificity"] = spec
        metrics["PPV"] = ppv_value
        metrics["NPV"] = npv_value
        metrics["AUC_ROC"] = auc
        metrics["PR_AUC"] = avg_precision
        metrics["PPV_05"] = ppv_05
        metrics["PPV_50"] = ppv_50
        metrics["HC"] = HC
        metrics["HC_ACC"] = accuracy_HC
        metrics["HC_PPV05"] = ppv_05_HC
        metrics["HC_FScore"] = fscore_HC
        metrics["HC_Ratio"] = ratio_correct_to_incorrect

        return pd.DataFrame([metrics])

    def results2x2(self):
        """
        Main method to process results for 2x2 evaluation.
        """
        # model_n = self._get_model_index()
        df2 = self.probabilities
        df2 = self._update_probabilities(df2)

        y_val = df2["y_val_updated"].values
        y_pred_proba = df2[self.labels].values
        y_pred = df2["y_pred_final"]
        # num_classes = len(np.unique(y_val))

        # overall_auc = roc_auc_score(y_val, y_pred_proba[:, 1])
        # print(f"AUC for {self.labels[0]} vs {self.labels[1]}: {overall_auc:.2f}")

        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)

        results_df = self.results.drop(
            columns=[
                col
                for col in self.results.columns
                if any(
                    metric in col
                    for metric in [
                        "Accuracy",
                        "AUC_ROC",
                        "Sensitivity",
                        "Specificity",
                        "PPV",
                        "NPV",
                        "AUC_PR",
                    ]
                )
            ]
        )

        results_df["Labels"] = [",".join(self.labels)]
        results_df = pd.concat([results_df, metrics], axis=1)
        # results_df.update({key: np.mean(value) for key, value in metrics.items()})

        return results_df


class SaveResults:
    def __init__(
        self,
        ranking_full,
        folderpath,
        csvfilename,
    ):
        """
        Initialize the SaveResults class.

        Parameters:
        - results_df_full: Full results DataFrame.
        - results_df_rfecv: RFECV results DataFrame.
        - ranking_full: List of DataFrames for full rankings.
        - ranking_rfecv: List of DataFrames for RFECV rankings.
        - folderpath: Path to save the results.
        - csvfilename: Base name for the output file.
        """
        self.ranking_full = ranking_full
        self.folderpath = Path(folderpath)  # Use Path for handling paths
        self.csvfilename = csvfilename

    def ensure_folder_exists(self):
        """
        Ensures that the output folder exists. If it doesn't, it creates the folder.
        """
        self.folderpath.mkdir(parents=True, exist_ok=True)
        # print(f"Folder ensured at: {self.folderpath}")

    def save_ranking(self):
        """
        Saves the provided DataFrames as individual CSV files in the specified folder.
        """
        # Ensure the folder exists before saving
        self.ensure_folder_exists()

        # Define the target substring
        target_substring = "['(0,1)', '(2,3)']"

        # Check if the substring is in the filename
        file_name = f"{self.csvfilename}_ranking.csv"
        if target_substring in file_name:
            file_path = self.folderpath / file_name
            self.ranking_full = pd.DataFrame(self.ranking_full)
            # print(f"Dataframe: {isinstance(self.ranking_full, pd.DataFrame)}")
            self.ranking_full.sort_values(
                by="Gini Importance boot_ave", ascending=False
            ).to_csv(file_path, index=False)
            print(f"Saved ranking_full to: {file_path}")

    def save_summary(self):
        # Ensure the folder exists before saving
        self.ensure_folder_exists()

        # Save results DataFrames
        results_full_path = op.join(
            self.folderpath, f"{self.csvfilename}_results_full.csv"
        )
        if "OP" in self.results_df_full.columns:
            self.results_df_full = self.results_df_full[
                [
                    "csv",
                    "OP",
                    "Mode",
                    "Labels",
                    "Model",
                    "Downsampling",
                    "Include Previous Year",
                    "Year_n0",
                    "Year_n1",
                    "rci",
                    "item",
                    "nCBCL_cols",
                    "nFeat",
                    "nSubjs",
                    "nClass0",
                    "nClass1",
                    "nClass2",
                    "nClass3",
                    "Accuracy",
                    "Sensitivity",
                    "Specificity",
                    "PPV",
                    "NPV",
                    "AUC_ROC",
                    "PR_AUC",
                    "PPV_05",
                    "PPV_50",
                    "HC",
                    "HC_ACC",
                    "HC_FScore",
                    "HC_PPV05",
                    "HC_Ratio",
                ]
            ]
        else:
            self.results_df_full = self.results_df_full[
                [
                    "csv",
                    "Mode",
                    "Labels",
                    "Model",
                    "Downsampling",
                    "Include Previous Year",
                    "Year_n0",
                    "Year_n1",
                    "rci",
                    "item",
                    "nCBCL_cols",
                    "nFeat",
                    "nSubjs",
                    "nClass0",
                    "nClass1",
                    "nClass2",
                    "nClass3",
                    "Accuracy",
                    "Sensitivity",
                    "Specificity",
                    "PPV",
                    "NPV",
                    "AUC_ROC",
                    "PR_AUC",
                    "PPV_05",
                    "PPV_50",
                    "HC",
                    "HC_ACC",
                    "HC_FScore",
                    "HC_PPV05",
                    "HC_Ratio",
                ]
            ]

        self.results_df_full.to_csv(results_full_path, index=False)
        print(f"Saved results_df_full to: {results_full_path}")


class WorkFlow:
    def __init__(
        self, csv_file, paths, mode, item, label_pair, include_previous_year, downsample
    ):
        self.csv_file = csv_file
        self.paths = paths
        self.mode = mode
        self.item = item
        self.label_pair = label_pair
        self.include_previous_year = include_previous_year
        self.downsample = downsample
        self.rci = None
        self.results_full = []  # Store results for each iteration

    def sub_WorkFlow(self, data_handler, iterate_year_y1=None):
        # Reading outcomes
        self.cbcl_delta = data_handler.get_outcome(year_n1=iterate_year_y1)

        # Reading predictors
        self.predictor_df, self.csvfilename, self.year = data_handler.get_predictors()

        # Reading optional data (only available for the second instance)
        _, self.common_columns = data_handler.get_optional_data()

        print(
            f"Processing {self.csv_file} mode={self.mode}, item={self.item}, include_previous_year={self.include_previous_year}, labels={self.label_pair}, downsample:{self.downsample}"
        )

        # ** Organize Data into X, y **
        if iterate_year_y1 is not None:
            cbcl_processor = CBCLPreProcessing(
                cbcl_delta=self.cbcl_delta,
                data_df=self.predictor_df,
                year_n0=iterate_year_y1 - 1,
                rci=self.rci,
            )

        else:
            cbcl_processor = CBCLPreProcessing(
                cbcl_delta=self.cbcl_delta,
                data_df=self.predictor_df,
                year_n0=self.year,
                rci=self.rci,
            )

        # Preprocess Data
        X, y, df_info = cbcl_processor.preprocess(
            mode=self.mode,
            # year_n0=self.year,
            item=self.item,
            common_columns=self.common_columns,
            include_previous_year=self.include_previous_year,
        )

        # ** Get Training and Validation Data **
        preprocessor = DataPreprocessor(X, y, downsampling=self.downsample)

        # Process the data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.process_data()

        # ** Train Data **
        auc_scorer = make_scorer(
            roc_auc_score, response_method="predict_proba", multi_class="ovr"
        )  # needs_threshold=True, response_method="predict"

        # Define Model
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

        # Setup trainer
        trainer = ModelTrainer2(
            X=X,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            csvname=self.csvfilename,
            df_info=df_info,
            downsample=True,
            model=model,
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        results, probabilities = trainer.evaluate()

        # Get importance
        perm_importances_full = permutation_importance(
            model,
            X_val,
            y_val,
            scoring=auc_scorer,
            n_repeats=10,
            random_state=42,
        )

        # Extract Gini importances from the model
        feature_names = X.columns
        gini_importances = model.feature_importances_
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        # Create a DataFrame for Gini importances
        self.ranking_full = pd.DataFrame(
            {
                "Feature": feature_names,
                "Gini Importance": gini_importances,
                "Full Perm Importance": perm_importances_full.importances_mean,
            }
        )

        # print(f"    Ranking shape: {self.ranking_full.shape}")

        # ** Collapse Results **
        # Instantiate the Collapsing class
        collapsing_full = Collapsing(
            results=results,
            probabilities=probabilities,
            labels=self.label_pair,
            rci=self.rci,
            mode=self.mode,
            include_previous_year=self.include_previous_year,
            downsample=self.downsample,
            item=self.item,
        )

        # Execute the results2x2 method
        updated_results_full = collapsing_full.results2x2()
        # self.results_full.append(updated_results_full)

        # Initialize the class
        saver = SaveResults(
            ranking_full=self.ranking_full,
            folderpath=self.paths["SaveFolderPath"],
            csvfilename=f"{self.csv_file[:-4]}_[{self.mode}]_rci_[{self.rci:.2f}]_item_[{self.item}]_{self.label_pair}_includePrevYear_[{self.include_previous_year}]_downsample_[{self.downsample}]_Y{self.year}_Y{df_info["year_n1"]}",  # add years
        )
        # Save the rankings as CSVs
        saver.save_ranking()

        return pd.DataFrame(updated_results_full)

    def sub_WorkFlow_bootstrap(
        self, data_handler, iterate_year_y1=None, num_bootstraps=10
    ):
        # Reading outcomes
        self.cbcl_delta = data_handler.get_outcome(year_n1=iterate_year_y1)

        # Reading predictors
        self.predictor_df, self.csvfilename, self.year = data_handler.get_predictors()

        # Reading optional data (only available for the second instance)
        _, self.common_columns = data_handler.get_optional_data()

        print(
            f"Processing {self.csv_file} mode={self.mode}, item={self.item}, include_previous_year={self.include_previous_year}, labels={self.label_pair}, downsample:{self.downsample}"
        )

        # ** Organize Data into X, y **
        if iterate_year_y1 is not None:
            cbcl_processor = CBCLPreProcessing(
                cbcl_delta=self.cbcl_delta,
                data_df=self.predictor_df,
                year_n0=iterate_year_y1 - 1,
                rci=self.rci,
            )

        else:
            cbcl_processor = CBCLPreProcessing(
                cbcl_delta=self.cbcl_delta,
                data_df=self.predictor_df,
                year_n0=self.year,
                rci=self.rci,
            )

        # Preprocess Data
        X, y, df_info = cbcl_processor.preprocess(
            mode=self.mode,
            # year_n0=self.year,
            item=self.item,
            common_columns=self.common_columns,
            include_previous_year=self.include_previous_year,
        )

        # ** Get Training and Validation Data **
        preprocessor = DataPreprocessor(
            X, y, downsampling=self.downsample, num_bootstraps=num_bootstraps
        )

        # 1) Split data in train, val, and test
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()

        # 2) Bootstrap data
        train_bootstrap_indices, val_bootstrap_indices = preprocessor.bootstrap_data(
            X_train, X_val
        )

        # ** Train Data **

        # Define scorer
        auc_scorer = make_scorer(
            roc_auc_score, response_method="predict_proba", multi_class="ovr"
        )  # needs_threshold=True

        # Define Model
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

        results_boot = []
        probabilities_boot = {}
        gini = []
        perm_importance = []
        for i in range(num_bootstraps):
            X_train_boot = X_train.iloc[train_bootstrap_indices[i], :]
            X_val_boot = X_val.iloc[val_bootstrap_indices[i], :]
            y_train_boot = y_train.iloc[train_bootstrap_indices[i]]
            y_val_boot = y_val.iloc[val_bootstrap_indices[i]]

            trainer = ModelTrainer2(
                X=X,
                X_train=X_train_boot,
                y_train=y_train_boot,
                X_val=X_val_boot,
                y_val=y_val_boot,
                csvname=self.csvfilename,
                df_info=df_info,
                downsample=True,
                model=model,
            )

            # Train the model
            trainer.train()

            # Evaluate the model
            results, probabilities_boot[i] = trainer.evaluate()
            results_boot.append(results)

            # Get importance
            perm_importances_full = permutation_importance(
                model,
                X_val_boot,
                y_val_boot,
                scoring=auc_scorer,
                n_repeats=10,
                random_state=42,
            )

            # Extract Gini importances from the model
            feature_names = X.columns
            gini_importances = model.feature_importances_
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

            gini.append(gini_importances)
            perm_importance.append(perm_importances_full.importances_mean)

        results_boot = pd.concat(results_boot, ignore_index=True)

        # Find the mean
        results_boot_ave = results_boot[
            [
                "nClass0",
                "nClass1",
                "nClass2",
                "nClass3",
                "Accuracy",
                "AUC_PR",
                "Avg Sensitivity",
                "Avg Specificity",
                "Avg PPV",
                "Avg NPV",
                "Class 0 Sensitivity",
                "Class 1 Sensitivity",
                "Class 2 Sensitivity",
                "Class 3 Sensitivity",
                "Class 0 Specificity",
                "Class 1 Specificity",
                "Class 2 Specificity",
                "Class 3 Specificity",
                "Class 0 PPV",
                "Class 1 PPV",
                "Class 2 PPV",
                "Class 3 PPV",
                "Class 0 NPV",
                "Class 1 NPV",
                "Class 2 NPV",
                "Class 3 NPV",
                "Class 0 AUC_ROC",
                "Class 1 AUC_ROC",
                "Class 2 AUC_ROC",
                "Class 3 AUC_ROC",
            ]
        ].mean()

        non_integer_cols = [
            "Downsampling",
            "csv",
            "Model",
            "Mode",
            "Include Previous Year",
            "Year_n0",
            "Year_n1",
            "rci",
            "item",
            "nCBCL_cols",
            "nFeat",
            "nSubjs",
        ]

        # Add non-integer (or just info about the model run) to dataframe
        for col in non_integer_cols:
            results_boot_ave[col] = results_boot[col].iloc[0]
        results_boot_ave_df = pd.DataFrame(
            [results_boot_ave.values], columns=results_boot_ave.index
        )
        # Create a DataFrame for Gini importances
        self.ranking_full = pd.DataFrame(
            {
                "Feature": feature_names,
                "Gini Importance boot_ave": np.array(gini).mean(axis=0),
                "Gini Importance boot_std": np.array(gini).std(axis=0),
                "Full Perm Importance boot_ave": np.array(perm_importance).mean(axis=0),
                "Full Perm Importance boot_std": np.array(perm_importance).std(axis=0),
            }
        )

        # ** Collapse Results **
        results_full_boot = []
        for i in range(num_bootstraps):
            # Collapse full dataset
            collapsing_full = Collapsing(
                results=results_boot_ave_df,
                probabilities=probabilities_boot[i],
                labels=self.label_pair,
                rci=self.rci,
                mode=self.mode,
                include_previous_year=self.include_previous_year,
                downsample=self.downsample,
                item=self.item,
            )

            updated_results_full = collapsing_full.results2x2()
            results_full_boot.append(updated_results_full)
        results_full_boot = pd.concat(results_full_boot, ignore_index=True)

        # Compute mean and std for numerical metric columns
        metrics = [
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "AUC_ROC",
            "PR_AUC",
            "PPV_05",
            "PPV_50",
            "HC",
            "HC_ACC",
            "HC_FScore",
            "HC_PPV05",
            "HC_Ratio",
        ]

        results_full_boot_ave = results_full_boot[metrics].mean()
        results_full_boot_std = results_full_boot[metrics].std()

        # Metadata columns (categorical info)
        non_integer_cols = [
            "nClass0",
            "nClass1",
            "nClass2",
            "nClass3",
            "Downsampling",
            "csv",
            "Model",
            "Mode",
            "Include Previous Year",
            "Year_n0",
            "Year_n1",
            "rci",
            "item",
            "nCBCL_cols",
            "nFeat",
            "nSubjs",
            "Labels",
        ]

        # Assign metadata values (handling missing columns gracefully)
        for col in non_integer_cols:
            results_full_boot_ave[col] = results_full_boot.get(col, pd.NA).iloc[0]
            results_full_boot_std[col] = results_full_boot.get(col, pd.NA).iloc[0]

        # Convert to DataFrames (preserving column names)
        results_full_boot_ave_df = results_full_boot_ave.to_frame().T
        results_full_boot_ave_df["OP"] = "AVE"

        results_full_boot_std_df = results_full_boot_std.to_frame().T
        results_full_boot_std_df["OP"] = "STD"

        # Concatenate the two DataFrames
        results_full_boot_ave_df = pd.concat(
            [results_full_boot_ave_df, results_full_boot_std_df], ignore_index=True
        )

        # Ensure correct column order (only selecting existing columns)
        all_cols = ["OP"] + non_integer_cols + metrics
        existing_cols = [
            col for col in all_cols if col in results_full_boot_ave_df.columns
        ]

        results_full_boot_ave_df = results_full_boot_ave_df[existing_cols]

        # Initialize the class
        saver = SaveResults(
            ranking_full=self.ranking_full,
            folderpath=self.paths["SaveFolderPath"],
            csvfilename=f"{self.csv_file[:-4]}_[{self.mode}]_rci_[{self.rci:.2f}]_item_[{self.item}]_{self.label_pair}_includePrevYear_[{self.include_previous_year}]_downsample_[{self.downsample}]_Y{self.year}_Y{df_info["year_n1"]}",  # add years
        )
        # Save the rankings as CSVs
        saver.save_ranking()

        return pd.DataFrame(results_full_boot_ave_df)

    def run(self):
        # Data handler
        data_handler = GetData(
            OutComePath=self.paths["OutComePath"],
            OutComeFile=self.paths["OutComeFile"],  # "cbcl_data.csv"
            PredictorPath=self.paths["PredictorPath"],
            PredictorFile=self.csv_file,
            OptionalPath=self.paths["OptionalPath"],
            OptionalFile=self.paths["OptionalFile"],
        )

        # Initialize the RCI class with default parameters
        rci_calculator = RCI(sd=10, rel=0.8)
        if self.rci is None:
            self.rci = rci_calculator.calculate_threshold(
                p=0.2
            )  # 20% significance level, low rci

        else:
            # Calculate RCI thresholds for different significance levels
            self.rci = rci_calculator.calculate_threshold(
                p=0.05
            )  # 5% significance level, high rci

        # ** Iterate Over Different Years **
        if "dhx" in self.csv_file or "fhx" in self.csv_file:
            iterate_years_y1 = [1, 2, 3]
            self.results_full = []
            for iterate_year_y1 in iterate_years_y1:
                results = self.sub_WorkFlow_bootstrap(
                    data_handler, iterate_year_y1=iterate_year_y1, num_bootstraps=10
                )
                self.results_full.append(results)
            return pd.concat(self.results_full, ignore_index=True)

        # ** Standard 1-year out prediction **
        else:
            results_full = self.sub_WorkFlow_bootstrap(
                data_handler, iterate_year_y1=None, num_bootstraps=10
            )

            return pd.DataFrame(results_full)
