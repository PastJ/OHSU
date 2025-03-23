from concurrent.futures import ProcessPoolExecutor
import itertools
from utilities_CBCL4 import WorkFlow, CSVLoader
import pandas as pd
import os.path as op


def run_workflows(paths, model_configs):
    # Load all CSV files
    csvloader = CSVLoader(folder_path=paths["PredictorPath"])
    csv_files = csvloader.list_csv_files()

    # Print Paths
    print("CSV Files:", csv_files)
    print("Outcome Path:", paths["OutComePath"])
    print("Predictor Path:", paths["PredictorPath"])
    print("Optional Path:", paths["OptionalPath"])
    print("Save Folder Path:", paths["SaveFolderPath"])

    # Create tasks for all combinations of CSV files and parameters
    tasks = []
    for csv_file in csv_files:
        for (
            mode,
            item,
            label_pair,
            include_previous_year,
            downsample,
        ) in itertools.product(
            model_configs["Modes"],
            model_configs["Item_Level"],
            model_configs["Labels"],
            model_configs["Include_Previous_Year"],
            model_configs["Downsample"],
        ):
            tasks.append(
                WorkFlow(
                    csv_file=csv_file,
                    paths=paths,
                    mode=mode,
                    item=item,
                    label_pair=label_pair,
                    include_previous_year=include_previous_year,
                    downsample=downsample,
                )
            )

    # Collect results into a single DataFrame
    final_results = []

    # Process tasks in parallel
    with ProcessPoolExecutor() as executor:
        for result in executor.map(WorkFlow.run, tasks):
            if result is not None:  # Ensure that only non-None results are appended
                final_results.append(result)

    # Combine all DataFrames
    final_results_df = pd.concat(final_results, ignore_index=True)

    # Save results DataFrames
    results_full_path = op.join(paths["SaveFolderPath"], "results_full.csv")
    final_results_df = final_results_df[
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

    final_results_df.to_csv(results_full_path, index=False)
    print(f"\n COMPLETE \n Saved results_df_full to: {results_full_path}")


if __name__ == "__main__":
    # Various Paths
    paths = {
        "OutComePath": "/Users/pastore/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/Documents/OHSU/20241028_TAN_Project/Data/ABCD/cbcl_sandbox/data/",
        "OutComeFile": "cbcl_data.csv",
        "PredictorPath": "/Users/pastore/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/Documents/OHSU/20241028_TAN_Project/Data/ABCD/cbcl_sandbox/20241216/abcd_processed_usable_12_13_2024_testing_year/",
        "OptionalPath": "/Users/pastore/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/Documents/OHSU/20241028_TAN_Project/Data/ABCD/cbcl_sandbox/20241216/",
        "OptionalFile": "abcd_final_varlist.csv",
        "SaveFolderPath": "/Users/pastore/Library/CloudStorage/OneDrive-OregonHealth&ScienceUniversity/Documents/OHSU/20241028_TAN_Project/Data/ABCD/cbcl_sandbox/20241216/abcd_processed_usable_12_13_2024_testing_year/output/",
    }

    # Model Configs
    model_configs = {
        "Modes": ["int", "ext"],
        "Item_Level": [False, True],
        "Labels": [["(0,1)", "(2,3)"], ["(0,2)", "(1,3)"]],
        "Include_Previous_Year": [False, True],
        "Downsample": [True],
    }

    # Run Everything
    run_workflows(paths, model_configs)
