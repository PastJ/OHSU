# Example projects completed at OHSU

## Basic Codes 
1) **ADHD_1000_TableCreation_V1.ipynb**

Description: Read in multiple data files and cleaned data, dropped missing data over an 80% threshold. Covered columns to be one-hot-encoded, if applicable. Lastly, created a dataframe for each unique table for each unique year grouping the different features based on their table description. 

2) **ABCD_Var_Description_Clean_English_Spanish.ipynb**

Description: Cleaning variable descriptions from a dataframe. Removing non-English characters and Spanish words. 

## More Complex Codes
1) **ABCD_Organizing_V3.ipynb**

Description: 

Part 1:
Load in various csv files including a refined dictionary detailing features within overarching tables. Group the years as rows and features as columns. For each year (row) and feature (column) find 1) the percentage of missing data, 2) the variance of present data, and 3) the unique count of present data. All 3 will be concatentaed together and tranposed resulting in a dataframe of rows, features, and columns, the various year, of missing data, data variance, and unique data count. 

Part 2: 
Of the years and features, plot as a heatmap different bins of percentage missing data with features as the rows and years as a columns. 

Part 3: Using the refined dictionary, group various features into 3 main groups: passive, ideation, and attempt with subgroups of parent (p), youth (y) or parent and youth (py). This data was converted into a wide dataframe and the various combinations per year of passive, attempt, and ideation can be plotted for each year of each main group: parent, youth, or parent and youth. 

Part 4: From year to year find how subjects change from none to passive, ideation, or attempt using a Sankey Diagram. 

Part 5: Attempt to find highly correlation features or feature groupings. First, correlation heatmaps were created between all features, including a clustered heatmap/correlational heatmap. A network graph was created from the heatmap. Nodes were grouped together by the larger table they were associated with. A similar attempt was made with TNSE and UMAP feature reduction plotting. 

Plots include: Heatplots (including correlational plots), Sankey Diagrams, Network Graphs, TSNE, UMAP 

## Cluster (folder)
Description: Run a random forest binary classification task on a high performance cluster. Various variations of the same files had to be computed such as accessing specific variables, down sampling, specific thresholds for variables, and others. 
1) **config.json**; configuration file on cluster directing the various paths for grabing data and saving results.
2) **main.py**; main python file that runs the script
3) **main.sh**; bash file script running the main.py script on the cluster
4) **run_workflow_wrapper.py**; wrapper function that runs a for loop for the various variations of data and file manipulation.
5) **utilites_CBCL4.py**; all of the various classes and functions needed to get data, preprocess the data, setup data for machine learning, run model, evaluation model, and save the results. 
