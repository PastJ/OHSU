# main
from utilities_CBCL import (
    CSVLoader,
    GetData,
    RCI,
    CBCLPreProcessing,
    DataPreprocessor,
    ModelTrainer,
    Collapsing,
    SaveResults,
    Iteration,
)
import json
import os
import argparse

def main(config_path): 
    # Load config
    with open(config_path,'r') as f: 
        config = json.load(f)

    # Paths
    PredictorPath = config["predictors_path"]
    OutComePath = config["outcome_path"]
    OptionalPath = config["optional_path"]
    SavePath = config["output_path"]
    
    # Files
    OutComeFile = config["outcome_file"]
    OptionalFile = config["optional_file"]

    # Variables
    modes = config["modes"]
    include_previous_years = config["include_previous_years"]
    items = config["items"]
    labels = config["labels"]
    downsample = config["downsample"]


    # Check if paths exist
    if not os.path.exists(PredictorPath):
        raise FileNotFoundError(f"Predictors file not found: {PredictorPath}")
    if not os.path.exists(OutComePath):
        raise FileNotFoundError(f"Outcomes file not found: {OutComePath}")
    if not os.path.exists(OptionalPath):
        raise FileNotFoundError(f"Optional file not found: {OptionalPath}")
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)

    print(f"Predictors Path: {PredictorPath}")
    print(f"Outcomes Path: {OutComePath}")
    print(f"Optional Path: {OptionalPath}")
    print(f"Output Directory: {SavePath}")


    iterate = Iteration()
    iterate.iterater(
        OutComePath=OutComePath,
        OutComeFile=OutComeFile,
        PredictorPath=PredictorPath,
        OptionalPath=OptionalPath,
        OptionalFile=OptionalFile,
        SaveFoldPath=SavePath,
        modes=modes,
        items=items,
        labels=labels,
        include_previous_years=include_previous_years,
        downsamples=downsample,
    )

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the script with a config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config.json file")
    
    # Parse arguments
    args = parser.parse_args()

    # Call main() and pass the config path
    main(args.config)
