#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --output=test1.log
#SBATCH --error=test1.err
#SBATCH --time=072:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=12 

# Load required modules
module load python/3.11

# Activate environment
source /home/exacloud/gscratch/NagelLab/abcd/staff/pastore/CBCL/env/myenv/bin/activate

# Verify that the enviroment is activated
echo "Enviroment activated: $(which python)"

# Navigate to the source directory 
cd /home/exacloud/gscratch/NagelLab/abcd/staff/pastore/CBCL/scripts/

# Define the path to the config.json file
CONFIG_PATH="/home/exacloud/gscratch/NagelLab/abcd/staff/pastore/CBCL/scripts/config.json"

# Run the Python script with the config.json path as an argument
python -u main.py --config "$CONFIG_PATH"
