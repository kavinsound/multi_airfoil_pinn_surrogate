#!/bin/bash

#SBATCH --partition=main
#SBATCH --job-name=openfoam_batch
#SBATCH --array=1-16%16
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A/job_%a.out
module purge

source $HOME/.bashrc

mamba activate cfd_env

FOLDER_FILE="case_list.txt"
LINE_NUM=$SLURM_ARRAY_TASK_ID

TARGET_FOLDER=$(sed -n "${LINE_NUM}p" "$FOLDER_FILE")

cd "generated_cases/$TARGET_FOLDER"

python -u mesher.py

chmod +x Allrun

./Allrun

# add the rest later
