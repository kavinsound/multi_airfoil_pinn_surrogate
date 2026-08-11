#!/bin/bash

#SBATCH --partition=main-redhat
#SBATCH --job-name=openfoam_batch
#SBATCH --array=1-16%16
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/job_%A_%a.out
module purge

source $HOME/.bashrc

mamba activate cfd_env

FOLDER_FILE="case_list.txt"
LINE_NUM=$SLURM_ARRAY_TASK_ID

TARGET_FOLDER=$(sed -n "${LINE_NUM}p" "$FOLDER_FILE")

cd "generated_cases/$TARGET_FOLDER"

python -u mesher.py

# add the rest later
