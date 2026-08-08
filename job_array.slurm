#!/bin/bash

#SBATCH --partition=main-redhat

#SBATCH --job-name=openfoam_batch

#SBATCH --array=1-16%16

#SBATCH --cpus-per-task=4

#SBATCH --mem=8G

#SBATCH --time=00:40:00

module purge

module load $HOME/.bashrc

FOLDER_LIST="case_list.txt"
LINE_NUM=$SLURM_ARRAY_TASK_ID

TARGET_FOLDER=$(sed -n "${LINE_NUM}p" "$FOLDER_FILE")

cd "generated_cases/$TARGET_FOLDER"

python mesher.py

# add the rest later