#!/bin/bash

#SBATCH --partition=main          # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --job-name=foam_compile       # Assign a short name to your job

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=1                # Total # of tasks across all nodes

#SBATCH --cpus-per-task=8         # Cores per task (>1 if multithread tasks)

#SBATCH --mem=8000                # Real memory (RAM) required (MB)

#SBATCH --time=12:00:00           # Total run time limit (HH:MM:SS)

#SBATCH --output=logs/download.%j.out  # STDOUT output file

#SBATCH --error=logs/download.%j.err   # STDERR output file (optional)

source $HOME/.bashrc

mamba deactivate

mkdir -p $HOME/openfoam
cd $HOME/openfoam

wget https://dl.openfoam.com/source/v2512/OpenFOAM-v2512.tgz
tar -xf OpenFOAM-v2512.tgz

wget https://dl.openfoam.com/source/v2512/ThirdParty-v2512.tar.gz
tar -xf ThirdParty-v2512.tar.gz

source $HOME/openfoam/OpenFOAM-v2512/etc/bashrc
cd $HOME/openfoam/OpenFOAM-v2512

./Allwmake -j -k -l -s

