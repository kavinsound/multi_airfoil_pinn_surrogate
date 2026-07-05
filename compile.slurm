#!/bin/bash

#SBATCH --partition=main          # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --job-name=foam_compile       # Assign a short name to your job

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=1                # Total # of tasks across all nodes

#SBATCH --cpus-per-task=8         # Cores per task (>1 if multithread tasks)

#SBATCH --mem=8000                # Real memory (RAM) required (MB)

#SBATCH --time=12:00:00           # Total run time limit (HH:MM:SS)

#SBATCH --output=slurm.%N.%j.out  # STDOUT output file

#SBATCH --error=slurm.%N.%j.err   # STDERR output file (optional)

cd $HOME/openfoam

source $HOME/.bashrc 

mamba deactivate

wget https://dl.openfoam.com/source/v2512/OpenFOAM-v2512.tgz
tar -xzf OpenFOAM-v2512.tgz

wget https://dl.openfoam.com/source/v2512/ThirdParty-v2512.tgz
tar -xzf ThirdParty-v2512.tgz

source $HOME/openfoam/OpenFOAM-v2512/etc/bashrc

cd $HOME/openfoam/OpenFOAM-v2512

./Allwmake -j 8

echo "source $HOME/openfoam/OpenFOAM-v2512/etc/bashrc" >> $HOME/.bashrc

rm -rf *.tar *.tgz
