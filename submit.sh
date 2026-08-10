#!/bin/bash


num_jobs=$(wc -l < case_list.txt)

# Submit the actual worker script, passing the dynamic range
sbatch --array=1-$num_jobs%16 job_array.slurm
