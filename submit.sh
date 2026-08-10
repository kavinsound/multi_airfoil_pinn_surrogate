#!/bin/bash


num_jobs=$(wc -l < case_list.txt)
max_index=$((num_jobs - 1))

# Submit the actual worker script, passing the dynamic range
sbatch --array=0-$max_index%16 job_array.slurm