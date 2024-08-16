#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 1            # number of cores 
#SBATCH -t 1-00:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH --gpus=a100:1
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest
# mamba env list
# Using python, so source activate an appropriate environment
source activate wenA100

python train_dtfd_naive.py  --log_dir='dtfd_AFS_vit_c16' --dataroot='dataset/c16/vit' --distill_type='AFS'