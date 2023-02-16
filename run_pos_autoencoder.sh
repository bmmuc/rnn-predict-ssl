#!/bin/sh
#SBATCH -o slurm-%j.out # Write the log here
#SBATCH --gres=gpu:1 # Ask for 1 GPU
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
conda activate bmmuc
git pull
python train_pos_autoencoder.py
