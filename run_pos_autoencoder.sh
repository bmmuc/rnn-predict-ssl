#!/bin/sh
#SBATCH --job-name=rl-isaac-training
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32

conda activate bmmuc
git pull
python train_pos_autoencoder.py
