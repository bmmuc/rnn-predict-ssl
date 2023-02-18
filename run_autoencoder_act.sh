#!/bin/sh
#SBATCH --job-name=act_encoder
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
. "/usr/local/anaconda3/etc/profile.d/conda.sh"
conda activate bmmuc
git pull
python train_pos_autoencoder.py
