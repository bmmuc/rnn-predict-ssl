#!/bin/sh
#SBATCH --job-name=pred_latent
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=50
. "/usr/local/anaconda3/etc/profile.d/conda.sh"
conda activate bmmuc
git pull
python train_pos_latent.py
