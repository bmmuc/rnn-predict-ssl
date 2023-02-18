#!/bin/sh
#SBATCH --job-name=install_bmmuc
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=10
. "/usr/local/anaconda3/etc/profile.d/conda.sh"
conda activate bmmuc
pip install -r requirements.txt
pip install torch --upgrade --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu116