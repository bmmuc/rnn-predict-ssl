#!/bin/sh
#SBATCH -o slurm-%j.out # Write the log here
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
conda create -n bmmuc python=3.8.0
conda activate bmmuc
git pull
pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116