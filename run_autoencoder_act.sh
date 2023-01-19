#!/bin/sh
#SBATCH -o slurm-%j.out # Write the log here
#SBATCH --gres=gpu:1 # Ask for 1 GPU
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
. "/usr/local/anaconda3/etc/profile.d/conda.sh"
conda create --name bmmuc python=3.8.0 pip
conda activate bmmuc
rm -rf rnn-predict-v3
git clone https://github.com/bmmuc/rnn-predict-v3.git
cd rnn-predict-v3
pip install -r env.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python train_autoencoder.py
