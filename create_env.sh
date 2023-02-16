conda create -n bmmuc python=3.8.0
conda activate bmmuc
cd rnn-predict-ssl
git pull
pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116