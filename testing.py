from pytorch_lightning import Trainer
from scipy import rand
from src.pos_latent import PosLatent
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from utils.create_window import create_window
import numpy as np
import copy
import torch as th
from torch.nn import functional as F
import ipdb
import tqdm
import random

def train():
	WINDOW_SIZE = 10
	BATCH_SIZE = 256 # testar 64
	HIDDEN_SIZE = 256
	WEIGHT_DECAY = 0
	MAX_EPOCHS = 128
	VAL_INTERVAL = 4
	HIDDEN_IS_WITH_NOISE = True
	NUM_OF_DATA_TO_TEST = 1000
	LEARNING_RATE = 1e-4
	FEATURES = 40
	AUTOENCODER = True
	percentage_to_train = 0.9
	WEIGHTS = (0.7, 0.3)
	percentage_to_val = 0.05
	total_of_data = 50000

	name = f'1 opt and w: {WEIGHTS} and lr: e-4, new_net_2 and 256 batch_size'

	model = PosLatent(
		window= WINDOW_SIZE, 
		input_size= FEATURES , 
		hidden_size= HIDDEN_SIZE,
		batch_size=BATCH_SIZE,
		should_test_the_new_data_set=True,
		data_root= '../all_data/data-3v3-v2',
		# num_of_data_to_train = 256, 
		output_size=40,
		act_path = './autoencoder/rnn-autoencoder-v3/21z8udnb/checkpoints/epoch=127-step=60927.ckpt',
		pos_path='./autoencoder/rnn-autoencoder-v3/q9uv6d3w/checkpoints/epoch=127-step=30463.ckpt',
		num_of_data_to_train = 30450,
		# act_path_autoencoder='./autoencoder/rnn-autoencoder-v3/3r58fxvn/checkpoints/epoch=255-step=60927.ckpt',
		# num_of_data_to_train = 256, 
		weights = WEIGHTS,
		# num_of_data_to_val = 256,
		num_of_data_to_val = 1000
		)

	model = model.load_from_checkpoint('./next_positions/rnn-predict-next-positions-with-v3/37j6va2n/checkpoints/epoch=127-step=30463.ckpt')
	model = model.eval()

	datas = list()

	for i in tqdm.tqdm(range(250), desc=f'Creating the datasets for analyse'):
		data = np.loadtxt(open('../all_data/data-3v3-v1-val' + f'/positions-{i}.txt'), dtype=np.float32)
		if len(data) <= 30:
			continue
		datas.append(data)

	for horizon in [1, 5, 10, 15, 20]:
		print(f'carregando horizon: {horizon}')
		data_frame = pd.DataFrame()
		arr_loss_pos = list()
		arr_loss_act = list()
		arr_general_loss = list()
		for data in tqdm.tqdm(datas, desc=f'Creating the dataset for analyse: {horizon}'):
			out_to_test = create_window(data, WINDOW_SIZE, horizon)
			should_use = int(len(out_to_test) * round(random.randint(50, 100) / 100, 2))

			# ipdb.set_trace()
			for seq, y_true in tqdm.tqdm(out_to_test[:should_use], desc=f'doing preds in the window'):

				pos, act = model.predict_n_steps(seq, horizon)

				pos = pos[0][-1][:]
				act = act[0][-1][:]
				y_copy = th.FloatTensor(y_true[-1])
				y_copy = y_copy.to('cuda')

				y_pos = y_copy[[0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 21, 25, 26, 30, 31, 35, 36]]
				y_act = y_copy[[8, 9, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 32,33, 34, 37, 38, 39]]


				loss_pos = F.mse_loss(pos, y_pos)
				loss_act = F.mse_loss(act, y_act)

				loss_pos = loss_pos.item()
				loss_act = loss_act.item()

				general_loss = 0.7 * loss_pos +  0.3 * loss_act
				arr_loss_pos.append(loss_pos)
				arr_loss_act.append(loss_act)
				arr_general_loss.append(general_loss)

			# ipdb.set_trace()

		data_frame[f'loss_pos-{horizon}'] = arr_loss_pos
		data_frame[f'loss_act-{horizon}'] = arr_loss_act
		data_frame[f'general_loss-{horizon}'] = arr_general_loss
		data_frame.to_csv(f'comparacao-{horizon}.csv', index = False)


if __name__ == '__main__':
	train()
