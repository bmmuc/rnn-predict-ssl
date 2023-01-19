from pytorch_lightning import Trainer
from src.model_with_encoder import Predictor_encoder
from src.new_pred import Act_pos_forecasting
from src.predictor_with_multiple_hidden import PredictorEncoder
from src.pos_latent import PosLatent
from src.model import Predictor
from src.act_pos_latent import ActLatent

from pytorch_lightning.loggers import WandbLogger

def train():
	WINDOW_SIZE = 10
	BATCH_SIZE = 128 # testar 64
	HIDDEN_SIZE = 256
	WEIGHT_DECAY = 0
	MAX_EPOCHS = 128
	VAL_INTERVAL = 4
	HIDDEN_IS_WITH_NOISE = True
	NUM_OF_DATA_TO_TEST = 1000
	LEARNING_RATE = 1e-3
	FEATURES = 40
	AUTOENCODER = True
	percentage_to_train = 0.9
	WEIGHTS = (0.5, 0.5)
	percentage_to_val = 0.05
	total_of_data = 50000

	name = f'2 opt and w: {WEIGHTS}, more info for actions, gradient_clipping 0.0175, fix autoencoders_eval'
	# name = f'plotting max grads'

	wb_logger = WandbLogger(
				  name=name,
				  project='rnn-predict-next-positions-with-v3', 
				  save_dir='./next_positions',
				  log_model='all'
				)

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
		lr=LEARNING_RATE,
		# num_of_data_to_val = 256,
		num_of_data_to_val = 1000
		)


	# model = ActLatent(
	#     window= WINDOW_SIZE, 
	#     input_size= FEATURES , 
	#     hidden_size= HIDDEN_SIZE,
	#     batch_size=BATCH_SIZE,
	#     should_test_the_new_data_set=True,
	#     data_root= '../all_data/data-3v3-v2',
	#     # num_of_data_to_train = 256, 
	#     output_size=40,
	#     act_path = './next_positions/rnn-predict-next-positions-with-v3/2a5vkx8o/checkpoints/epoch=255-step=60927.ckpt',
	#     pos_path='./autoencoder/rnn-autoencoder-v3/1hzlnvtj/checkpoints/epoch=127-step=30463.ckpt',
	#     num_of_data_to_train = 30450,
	#     act_path_autoencoder='./autoencoder/rnn-autoencoder-v3/1cfcd03p/checkpoints/epoch=127-step=30463.ckpt',
	#     # num_of_data_to_train = 256, 
	#     # num_of_data_to_val = 256
	#     num_of_data_to_val = 256
	# )

	trainer = Trainer(gpus = 1,
					  max_epochs = MAX_EPOCHS,
					  logger=wb_logger,
					#   check_val_every_n_epoch = VAL_INTERVAL
					  )

	wb_logger.watch(model)
	trainer.fit(model)
	# model = model.load_from_checkpoint('./next_positions/rnn-predict-next-positions-with-v3/37j6va2n/checkpoints/epoch=127-step=30463.ckpt')

if __name__ == '__main__':
	train()
