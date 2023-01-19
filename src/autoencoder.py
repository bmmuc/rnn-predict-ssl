import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from src.concat_data_set_autoencoder import ConcatDataSetAutoencoder
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F
import ipdb



class Encoder(nn.Module):
    def __init__(self, seq_len, no_features, embedding_size):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features    
        self.embedding_size = embedding_size   
        self.hidden_size = (2 * embedding_size)  
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = embedding_size,
            num_layers = 1,
            batch_first=True
        )
        
    def forward(self, x):
        
        x, (hidden_state, cell_state) = self.LSTM1(x)  
        last_lstm_layer_hidden_state = hidden_state[-1,:,:]
		
        return last_lstm_layer_hidden_state

class Decoder(nn.Module):
    def __init__(self, seq_len, no_features, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True
        )

        self.fc = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        out = self.fc(x)
        return out


class RecurrentAutoencoder(pl.LightningModule):
	def __init__(self, window = 10, input_size = 10, hidden_size=256, 
		batch_size = 32,
		data_root = '',
		num_of_data_to_train = 0, num_of_data_to_val = 0,
		should_test_the_new_data_set = False):
		super().__init__()
		self.batch_size = batch_size

		self.encoder = Encoder(window, input_size, hidden_size)
		self.decoder = Decoder(window, hidden_size, input_size)
		self.data_root = data_root
		self.should_test_the_new_data_set = should_test_the_new_data_set
		self.window = window
		self.batch_size = 128
		self.num_of_data_to_train = num_of_data_to_train
		self.num_of_data_to_val = num_of_data_to_val
		self.automatic_optimization = False
		self.save_hyperparameters()
		self.opt = self.configure_optimizers()

	def train_dataloader(self):
		
		dataset = ConcatDataSetAutoencoder(
			root_dir = self.data_root,
			num_of_data_sets= self.num_of_data_to_train,
			window = self.window,
			should_test_the_new_data_set = self.should_test_the_new_data_set,
			type_of_data= 'train',
			horizon = 1
			)

		loader = DataLoader(
					dataset,
					shuffle= True,
					batch_size=self.batch_size,
					num_workers=6,
					pin_memory=True
				)

		return loader

	def val_dataloader(self):
		dataset = ConcatDataSetAutoencoder(
			root_dir = self.data_root + '-val',
			num_of_data_sets = self.num_of_data_to_val,
			window = self.window,
			should_test_the_new_data_set = self.should_test_the_new_data_set,
			type_of_data= 'val',
			horizon = 1
			)

		loader = DataLoader(
					dataset,
					batch_size=self.batch_size,
					num_workers=1,
					pin_memory=True
				)

		return loader

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		
		return encoded, decoded

	def encoding(self, x):
		encoded = self.encoder(x)
		return encoded

	def decoding(self, x):
		decoded = self.decoder(x)
		return decoded

	def training_step(self, batch, batch_idx):
		self.train()
		X, y = batch

		# X = X.view((X.shape[0], 1 ,X.shape[1] * X.shape[2])).float()
		_, pred = self.forward(X)
		pred = pred.squeeze()
		y = y.squeeze()

		general_loss = F.mse_loss(pred, y)
		
		self.opt.zero_grad()
		self.manual_backward(general_loss)
		self.opt.step()
		

		ball_x_loss = F.mse_loss(pred[:,:,0], y[:,:,0])
		ball_y_loss = F.mse_loss(pred[:,:,1], y[:,:,1])
		ball_vx_loss = F.mse_loss(pred[:,:,2], y[:,:,2])
		ball_vy_loss = F.mse_loss(pred[:,:,3], y[:,:,3])        


		robot_blue_x_loss = F.mse_loss(pred[:,:,4], y[:,:,4])
		robot_blue_y_loss = F.mse_loss(pred[:,:,5], y[:,:,5])
		
		robot_blue_sin_loss = F.mse_loss(pred[:,:,6], y[:,:,6])
		robot_blue_cos_loss = F.mse_loss(pred[:,:,7], y[:,:,7])
		
		robot_blue_vx_loss = F.mse_loss(pred[:,:,8], y[:,:,8])
		robot_blue_vy_loss = F.mse_loss(pred[:,:,9], y[:,:,9])
		robot_blue_vw_loss = F.mse_loss(pred[:,:,10], y[:,:,10])
		
		robot_1blue_x_loss = F.mse_loss(pred[:,:,11], y[:,:,11])
		robot_1blue_y_loss = F.mse_loss(pred[:,:,12], y[:,:,12])

		robot_1blue_sin_loss = F.mse_loss(pred[:,:,13], y[:,:,13])
		robot_1blue_cos_loss = F.mse_loss(pred[:,:,14], y[:,:,14])

		robot_1blue_vx_loss = F.mse_loss(pred[:,:,15], y[:,:,15])
		robot_1blue_vy_loss = F.mse_loss(pred[:,:,16], y[:,:,16])
		robot_1blue_vw_loss = F.mse_loss(pred[:,:,17], y[:,:,17])

		robot_2blue_x_loss = F.mse_loss(pred[:,:,18], y[:,:,18])
		robot_2blue_y_loss = F.mse_loss(pred[:,:,19], y[:,:,19])

		robot_2blue_sin_loss = F.mse_loss(pred[:,:,20], y[:,:,20])
		robot_2blue_cos_loss = F.mse_loss(pred[:,:,21], y[:,:,21])

		robot_2blue_vx_loss = F.mse_loss(pred[:,:,22], y[:,:,22])
		robot_2blue_vy_loss = F.mse_loss(pred[:,:,23], y[:,:,23])
		robot_2blue_vw_loss = F.mse_loss(pred[:,:,24], y[:,:,24])

		robot_yellow_x_loss = F.mse_loss(pred[:,:,25], y[:,:,25])
		robot_yellow_y_loss = F.mse_loss(pred[:,:,26], y[:,:,26])


		robot_yellow_vx_loss = F.mse_loss(pred[:,:,27], y[:,:,27])
		robot_yellow_vy_loss = F.mse_loss(pred[:,:,28], y[:,:,28])
		robot_yellow_vw_loss = F.mse_loss(pred[:,:,29], y[:,:,29])

		robot_1yellow_x_loss = F.mse_loss(pred[:,:,30], y[:,:,30])
		robot_1yellow_y_loss = F.mse_loss(pred[:,:,31], y[:,:,31])

		robot_1yellow_vx_loss = F.mse_loss(pred[:,:,32], y[:,:,32])
		robot_1yellow_vy_loss = F.mse_loss(pred[:,:,33], y[:,:,33])
		robot_1yellow_vw_loss = F.mse_loss(pred[:,:,34], y[:,:,34])

		robot_2yellow_x_loss = F.mse_loss(pred[:,:,35], y[:,:,35])
		robot_2yellow_y_loss = F.mse_loss(pred[:,:,36], y[:,:,36])

		robot_2yellow_vx_loss = F.mse_loss(pred[:,:,37], y[:,:,37])
		robot_2yellow_vy_loss = F.mse_loss(pred[:,:,38], y[:,:,38])
		robot_2yellow_vw_loss = F.mse_loss(pred[:,:,39], y[:,:,39])

		reward_loss = F.mse_loss(pred[:,:,40], y[:,:,40])

		# robot_yellow_x_loss = F.mse_loss(pred[:,:,11], y[:,:,11])
		# robot_yellow_y_loss = F.mse_loss(pred[:,:,12], y[:,:,12])
		# robot_yellow_vx_loss = F.mse_loss(pred[:,:,13], y[:,:,13])
		# robot_yellow_vy_loss = F.mse_loss(pred[:,:,14], y[:,:,14])
		# robot_yellow_vw_loss = F.mse_loss(pred[:,:,15], y[:,:,15])



		self.log_dict({ 'train/loss/general_loss': general_loss, 
						'train/loss/ball_x_loss': ball_x_loss, 
						'train/loss/ball_y_loss': ball_y_loss,
						'train/loss/ball_vx_loss': ball_vx_loss,
						'train/loss/ball_vy_loss': ball_vy_loss,
						'train/loss/robot_0_blue_x_loss': robot_blue_x_loss, 
						'train/loss/robot_0_blue_y_loss': robot_blue_y_loss,
						'train/loss/robot_0_blue_sin_loss': robot_blue_sin_loss,
						'train/loss/robot_0_blue_cos_loss': robot_blue_cos_loss,
						'train/loss/robot_0_blue_vx_loss': robot_blue_vx_loss,
						'train/loss/robot_0_blue_vy_loss': robot_blue_vy_loss,
						'train/loss/robot_0_blue_vw_loss': robot_blue_vw_loss,
						'train/loss/robot_1_blue_x_loss': robot_1blue_x_loss,
						'train/loss/robot_1_blue_y_loss': robot_1blue_y_loss,
						'train/loss/robot_1_blue_sin_loss': robot_1blue_sin_loss,
						'train/loss/robot_1_blue_cos_loss': robot_1blue_cos_loss,
						'train/loss/robot_1_blue_vx_loss': robot_1blue_vx_loss,
						'train/loss/robot_1_blue_vy_loss': robot_1blue_vy_loss,
						'train/loss/robot_1_blue_vw_loss': robot_1blue_vw_loss,
						'train/loss/robot_2_blue_x_loss': robot_2blue_x_loss,
						'train/loss/robot_2_blue_y_loss': robot_2blue_y_loss,
						'train/loss/robot_2_blue_sin_loss': robot_2blue_sin_loss,
						'train/loss/robot_2_blue_cos_loss': robot_2blue_cos_loss,
						'train/loss/robot_2_blue_vx_loss': robot_2blue_vx_loss,
						'train/loss/robot_2_blue_vy_loss': robot_2blue_vy_loss,
						'train/loss/robot_2_blue_vw_loss': robot_2blue_vw_loss,
						'train/loss/robot_0_yellow_x_loss': robot_yellow_x_loss,
						'train/loss/robot_0_yellow_y_loss': robot_yellow_y_loss,
						'train/loss/robot_0_yellow_vx_loss': robot_yellow_vx_loss,
						'train/loss/robot_0_yellow_vy_loss': robot_yellow_vy_loss,
						'train/loss/robot_0_yellow_vw_loss': robot_yellow_vw_loss,
						'train/loss/robot_1_yellow_x_loss': robot_1yellow_x_loss,
						'train/loss/robot_1_yellow_y_loss': robot_1yellow_y_loss,
						'train/loss/robot_1_yellow_vx_loss': robot_1yellow_vx_loss,
						'train/loss/robot_1_yellow_vy_loss': robot_1yellow_vy_loss,
						'train/loss/robot_1_yellow_vw_loss': robot_1yellow_vw_loss,
						'train/loss/robot_2_yellow_x_loss': robot_2yellow_x_loss,
						'train/loss/robot_2_yellow_y_loss': robot_2yellow_y_loss,
						'train/loss/robot_2_yellow_vx_loss': robot_2yellow_vx_loss,
						'train/loss/robot_2_yellow_vy_loss': robot_2yellow_vy_loss,
						'train/loss/robot_2_yellow_vw_loss': robot_2yellow_vw_loss,
						'train/loss/reward_loss': reward_loss,
						})

		return general_loss

	def validation_step(self, batch, batch_idx):
		self.eval()
		X, y = batch
		# ipdb.set_trace()
		# X = X.view((X.shape[0], 1 ,X.shape[1] * X.shape[2])).float()
		_, pred = self.forward(X)

		pred = pred.squeeze()
	

		y = y.squeeze()

		general_loss = F.mse_loss(pred, y)
		# ipdb.set_trace()
		ball_x_loss = F.mse_loss(pred[:,:,0], y[:,:,0])
		ball_y_loss = F.mse_loss(pred[:,:,1], y[:,:,1])
		ball_vx_loss = F.mse_loss(pred[:,:,2], y[:,:,2])
		ball_vy_loss = F.mse_loss(pred[:,:,3], y[:,:,3])        


		robot_blue_x_loss = F.mse_loss(pred[:,:,4], y[:,:,4])
		robot_blue_y_loss = F.mse_loss(pred[:,:,5], y[:,:,5])
		
		robot_blue_sin_loss = F.mse_loss(pred[:,:,6], y[:,:,6])
		robot_blue_cos_loss = F.mse_loss(pred[:,:,7], y[:,:,7])
		
		robot_blue_vx_loss = F.mse_loss(pred[:,:,8], y[:,:,8])
		robot_blue_vy_loss = F.mse_loss(pred[:,:,9], y[:,:,9])
		robot_blue_vw_loss = F.mse_loss(pred[:,:,10], y[:,:,10])
		
		robot_1blue_x_loss = F.mse_loss(pred[:,:,11], y[:,:,11])
		robot_1blue_y_loss = F.mse_loss(pred[:,:,12], y[:,:,12])

		robot_1blue_sin_loss = F.mse_loss(pred[:,:,13], y[:,:,13])
		robot_1blue_cos_loss = F.mse_loss(pred[:,:,14], y[:,:,14])

		robot_1blue_vx_loss = F.mse_loss(pred[:,:,15], y[:,:,15])
		robot_1blue_vy_loss = F.mse_loss(pred[:,:,16], y[:,:,16])
		robot_1blue_vw_loss = F.mse_loss(pred[:,:,17], y[:,:,17])

		robot_2blue_x_loss = F.mse_loss(pred[:,:,18], y[:,:,18])
		robot_2blue_y_loss = F.mse_loss(pred[:,:,19], y[:,:,19])

		robot_2blue_sin_loss = F.mse_loss(pred[:,:,20], y[:,:,20])
		robot_2blue_cos_loss = F.mse_loss(pred[:,:,21], y[:,:,21])

		robot_2blue_vx_loss = F.mse_loss(pred[:,:,22], y[:,:,22])
		robot_2blue_vy_loss = F.mse_loss(pred[:,:,23], y[:,:,23])
		robot_2blue_vw_loss = F.mse_loss(pred[:,:,24], y[:,:,24])

		robot_yellow_x_loss = F.mse_loss(pred[:,:,25], y[:,:,25])
		robot_yellow_y_loss = F.mse_loss(pred[:,:,26], y[:,:,26])


		robot_yellow_vx_loss = F.mse_loss(pred[:,:,27], y[:,:,27])
		robot_yellow_vy_loss = F.mse_loss(pred[:,:,28], y[:,:,28])
		robot_yellow_vw_loss = F.mse_loss(pred[:,:,29], y[:,:,29])

		robot_1yellow_x_loss = F.mse_loss(pred[:,:,30], y[:,:,30])
		robot_1yellow_y_loss = F.mse_loss(pred[:,:,31], y[:,:,31])

		robot_1yellow_vx_loss = F.mse_loss(pred[:,:,32], y[:,:,32])
		robot_1yellow_vy_loss = F.mse_loss(pred[:,:,33], y[:,:,33])
		robot_1yellow_vw_loss = F.mse_loss(pred[:,:,34], y[:,:,34])

		robot_2yellow_x_loss = F.mse_loss(pred[:,:,35], y[:,:,35])
		robot_2yellow_y_loss = F.mse_loss(pred[:,:,36], y[:,:,36])

		robot_2yellow_vx_loss = F.mse_loss(pred[:,:,37], y[:,:,37])
		robot_2yellow_vy_loss = F.mse_loss(pred[:,:,38], y[:,:,38])
		robot_2yellow_vw_loss = F.mse_loss(pred[:,:,39], y[:,:,39])

		reward_loss = F.mse_loss(pred[:,:,40], y[:,:,40])


		self.log_dict({ 'train/loss/general_loss': general_loss, 
						'train/loss/ball_x_loss': ball_x_loss, 
						'train/loss/ball_y_loss': ball_y_loss,
						'train/loss/ball_vx_loss': ball_vx_loss,
						'train/loss/ball_vy_loss': ball_vy_loss,
						'train/loss/robot_0_blue_x_loss': robot_blue_x_loss, 
						'train/loss/robot_0_blue_y_loss': robot_blue_y_loss,
						'train/loss/robot_0_blue_sin_loss': robot_blue_sin_loss,
						'train/loss/robot_0_blue_cos_loss': robot_blue_cos_loss,
						'train/loss/robot_0_blue_vx_loss': robot_blue_vx_loss,
						'train/loss/robot_0_blue_vy_loss': robot_blue_vy_loss,
						'train/loss/robot_0_blue_vw_loss': robot_blue_vw_loss,
						'train/loss/robot_1_blue_x_loss': robot_1blue_x_loss,
						'train/loss/robot_1_blue_y_loss': robot_1blue_y_loss,
						'train/loss/robot_1_blue_sin_loss': robot_1blue_sin_loss,
						'train/loss/robot_1_blue_cos_loss': robot_1blue_cos_loss,
						'train/loss/robot_1_blue_vx_loss': robot_1blue_vx_loss,
						'train/loss/robot_1_blue_vy_loss': robot_1blue_vy_loss,
						'train/loss/robot_1_blue_vw_loss': robot_1blue_vw_loss,
						'train/loss/robot_2_blue_x_loss': robot_2blue_x_loss,
						'train/loss/robot_2_blue_y_loss': robot_2blue_y_loss,
						'train/loss/robot_2_blue_sin_loss': robot_2blue_sin_loss,
						'train/loss/robot_2_blue_cos_loss': robot_2blue_cos_loss,
						'train/loss/robot_2_blue_vx_loss': robot_2blue_vx_loss,
						'train/loss/robot_2_blue_vy_loss': robot_2blue_vy_loss,
						'train/loss/robot_2_blue_vw_loss': robot_2blue_vw_loss,
						'train/loss/robot_0_yellow_x_loss': robot_yellow_x_loss,
						'train/loss/robot_0_yellow_y_loss': robot_yellow_y_loss,
						'train/loss/robot_0_yellow_vx_loss': robot_yellow_vx_loss,
						'train/loss/robot_0_yellow_vy_loss': robot_yellow_vy_loss,
						'train/loss/robot_0_yellow_vw_loss': robot_yellow_vw_loss,
						'train/loss/robot_1_yellow_x_loss': robot_1yellow_x_loss,
						'train/loss/robot_1_yellow_y_loss': robot_1yellow_y_loss,
						'train/loss/robot_1_yellow_vx_loss': robot_1yellow_vx_loss,
						'train/loss/robot_1_yellow_vy_loss': robot_1yellow_vy_loss,
						'train/loss/robot_1_yellow_vw_loss': robot_1yellow_vw_loss,
						'train/loss/robot_2_yellow_x_loss': robot_2yellow_x_loss,
						'train/loss/robot_2_yellow_y_loss': robot_2yellow_y_loss,
						'train/loss/robot_2_yellow_vx_loss': robot_2yellow_vx_loss,
						'train/loss/robot_2_yellow_vy_loss': robot_2yellow_vy_loss,
						'train/loss/robot_2_yellow_vw_loss': robot_2yellow_vw_loss,
						'train/loss/reward_loss': reward_loss,
						})

	def configure_optimizers(self):
		return Adam(self.parameters())