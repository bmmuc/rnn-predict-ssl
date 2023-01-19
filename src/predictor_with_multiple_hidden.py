import torch as th
from torch import nn
import pytorch_lightning as pl
from src.encoders_decoders import AttnEncoder, AttnDecoder
from src.pos_autoencoder import PositionAutoEncoder
from src.act_autoencoder import ActAutoEncoder
from src.autoencoder import RecurrentAutoencoder
from src.new_autoencoder import RecurrentAutoencoder2
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.concat_data_set import ConcatDataSet
import ipdb
import copy

from utils.create_window import create_window

class PredictorEncoder(pl.LightningModule):

    def __init__(self, input_size = 4, hidden_size = 256, hidden_size_pos = 128, 
    hidden_size_act = 64 ,window = 5,
    num_of_data_to_train = 1, num_of_data_to_val = 1, num_of_data_to_test = 6,
    batch_size = 32,  num_workers = 1, num_layers1 = 1, num_layers2 = 1,
    out_size = 4, out_size_pos = 20, out_size_act = 10, weight_decay = 0.0, lr = 1e-3, data_root = '',
    autoencoder_pos_path = '', autoencoder_act_path = '', decooder_pos_act_path = '',
    should_test_overffit = False, should_test_the_new_data_set = False,
    ) -> None:
        super().__init__()
        
        self.out_size = out_size

        self.weight_decay = weight_decay
        self.lr = lr

        self.num_of_data_to_train = num_of_data_to_train
        self.num_of_data_to_val = num_of_data_to_val
        self.num_of_data_to_test = num_of_data_to_test

        self.should_test_the_new_data_set = should_test_the_new_data_set

        self.batch_size = batch_size
        self.num_layers1 = num_layers1
        self.window = window
        self.num_layers2 = num_layers2
        self.num_workers = num_workers
        self.data_root = data_root

        self.should_test_overffit = should_test_overffit

        self.autoencoder_pos_path = autoencoder_pos_path
        self.autoencoder_act_path = autoencoder_act_path
        self.decooder_pos_act_path = decooder_pos_act_path

        self.hidden_size1 = hidden_size
        self.hidden_size_pos = hidden_size_pos
        self.hidden_size_act = hidden_size_act

        self.pos_autoencoder = PositionAutoEncoder(input_size=input_size, 
                                                hidden_size=self.hidden_size_pos,
                                                window=self.window,
                                                output_size=out_size_pos,
                                                )
        self.pos_autoencoder = self.pos_autoencoder.load_from_checkpoint(self.autoencoder_pos_path)
        self.pos_autoencoder = self.pos_autoencoder.to(th.device('cuda'))
        
        self.pos_autoencoder = self.pos_autoencoder.eval()

        self.act_autoencoder = ActAutoEncoder(input_size=input_size,
                                                hidden_size=self.hidden_size_act,
                                                window=self.window,
                                                output_size=out_size_act,
                                                )
        self.act_autoencoder = self.act_autoencoder.load_from_checkpoint(self.autoencoder_act_path)
        self.act_autoencoder = self.act_autoencoder.to(th.device('cuda'))
        
        self.act_autoencoder = self.act_autoencoder.eval()

        size_of_last_decoder = hidden_size_act + hidden_size_pos

        self.decoder_pos_act = AttnDecoder(hidden_size, size_of_last_decoder, window, out_size, True)

        self.model = nn.ModuleDict({
            'linear1': nn.Linear(size_of_last_decoder, 128),
            'linear2': nn.Linear(128, 128),
            'linear3': nn.Linear(128, size_of_last_decoder),
        })

        self.decoder_pos_act = self.decoder_pos_act.to(th.device('cuda'))

        th.autograd.set_detect_anomaly(True)

        # self.model2 = nn.ModuleDict({
        #     'ball_preds': self.ball_module,
        #     'ally_preds': self.ally_module,
        #     'enemy_preds': self.enemy_module,
        # })
        self = self.train()
        self = self.to(th.device('cuda'))
        self.save_hyperparameters()
        # self.automatic_optimization = False
        self.opt = self.configure_optimizers()
        

    # def update_indexes(self,n):
        # self.indexes_used.add(n)

    def train_dataloader(self):
        
        dataset = ConcatDataSet(
            root_dir = self.data_root,
            num_of_data_sets= self.num_of_data_to_train,
            window = self.window,
            type_of_data= 'train',
            should_test_the_new_data_set = self.should_test_the_new_data_set,
            should_test_overffit = self.should_test_overffit,
            horizon = 1
            )

        # self.num_of_windows = dataset.num_of_windows

        loader = DataLoader(
                    dataset,
                    shuffle= True,
                    batch_size=self.batch_size,
                    num_workers=6,
                    pin_memory=True
                )

        return loader

    def val_dataloader(self):
        dataset = ConcatDataSet(
                root_dir = self.data_root + '-val',
                num_of_data_sets = self.num_of_data_to_val,
                window = self.window,
                type_of_data= 'val',
                should_test_the_new_data_set = self.should_test_the_new_data_set,
                horizon = 1
            )

        loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=1,
                    pin_memory=True
                )

        return loader

    def test_dataloader(self):
        dataset = ConcatDataSet(
            root_dir = self.data_root,
            num_of_data_sets= self.num_of_data_to_test,
            window = self.window,
            horizon = 1
            )

        loader = DataLoader(
                    dataset,
                    shuffle= True,
                    batch_size=self.batch_size,
                    num_workers=8,
                    pin_memory=True
                )

        print('train_loaders loaded')

        return loader


    def forward(self, x):
        # ipdb.set_trace()

        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = x.to(th.device('cuda'))

        y_hist = copy.deepcopy(x)

        latent_pos = self.pos_autoencoder.encoding(x)

        latent_act = self.act_autoencoder.encoding(x)

        latent_combined = th.cat((latent_pos, latent_act), dim=2).to(th.device('cuda'))

        latent_combined = self.model['linear1'](latent_combined)
        latent_combined = self.model['linear2'](latent_combined)

        out = self.model['linear3'](latent_combined)

        out = self.decoder_pos_act(out, y_hist)

        out = out.view(out.shape[0], out.shape[2], out.shape[1])

        return out[:,:,-1]

    def predict_t_steps(self, x, t_steps):
        # ipdb.set_trace()
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        y_hist = copy.deepcopy(x)

        x = self.autoencoder.encoding(x)
        x = x.to('cuda')
        x = self.model['linear1'](x)
        x = self.model['linear2'](x)
        x = self.model['linear3'](x)

        # x = nn.Dropout(0.1)(x)

        x = self.model['linear4'](x)
        x = self.model['linear5'](x)
        x = self.model['linear6'](x)

        out = self.model['linear7'](x)
        hidden = copy.copy(out)
        # ipdb.set_trace()
        # out = out.view(out.shape[1], out.shape[0])
        out = self.autoencoder.decoding(out, y_hist)
        out = out.view(out.shape[0], out.shape[2], out.shape[1])
        y_hist = y_hist[:,:-1,:]
        out = out.view(out.shape[0], out.shape[2], out.shape[1])

        out = out[:, -1, :].view(out.shape[0], 1 ,out.shape[2])

        y_hist = th.cat((y_hist, out), dim=1)

        for _ in range(t_steps):
            hidden = hidden.to('cuda')
            hidden = self.model['linear1'](hidden)
            hidden = self.model['linear2'](hidden)
            hidden = self.model['linear3'](hidden)

            hidden = self.model['linear4'](hidden)
            hidden = self.model['linear5'](hidden)
            hidden = self.model['linear6'](hidden)

            hidden = self.model['linear7'](hidden)

            # hidden = copy.copy(out)
            # out = out.view(out.shape[1], out.shape[0])
            encoded = self.autoencoder.decoding(hidden, y_hist)

            encoded = encoded.view(encoded.shape[0], encoded.shape[2], encoded.shape[1])

            y_hist = y_hist[:,:-1,:]

            encoded = encoded.view(encoded.shape[0], encoded.shape[2], encoded.shape[1])
            encoded = encoded[:, -1, :].view(encoded.shape[0], 1 ,encoded.shape[2])

            y_hist = th.cat((y_hist, encoded), dim=1)

        return encoded

    def training_step(self, batch, batch_idx):

        # idx , X, y = batch
        X, y = batch

        # self.update_indexes(idx)
        # self.has_reached_all_windows()
        
        # ipdb.set_trace()
        X = X.view((X.shape[0], X.shape[2] , X.shape[1])).float()

        # # X = th.cat((, robot_values), dim=1)
        # pred = self.forward(robot_values)
        # ipdb.set_trace()
        pred = self.forward(X)
        pred = pred.squeeze()

        y = y.squeeze()

        # pred = pred[:, :11]
        # y = y[:, :11]

        general_loss = F.mse_loss(pred, y)
       
        # self.opt.zero_grad()
        # self.manual_backward(general_loss)
        # self.opt.step()

        ball_x_loss = F.mse_loss(pred[:,0], y[:,0])
        ball_y_loss = F.mse_loss(pred[:,1], y[:,1])
        ball_vx_loss = F.mse_loss(pred[:,2], y[:,2])
        ball_vy_loss = F.mse_loss(pred[:,3], y[:,3])        

        robot_blue_x_loss = F.mse_loss(pred[:,4], y[:,4])
        robot_blue_y_loss = F.mse_loss(pred[:,5], y[:,5])
        
        robot_blue_sin_loss = F.mse_loss(pred[:,6], y[:,6])
        robot_blue_cos_loss = F.mse_loss(pred[:,7], y[:,7])
        
        robot_blue_vx_loss = F.mse_loss(pred[:,8], y[:,8])
        robot_blue_vy_loss = F.mse_loss(pred[:,9], y[:,9])
        robot_blue_vw_loss = F.mse_loss(pred[:,10], y[:,10])
        robot_1blue_x_loss = F.mse_loss(pred[:,11], y[:,11])
        robot_1blue_y_loss = F.mse_loss(pred[:,12], y[:,12])

        robot_1blue_sin_loss = F.mse_loss(pred[:,13], y[:,13])
        robot_1blue_cos_loss = F.mse_loss(pred[:,14], y[:,14])

        robot_1blue_vx_loss = F.mse_loss(pred[:,15], y[:,15])
        robot_1blue_vy_loss = F.mse_loss(pred[:,16], y[:,16])
        robot_1blue_vw_loss = F.mse_loss(pred[:,17], y[:,17])

        robot_2blue_x_loss = F.mse_loss(pred[:,18], y[:,18])
        robot_2blue_y_loss = F.mse_loss(pred[:,19], y[:,19])

        robot_2blue_sin_loss = F.mse_loss(pred[:,20], y[:,20])
        robot_2blue_cos_loss = F.mse_loss(pred[:,21], y[:,21])

        robot_2blue_vx_loss = F.mse_loss(pred[:,22], y[:,22])
        robot_2blue_vy_loss = F.mse_loss(pred[:,23], y[:,23])
        robot_2blue_vw_loss = F.mse_loss(pred[:,24], y[:,24])

        robot_yellow_x_loss = F.mse_loss(pred[:,25], y[:,25])
        robot_yellow_y_loss = F.mse_loss(pred[:,26], y[:,26])


        robot_yellow_vx_loss = F.mse_loss(pred[:,27], y[:,27])
        robot_yellow_vy_loss = F.mse_loss(pred[:,28], y[:,28])
        robot_yellow_vw_loss = F.mse_loss(pred[:,29], y[:,29])

        robot_1yellow_x_loss = F.mse_loss(pred[:,30], y[:,30])
        robot_1yellow_y_loss = F.mse_loss(pred[:,31], y[:,31])

        robot_1yellow_vx_loss = F.mse_loss(pred[:,32], y[:,32])
        robot_1yellow_vy_loss = F.mse_loss(pred[:,33], y[:,33])
        robot_1yellow_vw_loss = F.mse_loss(pred[:,34], y[:,34])

        robot_2yellow_x_loss = F.mse_loss(pred[:,35], y[:,35])
        robot_2yellow_y_loss = F.mse_loss(pred[:,36], y[:,36])

        robot_2yellow_vx_loss = F.mse_loss(pred[:,37], y[:,37])
        robot_2yellow_vy_loss = F.mse_loss(pred[:,38], y[:,38])
        robot_2yellow_vw_loss = F.mse_loss(pred[:,39], y[:,39])

        # reward_loss = F.mse_loss(pred[:, :, 40], y[:,40])

        robot_yellow_x_loss = F.mse_loss(pred[:,11], y[:,11])
        robot_yellow_y_loss = F.mse_loss(pred[:,12], y[:,12])
        robot_yellow_vx_loss = F.mse_loss(pred[:,13], y[:,13])
        robot_yellow_vy_loss = F.mse_loss(pred[:,14], y[:,14])
        robot_yellow_vw_loss = F.mse_loss(pred[:,15], y[:,15])

        mean_robot_x_loss = (robot_blue_x_loss + robot_1blue_x_loss + robot_2blue_x_loss \
                                 + robot_yellow_x_loss + robot_1yellow_x_loss + robot_2yellow_x_loss ) / 6

        mean_robot_y_loss = (robot_blue_y_loss + robot_1blue_y_loss + robot_2blue_y_loss \
                                    + robot_yellow_y_loss + robot_1yellow_y_loss + robot_2yellow_y_loss ) / 6
        
        mean_robot_sin_loss = (robot_blue_sin_loss + robot_1blue_sin_loss + robot_2blue_sin_loss) / 3

        mean_robot_cos_loss = (robot_blue_cos_loss + robot_1blue_cos_loss + robot_2blue_cos_loss) / 3

        mean_robot_vx_loss = (robot_blue_vx_loss + robot_1blue_vx_loss + robot_2blue_vx_loss \
                                 + robot_yellow_vx_loss + robot_1yellow_vx_loss + robot_2yellow_vx_loss ) / 6

        mean_robot_vy_loss = (robot_blue_vy_loss + robot_1blue_vy_loss + robot_2blue_vy_loss \
                                    + robot_yellow_vy_loss + robot_1yellow_vy_loss + robot_2yellow_vy_loss ) / 6
        
        mean_robot_vw_loss = (robot_blue_vw_loss + robot_1blue_vw_loss + robot_2blue_vw_loss \
                                    + robot_yellow_vw_loss + robot_1yellow_vw_loss + robot_2yellow_vw_loss ) / 6

        self.log_dict({ 'train/loss/general_loss': general_loss,
                        'train/loss/mean_robot_x_loss': mean_robot_x_loss,
                        'train/loss/mean_robot_y_loss': mean_robot_y_loss,
                        'train/loss/mean_robot_sin_loss': mean_robot_sin_loss,
                        'train/loss/mean_robot_cos_loss': mean_robot_cos_loss,
                        'train/loss/mean_robot_vx_loss': mean_robot_vx_loss,
                        'train/loss/mean_robot_vy_loss': mean_robot_vy_loss,
                        'train/loss/mean_robot_vw_loss': mean_robot_vw_loss,
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
                        # 'train/loss/reward_loss': reward_loss,
                        })

        return general_loss

    def validation_step(self, batch, batch_idx):
        
        X, y = batch
        # ipdb.set_trace()
        X = X.view((X.shape[0], X.shape[2] , X.shape[1])).float()
        # ipdb.set_trace()
        # X = X.view((X.shape[0], 1 ,X.shape[1] * X.shape[2])).float()
        pred = self.forward(X)
        pred = pred.squeeze()

        y = y.squeeze()
        # ipdb.set_trace()
        # pred = th.unique(pred, dim=1)

        # pred = pred[:, :11]
        # y = y[:, :11]

        general_loss = F.mse_loss(pred, y)

        ball_x_loss = F.mse_loss(pred[:,0], y[:,0])
        ball_y_loss = F.mse_loss(pred[:,1], y[:,1])
        ball_vx_loss = F.mse_loss(pred[:,2], y[:,2])
        ball_vy_loss = F.mse_loss(pred[:,3], y[:,3])        


        robot_blue_x_loss = F.mse_loss(pred[:,4], y[:,4])
        robot_blue_y_loss = F.mse_loss(pred[:,5], y[:,5])
        
        robot_blue_sin_loss = F.mse_loss(pred[:,6], y[:,6])
        robot_blue_cos_loss = F.mse_loss(pred[:,7], y[:,7])

        robot_blue_vx_loss = F.mse_loss(pred[:,8], y[:,8])
        robot_blue_vy_loss = F.mse_loss(pred[:,9], y[:,9])
        robot_blue_vw_loss = F.mse_loss(pred[:,10], y[:,10])

        robot_1blue_x_loss = F.mse_loss(pred[:,11], y[:,11])
        robot_1blue_y_loss = F.mse_loss(pred[:,12], y[:,12])

        robot_1blue_sin_loss = F.mse_loss(pred[:,13], y[:,13])
        robot_1blue_cos_loss = F.mse_loss(pred[:,14], y[:,14])

        robot_1blue_vx_loss = F.mse_loss(pred[:,15], y[:,15])
        robot_1blue_vy_loss = F.mse_loss(pred[:,16], y[:,16])
        robot_1blue_vw_loss = F.mse_loss(pred[:,17], y[:,17])

        robot_2blue_x_loss = F.mse_loss(pred[:,18], y[:,18])
        robot_2blue_y_loss = F.mse_loss(pred[:,19], y[:,19])

        robot_2blue_sin_loss = F.mse_loss(pred[:,20], y[:,20])
        robot_2blue_cos_loss = F.mse_loss(pred[:,21], y[:,21])

        robot_2blue_vx_loss = F.mse_loss(pred[:,22], y[:,22])
        robot_2blue_vy_loss = F.mse_loss(pred[:,23], y[:,23])
        robot_2blue_vw_loss = F.mse_loss(pred[:,24], y[:,24])

        robot_yellow_x_loss = F.mse_loss(pred[:,25], y[:,25])
        robot_yellow_y_loss = F.mse_loss(pred[:,26], y[:,26])


        robot_yellow_vx_loss = F.mse_loss(pred[:,27], y[:,27])
        robot_yellow_vy_loss = F.mse_loss(pred[:,28], y[:,28])
        robot_yellow_vw_loss = F.mse_loss(pred[:,29], y[:,29])

        robot_1yellow_x_loss = F.mse_loss(pred[:,30], y[:,30])
        robot_1yellow_y_loss = F.mse_loss(pred[:,31], y[:,31])

        robot_1yellow_vx_loss = F.mse_loss(pred[:,32], y[:,32])
        robot_1yellow_vy_loss = F.mse_loss(pred[:,33], y[:,33])
        robot_1yellow_vw_loss = F.mse_loss(pred[:,34], y[:,34])

        robot_2yellow_x_loss = F.mse_loss(pred[:,35], y[:,35])
        robot_2yellow_y_loss = F.mse_loss(pred[:,36], y[:,36])

        robot_2yellow_vx_loss = F.mse_loss(pred[:,37], y[:,37])
        robot_2yellow_vy_loss = F.mse_loss(pred[:,38], y[:,38])
        robot_2yellow_vw_loss = F.mse_loss(pred[:,39], y[:,39])


        mean_robot_x_loss = (robot_blue_x_loss + robot_1blue_x_loss + robot_2blue_x_loss \
                                 + robot_yellow_x_loss + robot_1yellow_x_loss + robot_2yellow_x_loss ) / 6

        mean_robot_y_loss = (robot_blue_y_loss + robot_1blue_y_loss + robot_2blue_y_loss \
                                    + robot_yellow_y_loss + robot_1yellow_y_loss + robot_2yellow_y_loss ) / 6
        
        mean_robot_sin_loss = (robot_blue_sin_loss + robot_1blue_sin_loss + robot_2blue_sin_loss) / 3

        mean_robot_cos_loss = (robot_blue_cos_loss + robot_1blue_cos_loss + robot_2blue_cos_loss) / 3

        mean_robot_vx_loss = (robot_blue_vx_loss + robot_1blue_vx_loss + robot_2blue_vx_loss \
                                 + robot_yellow_vx_loss + robot_1yellow_vx_loss + robot_2yellow_vx_loss ) / 6

        mean_robot_vy_loss = (robot_blue_vy_loss + robot_1blue_vy_loss + robot_2blue_vy_loss \
                                    + robot_yellow_vy_loss + robot_1yellow_vy_loss + robot_2yellow_vy_loss ) / 6
        
        mean_robot_vw_loss = (robot_blue_vw_loss + robot_1blue_vw_loss + robot_2blue_vw_loss \
                                    + robot_yellow_vw_loss + robot_1yellow_vw_loss + robot_2yellow_vw_loss ) / 6

        
        self.log_dict({ 'val/loss/general_loss': general_loss, 
                        'val/loss/mean_robot_x_loss': mean_robot_x_loss,
                        'val/loss/mean_robot_y_loss': mean_robot_y_loss,
                        'val/loss/mean_robot_sin_loss': mean_robot_sin_loss,
                        'val/loss/mean_robot_cos_loss': mean_robot_cos_loss,
                        'val/loss/mean_robot_vx_loss': mean_robot_vx_loss,
                        'val/loss/mean_robot_vy_loss': mean_robot_vy_loss,
                        'val/loss/mean_robot_vw_loss': mean_robot_vw_loss,
                        'val/loss/ball_x_loss': ball_x_loss, 
                        'val/loss/ball_y_loss': ball_y_loss,
                        'val/loss/ball_vx_loss': ball_vx_loss,
                        'val/loss/ball_vy_loss': ball_vy_loss,
                        'val/loss/robot_0_blue_x_loss': robot_blue_x_loss, 
                        'val/loss/robot_0_blue_y_loss': robot_blue_y_loss,
                        'val/loss/robot_0_blue_sin_loss': robot_blue_sin_loss,
                        'val/loss/robot_0_blue_cos_loss': robot_blue_cos_loss,
                        'val/loss/robot_0_blue_vx_loss': robot_blue_vx_loss,
                        'val/loss/robot_0_blue_vy_loss': robot_blue_vy_loss,
                        'val/loss/robot_0_blue_vw_loss': robot_blue_vw_loss,
                        'val/loss/robot_1_blue_x_loss': robot_1blue_x_loss,
                        'val/loss/robot_1_blue_y_loss': robot_1blue_y_loss,
                        'val/loss/robot_1_blue_sin_loss': robot_1blue_sin_loss,
                        'val/loss/robot_1_blue_cos_loss': robot_1blue_cos_loss,
                        'val/loss/robot_1_blue_vx_loss': robot_1blue_vx_loss,
                        'val/loss/robot_1_blue_vy_loss': robot_1blue_vy_loss,
                        'val/loss/robot_1_blue_vw_loss': robot_1blue_vw_loss,
                        'val/loss/robot_2_blue_x_loss': robot_2blue_x_loss,
                        'val/loss/robot_2_blue_y_loss': robot_2blue_y_loss,
                        'val/loss/robot_2_blue_sin_loss': robot_2blue_sin_loss,
                        'val/loss/robot_2_blue_cos_loss': robot_2blue_cos_loss,
                        'val/loss/robot_2_blue_vx_loss': robot_2blue_vx_loss,
                        'val/loss/robot_2_blue_vy_loss': robot_2blue_vy_loss,
                        'val/loss/robot_2_blue_vw_loss': robot_2blue_vw_loss,
                        'val/loss/robot_0_yellow_x_loss': robot_yellow_x_loss,
                        'val/loss/robot_0_yellow_y_loss': robot_yellow_y_loss,
                        'val/loss/robot_0_yellow_vx_loss': robot_yellow_vx_loss,
                        'val/loss/robot_0_yellow_vy_loss': robot_yellow_vy_loss,
                        'val/loss/robot_0_yellow_vw_loss': robot_yellow_vw_loss,
                        'val/loss/robot_1_yellow_x_loss': robot_1yellow_x_loss,
                        'val/loss/robot_1_yellow_y_loss': robot_1yellow_y_loss,
                        'val/loss/robot_1_yellow_vx_loss': robot_1yellow_vx_loss,
                        'val/loss/robot_1_yellow_vy_loss': robot_1yellow_vy_loss,
                        'val/loss/robot_1_yellow_vw_loss': robot_1yellow_vw_loss,
                        'val/loss/robot_2_yellow_x_loss': robot_2yellow_x_loss,
                        'val/loss/robot_2_yellow_y_loss': robot_2yellow_y_loss,
                        'val/loss/robot_2_yellow_vx_loss': robot_2yellow_vx_loss,
                        'val/loss/robot_2_yellow_vy_loss': robot_2yellow_vy_loss,
                        'val/loss/robot_2_yellow_vw_loss': robot_2yellow_vw_loss,
                        })

        return general_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
