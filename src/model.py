import torch as th
from torch import nn
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.concat_data_set import ConcatDataSet
import ipdb
import numpy as np

from utils.create_window import create_window

class Predictor(pl.LightningModule):

    def __init__(self, input_size = 4, hidden_size = 256, hidden_size2 = 128, 
    hidden_size3 = 64 ,window = 5,
    num_of_data_to_train = 1, num_of_data_to_val = 1, num_of_data_to_test = 6,
    batch_size = 32,  num_workers = 1, num_layers1 = 1, num_layers2 = 1,
    out_size = 4, weight_decay = 0.0, lr = 1e-3, data_root = '',
    should_test_overffit = False
    ) -> None:
        super().__init__()
        
        self.out_size = out_size
        self.weight_decay = weight_decay
        self.lr = lr

        self.num_of_data_to_train = num_of_data_to_train
        self.num_of_data_to_val = num_of_data_to_val
        self.num_of_data_to_test = num_of_data_to_test

        self.batch_size = batch_size
        self.num_layers1 = num_layers1
        self.window = window
        self.num_layers2 = num_layers2
        self.num_workers = num_workers
        self.data_root = data_root

        self.should_test_overffit = should_test_overffit
        
        self.keys = ['train_x_val/general_loss','train_x_val/ball_x_loss',
                    'train_x_val/ball_y_loss', 'train_x_val/robot_x_loss', 'train_x_val/robot_y_loss']
        
        self.hidden_size1 = hidden_size
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3

        self.indexes = range(self.num_of_data_to_val)
        
        
        self.hidden = (th.zeros(1,1,hidden_size).to('cuda').float(),
                       th.zeros(1,1,hidden_size).to('cuda').float(),
                       )

        self.count = 0

        self.model = nn.ModuleDict({
            'lstm1': nn.LSTM(
                input_size = window*input_size,
                hidden_size = self.hidden_size1,
                num_layers = num_layers1,
            ),
            # 'dropout1': nn.Dropout(0.25),
            'lstm2': nn.LSTM(
                input_size = self.hidden_size1,
                hidden_size = self.hidden_size1,
                num_layers = num_layers2,
            ),

            'linear2': nn.Linear(
                in_features = self.hidden_size1,
                out_features = 512,
            ),
            'linear4': nn.Linear(
                in_features = 512,
                out_features = 512,
            ),
            'linear3': nn.Linear(
                in_features = 512,
                out_features = 512,
            ),

            'linear5': nn.Linear(
                in_features = 512,
                out_features = 128,
            ),
            'linear6': nn.Linear(
                in_features = 128,
                out_features = 128,
            ),
            'linear7': nn.Linear(
                in_features = 128,
                out_features = 64,
            ),
            'linear8': nn.Linear(
                in_features = 64,
                out_features = 64,
            ),
            'linear9': nn.Linear(
                in_features = 64,
                out_features = 32,
            ),
            'linear10': nn.Linear(
                in_features = 32,
                out_features = 32,
            ),
            # 'dropout3': nn.Dropout(0.1),
            'linear': nn.Linear(
                in_features=32,
                out_features=out_size)
        })

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.opt = self.configure_optimizers()
        

    def train_dataloader(self):
        
        dataset = ConcatDataSet(
            root_dir = self.data_root,
            num_of_data_sets= self.num_of_data_to_train,
            window = self.window,
            type_of_data= 'train',
            should_test_the_new_data_set=True,
            should_test_overffit = self.should_test_overffit,
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
        dataset = ConcatDataSet(
            root_dir = self.data_root + '-val',
            num_of_data_sets = self.num_of_data_to_val,
            should_test_the_new_data_set=True,
            window = self.window,
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

    def initialize_hidden(self):
        self.hidden = ( 0.01 * th.rand(1, 1, self.hidden_size1).to('cuda').float(),
                        0.01 * th.rand(1, 1, self.hidden_size1).to('cuda').float(),
                       )

    def forward(self, x):

        out, self.hidden = self.model['lstm1'](x, self.hidden)

        # out = self.model['dropout1'](out)
        
        out, _ = self.model['lstm2'](out, self.hidden)

        # out = self.model['dropout2'](out)

        out = self.model['linear2'](out)

        out = self.model['linear4'](out)
    
        out = self.model['linear3'](out)
        
        # out = self.model['dropout4'](out)

        out = self.model['linear5'](out)

        out = self.model['linear6'](out)

        out = self.model['linear7'](out)

        out = self.model['linear8'](out)

        out = self.model['linear9'](out)

        out = self.model['linear10'](out)

        # out = self.model['dropout3'](out)

        out = self.model['linear'](out)

        return out
        
    def training_step(self, batch, batch_idx):

        self.initialize_hidden()
        self.count = 0
        self.model.train()
        
        X, y = batch

        X = X.view((X.shape[0], 1 ,X.shape[1] * X.shape[2])).float()
        pred = self.forward(X)
        pred = pred.squeeze()
        y = y.squeeze()

        general_loss = F.mse_loss(pred, y)
        
        self.opt.zero_grad()
        self.manual_backward(general_loss)
        self.opt.step()
        

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
        
        robot_yellow_x_loss = F.mse_loss(pred[:,11], y[:,11])
        robot_yellow_y_loss = F.mse_loss(pred[:,12], y[:,12])
        robot_yellow_vx_loss = F.mse_loss(pred[:,13], y[:,13])
        robot_yellow_vy_loss = F.mse_loss(pred[:,14], y[:,14])
        robot_yellow_vw_loss = F.mse_loss(pred[:,15], y[:,15])


        self.log_dict({ 'train/loss/general_loss': general_loss, 
                        'train/loss/ball_x_loss': ball_x_loss, 
                        'train/loss/ball_y_loss': ball_y_loss,
                        'train/loss/ball_vx_loss': ball_vx_loss,
                        'train/loss/ball_vy_loss': ball_vy_loss,
                        'train/loss/robot_blue_x_loss': robot_blue_x_loss, 
                        'train/loss/robot_blue_y_loss': robot_blue_y_loss,
                        'train/loss/robot_blue_sin_loss': robot_blue_sin_loss,
                        'train/loss/robot_blue_cos_loss': robot_blue_cos_loss,
                        'train/loss/robot_blue_vx_loss': robot_blue_vx_loss,
                        'train/loss/robot_blue_vy_loss': robot_blue_vy_loss,
                        'train/loss/robot_blue_vw_loss': robot_blue_vw_loss,
                        'train/loss/robot_yellow_x_loss': robot_yellow_x_loss,
                        'train/loss/robot_yellow_y_loss': robot_yellow_y_loss,
                        'train/loss/robot_yellow_vx_loss': robot_yellow_vx_loss,
                        'train/loss/robot_yellow_vy_loss': robot_yellow_vy_loss,
                        'train/loss/robot_yellow_vw_loss': robot_yellow_vw_loss,
                         })

        return general_loss
    
    def test_step(self, batch, batch_idx):
        self.hidden = (th.zeros(1, 1, self.hidden_size1).to('cuda').float(),
                       th.zeros(1, 1, self.hidden_size1).to('cuda').float(),
                       )

        X, y = batch

        X = X.view((X.shape[0], 1 ,X.shape[1] * X.shape[2])).float()

        pred = self.forward(X)
        pred = pred.squeeze()
        y = y.squeeze()

        general_loss = F.mse_loss(pred, y)

        ball_x_loss = F.mse_loss(pred[:,0], y[:,0])
        ball_y_loss = F.mse_loss(pred[:,1], y[:,1])
        robot_x_loss = F.mse_loss(pred[:,2], y[:,2])
        robot_y_loss = F.mse_loss(pred[:,3], y[:,3])
        
        self.log_dict({ 'test/loss/general_loss': general_loss, 'test/loss/ball_x_loss': ball_x_loss, 
                    'test/loss/ball_y_loss': ball_y_loss, 
                    'test/loss/robot_x_loss': robot_x_loss, 'test/loss/robot_y_loss': robot_y_loss })

        dict_to_return = {
            'loss': general_loss,
            'ball_x_loss': ball_x_loss,
            'ball_y_loss': ball_y_loss,
            'robot_x_loss': robot_x_loss,
            'robot_y_loss': robot_y_loss
        }

        return dict_to_return
    
    def _validation_step(self):

        self.count += 1
        
        general_loss = 0
        ball_x_loss = 0
        ball_y_loss = 0
        robot_x_loss = 0
        robot_y_loss = 0

        self.model.eval()

        with th.no_grad():
            for i in self.indexes:
                data = np.loadtxt(open('../all_data/data-v4-val' + f'/positions-{i}.txt'), dtype=np.float32)
                out_of_data_val = create_window(data, self.window)
                locale_general = 0
                locale_ball_x = 0
                locale_ball_y = 0
                locale_robot_x = 0
                locale_robot_y = 0

                self.initialize_hidden()
                for seq, y_true in out_of_data_val:
                    preds = []
                    seq = th.FloatTensor(seq)
                    
                    seq = seq.view((1,1,seq.shape[0]*seq.shape[1])).float()
                    seq = seq.to('cuda')
                
                    y_true = th.FloatTensor(y_true).to('cuda')

                    preds = self.forward(seq)
                    preds = preds.squeeze()
                    y_true = y_true.squeeze()
                    
                    locale_general += F.mse_loss(preds, y_true)
                    locale_ball_x += F.mse_loss(preds[0], y_true[0]) 
                    locale_ball_y += F.mse_loss(preds[1], y_true[1]) 
                    locale_robot_x += F.mse_loss(preds[4], y_true[4]) 
                    locale_robot_y += F.mse_loss(preds[5], y_true[5]) 
                
                locale_general /= len(out_of_data_val)
                locale_ball_x /= len(out_of_data_val)
                locale_ball_y /= len(out_of_data_val)
                locale_robot_x /= len(out_of_data_val)
                locale_robot_y /= len(out_of_data_val)

                general_loss += locale_general 
                ball_x_loss += locale_ball_x 
                ball_y_loss += locale_ball_y 
                robot_x_loss += locale_robot_x 
                robot_y_loss += locale_robot_y

            general_loss /= len(self.indexes)
            ball_x_loss /= len(self.indexes)
            ball_y_loss /= len(self.indexes)
            robot_x_loss /= len(self.indexes)
            robot_y_loss /= len(self.indexes)

            self.log_dict({ 'val/loss/general_loss': general_loss, 'val/loss/ball_x_loss': ball_x_loss, 
            'val/loss/ball_y_loss': ball_y_loss, 
            'val/loss/robot_x_loss': robot_x_loss, 'val/loss/robot_y_loss': robot_y_loss })

            dict_to_return = {
                'loss': general_loss,
                'ball_x_loss': ball_x_loss,
                'ball_y_loss': ball_y_loss,
                'robot_x_loss': robot_x_loss,
                'robot_y_loss': robot_y_loss
            }

            del preds
            del data
            del y_true
            del out_of_data_val

            return dict_to_return

    def validation_step(self, batch, batch_idx):
        
        self.initialize_hidden()
        self.count = 0
        self.model.train()
        
        X, y = batch

        X = X.view((X.shape[0], 1 ,X.shape[1] * X.shape[2])).float()
        pred = self.forward(X)
        pred = pred.squeeze()

        y = y.squeeze()

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
        
        robot_yellow_x_loss = F.mse_loss(pred[:,11], y[:,11])
        robot_yellow_y_loss = F.mse_loss(pred[:,12], y[:,12])
        robot_yellow_vx_loss = F.mse_loss(pred[:,13], y[:,13])
        robot_yellow_vy_loss = F.mse_loss(pred[:,14], y[:,14])
        robot_yellow_vw_loss = F.mse_loss(pred[:,15], y[:,15])


        self.log_dict({ 
                        'val/loss/general_loss': general_loss, 
                        'val/loss/ball_x_loss': ball_x_loss, 
                        'val/loss/ball_y_loss': ball_y_loss,
                        'val/loss/ball_vx_loss': ball_vx_loss,
                        'val/loss/ball_vy_loss': ball_vy_loss,
                        'val/loss/robot_blue_x_loss': robot_blue_x_loss, 
                        'val/loss/robot_blue_y_loss': robot_blue_y_loss,
                        'val/loss/robot_blue_sin_loss': robot_blue_sin_loss,
                        'val/loss/robot_blue_cos_loss': robot_blue_cos_loss,
                        'val/loss/robot_blue_vx_loss': robot_blue_vx_loss,
                        'val/loss/robot_blue_vy_loss': robot_blue_vy_loss,
                        'val/loss/robot_blue_vw_loss': robot_blue_vw_loss,
                        'val/loss/robot_yellow_x_loss': robot_yellow_x_loss,
                        'val/loss/robot_yellow_y_loss': robot_yellow_y_loss,
                        'val/loss/robot_yellow_vx_loss': robot_yellow_vx_loss,
                        'val/loss/robot_yellow_vy_loss': robot_yellow_vy_loss,
                        'val/loss/robot_yellow_vw_loss': robot_yellow_vw_loss,
                         })

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
