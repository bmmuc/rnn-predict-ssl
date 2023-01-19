import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
import pytorch_lightning as pl
from src.concat_data_set_autoencoder import ConcatDataSetAutoencoder
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F
import ipdb
from src.concat_data_set import ConcatDataSet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_hidden(x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True):
    """
    Initialize hidden.
    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)
    return Variable(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)


###########################################################################
################################ ENCODERS #################################
###########################################################################

class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, seq_len: int):
        """
        Initialize the model.
        Args:
            config:
            input_size: (int): size of the input
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input_data: torch.Tensor):
        """
        Run forward computation.
        Args:
            input_data: (torch.Tensor): tensor of input daa
        """
        h_t, c_t = (init_hidden(input_data, self.hidden_size),
                    init_hidden(input_data, self.hidden_size))
        input_encoded = Variable(torch.zeros(
            input_data.size(0), self.seq_len, self.hidden_size))

        for t in range(self.seq_len):
            _, (h_t, c_t) = self.lstm(
                input_data[:, t, :].unsqueeze(0), (h_t, c_t))
            input_encoded[:, t, :] = h_t
        return _, input_encoded


class AttnEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, seq_len: int, add_noise: bool = False):
        """
        Initialize the network.
        Args:
            config:
            input_size: (int): size of the input
        """
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.add_noise = add_noise
        self.directions = 1
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1
        )
        self.attn = nn.Linear(
            in_features=2 * self.hidden_size + self.seq_len,
            out_features=1
        )
        self.softmax = nn.Softmax(dim=1)

    def _get_noise(self, input_data: torch.Tensor, sigma=0.01, p=0.0):
        """
        Get noise.
        Args:
            input_data: (torch.Tensor): tensor of input data
            sigma: (float): variance of the generated noise
            p: (float): probability to add noise
        """
        normal = sigma * torch.randn(input_data.shape)
        mask = np.random.uniform(size=(input_data.shape))
        mask = (mask < p).astype(int)
        noise = normal * torch.tensor(mask)
        return noise

    def forward(self, input_data: torch.Tensor):
        """
        Forward computation.
        Args:
            input_data: (torch.Tensor): tensor of input data
        """
        # ipdb.set_trace()
        h_t, c_t = (init_hidden(input_data, self.hidden_size, num_dir=self.directions),
                    init_hidden(input_data, self.hidden_size, num_dir=self.directions))

        attentions, input_encoded = (Variable(torch.zeros(input_data.size(0), self.seq_len, self.input_size)),
                                     Variable(torch.zeros(input_data.size(0), self.seq_len, self.hidden_size)))

        if self.add_noise and self.training:
            input_data += self._get_noise(input_data).to(device)

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1).to(device)), dim=2).to(
                device)  # bs * input_size * (2 * hidden_dim + seq_len)

            # (bs * input_size) * 1
            e_t = self.attn(x.view(-1, self.hidden_size * 2 + self.seq_len))
            a_t = self.softmax(e_t.view(-1, self.input_size)
                               ).to(device)  # (bs, input_size)

            weighted_input = torch.mul(
                a_t, input_data[:, t, :].clone())  # (bs * input_size)
            self.lstm.flatten_parameters()
            # weighted_input = weighted_input.to(device)
            _, (h_t, c_t) = self.lstm(weighted_input.unsqueeze(0), (h_t, c_t))

            input_encoded[:, t, :] = h_t
            attentions[:, t, :] = a_t

        return attentions, input_encoded


###########################################################################
################################ DECODERS #################################
###########################################################################

class Decoder(nn.Module):
    def __init__(self, hidden_size: int, seq_len: int, output_size: int):
        """
        Initialize the network.
        Args:
            config:
        """
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(1, hidden_size, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _, y_hist: torch.Tensor):
        """
        Forward pass
        Args:
            _:
            y_hist: (torch.Tensor): shifted target
        """
        h_t, c_t = (init_hidden(y_hist, self.hidden_size),
                    init_hidden(y_hist, self.hidden_size))

        for t in range(self.seq_len):
            inp = y_hist[:, t].unsqueeze(0).unsqueeze(2)
            lstm_out, (h_t, c_t) = self.lstm(inp, (h_t, c_t))
        return self.fc(lstm_out.squeeze(0))


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size_decoder: int, hidden_size_encoder: int, seq_len: int, output_size: int, add_noise: bool = False):
        """
        Initialize the network.
        Args:
            config:
        """

        super(AttnDecoder, self).__init__()
        self.seq_len = seq_len
        self.encoder_hidden_size = hidden_size_encoder
        self.decoder_hidden_size = hidden_size_decoder
        self.out_feats = output_size

        self.attn = nn.Sequential(
            nn.Linear(2 * self.decoder_hidden_size +
                      self.encoder_hidden_size, self.encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_size, 1)
        )
        self.lstm = nn.LSTM(input_size=self.out_feats,
                            hidden_size=self.decoder_hidden_size)
        self.fc = nn.Linear(self.encoder_hidden_size +
                            self.out_feats, self.out_feats)
        self.fc_out = nn.Linear(
            self.decoder_hidden_size + self.encoder_hidden_size, self.out_feats * self.seq_len)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded: torch.Tensor, y_history: torch.Tensor):
        """
        Perform forward computation.
        Args:
            input_encoded: (torch.Tensor): tensor of encoded input
            y_history: (torch.Tensor): shifted target
        """
        # ipdb.set_trace()
        h_t, c_t = (
            init_hidden(input_encoded, self.decoder_hidden_size), init_hidden(input_encoded, self.decoder_hidden_size))
        context = Variable(torch.zeros(
            input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           input_encoded.to(device)), dim=2)

            x = tf.softmax(
                self.attn(
                    x.view(-1, 2 * self.decoder_hidden_size +
                           self.encoder_hidden_size)
                ).view(-1, self.seq_len),
                dim=1)

            context = torch.bmm(x.unsqueeze(1), input_encoded.to(device))[
                :, 0, :]  # (batch_size, encoder_hidden_size)

            y_tilde = self.fc(torch.cat((context.to(device), y_history[:, t].to(device)),
                                        dim=1))  # (batch_size, out_size)

            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(y_tilde.unsqueeze(0), (h_t, c_t))

        # predicting value at t=self.seq_length+1
        out = self.fc_out(torch.cat((h_t[0], context.to(device)), dim=1))
        out = out.view(-1, self.seq_len, self.out_feats)

        return out


class ActAutoEncoder(pl.LightningModule):
    def __init__(self, window=10, input_size=10, hidden_size=256,
                 batch_size=32,
                 data_root='',
                 num_of_data_to_train=0, num_of_data_to_val=0,
                 output_size=20,
                 should_test_the_new_data_set=False):
        super().__init__()
        self.batch_size = batch_size

        self.encoder = AttnEncoder(input_size, hidden_size, window, True)

        self.decoder = AttnDecoder(hidden_size, hidden_size, window,
                                   output_size, True)

        self.data_root = data_root
        self.should_test_the_new_data_set = should_test_the_new_data_set
        self.window = window
        self.batch_size = batch_size
        self.num_of_data_to_train = num_of_data_to_train
        self.num_of_data_to_val = num_of_data_to_val
        self.automatic_optimization = False
        self.save_hyperparameters()

        self = self.to(torch.device('cuda'))

        self.opt = self.configure_optimizers()
        self.indexes = [8, 9, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 32,33, 34, 37, 38, 39]
        self.should_test_overffit = False

    def train_dataloader(self):
        dataset = ConcatDataSetAutoencoder(
            root_dir=self.data_root,
            num_of_data_sets=self.num_of_data_to_train,
            window=self.window,
            should_test_the_new_data_set=self.should_test_the_new_data_set,
            type_of_data='train',
            horizon=1
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
        dataset = ConcatDataSetAutoencoder(
            root_dir=self.data_root,
            num_of_data_sets=self.num_of_data_to_train,
            window=self.window,
            should_test_the_new_data_set=self.should_test_the_new_data_set,
            type_of_data='val',
            horizon=1
        )

        loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=1,
                    pin_memory=True
                )

        return loader

        # dataset = ConcatDataSetAutoencoder(
        #     root_dir=self.data_root + '-val',
        #     num_of_data_sets=self.num_of_data_to_val,
        #     window=self.window,
        #     should_test_the_new_data_set=self.should_test_the_new_data_set,
        #     type_of_data='val',
        #     horizon=1
        # )

        # loader = DataLoader(
        #     dataset,
        #     batch_size=self.batch_size,
        #     num_workers=1,
        #     pin_memory=True
        # )

        # return loader

    def forward(self, x):
        # ipdb.set_trace()
        x = x[:, :, self.indexes]
        att, encoded = self.encoder(x)

        encoded = encoded.to(torch.device('cuda'))

        decoded = self.decoder(encoded, x)
        # decoded = decoded.view(decoded.shape[0], decoded.shape[2], decoded.shape[1])

        return encoded, decoded[:,:,:]
        # return encoded, decoded

    def encoding(self, x, is_traning = True):
        # ipdb.set_trace()
        if is_traning:
            x = x[:, :, self.indexes]

        att, encoded = self.encoder(x)
        return encoded

    def decoding(self, x, y_hist):
        decoded = self.decoder(x, y_hist)
        return decoded

    def training_step(self, batch, batch_idx):
        self.train()

        if self.global_step == 0:
            gen = self.validation_step(batch, batch_idx)
            return gen

        X, y = batch
        # ipdb.set_trace()

        _, pred = self.forward(X)

        pred = pred.squeeze()
        # pred = pred[:, :, -1]
        y = y.squeeze()

        y = y[:, :, self.indexes]

        general_loss = F.mse_loss(pred, y)

        self.opt.zero_grad()
        self.manual_backward(general_loss)
        self.opt.step()
        

        robot_blue_vx_loss = F.mse_loss(pred[:, :,  0], y[:, :,  0])
        robot_blue_vy_loss = F.mse_loss(pred[:, :,  1], y[:, :,  1])
        robot_blue_vw_loss = F.mse_loss(pred[:, :,  2], y[:, :,  2])


        robot_1blue_vx_loss = F.mse_loss(pred[:, :, 3], y[:, :, 3])
        robot_1blue_vy_loss = F.mse_loss(pred[:, :, 4], y[:, :, 4])
        robot_1blue_vw_loss = F.mse_loss(pred[:, :, 5], y[:, :, 5])


        robot_2blue_vx_loss = F.mse_loss(pred[:, :, 6], y[:, :, 6])
        robot_2blue_vy_loss = F.mse_loss(pred[:, :, 7], y[:, :, 7])
        robot_2blue_vw_loss = F.mse_loss(pred[:, :, 8], y[:, :, 8])

        robot_yellow_vx_loss = F.mse_loss(pred[:, :, 9], y[:, :, 9])
        robot_yellow_vy_loss = F.mse_loss(pred[:, :, 10], y[:, :, 10])
        robot_yellow_vw_loss = F.mse_loss(pred[:, :, 11], y[:, :, 11])

        robot_1yellow_vx_loss = F.mse_loss(pred[:, :, 12], y[:, :, 12])
        robot_1yellow_vy_loss = F.mse_loss(pred[:, :, 13], y[:, :, 13])
        robot_1yellow_vw_loss = F.mse_loss(pred[:, :, 14], y[:, :, 14])


        robot_2yellow_vx_loss = F.mse_loss(pred[:, :, 15], y[:, :, 15])
        robot_2yellow_vy_loss = F.mse_loss(pred[:, :, 16], y[:, :, 16])
        robot_2yellow_vw_loss = F.mse_loss(pred[:, :, 17], y[:, :, 17])

        mean_robot_vx_loss = (robot_blue_vx_loss + robot_1blue_vx_loss + robot_2blue_vx_loss \
                                 + robot_yellow_vx_loss + robot_1yellow_vx_loss + robot_2yellow_vx_loss ) / 6

        mean_robot_vy_loss = (robot_blue_vy_loss + robot_1blue_vy_loss + robot_2blue_vy_loss \
                                    + robot_yellow_vy_loss + robot_1yellow_vy_loss + robot_2yellow_vy_loss ) / 6

        mean_robot_vw_loss = (robot_blue_vw_loss + robot_1blue_vw_loss + robot_2blue_vw_loss \
                                    + robot_yellow_vw_loss + robot_1yellow_vw_loss + robot_2yellow_vw_loss ) / 6
        

        self.log_dict({ 'train/loss/general_vel_pos': general_loss, 
                        'train/loss/mean_robot_vx_loss': mean_robot_vx_loss,
                        'train/loss/mean_robot_vy_loss': mean_robot_vy_loss,
                        'train/loss/mean_robot_vw_loss': mean_robot_vw_loss,
                        'train/loss/robot_0_blue_vx_loss': robot_blue_vx_loss,
                        'train/loss/robot_0_blue_vy_loss': robot_blue_vy_loss,
                        'train/loss/robot_0_blue_vw_loss': robot_blue_vw_loss,
                        'train/loss/robot_1_blue_vx_loss': robot_1blue_vx_loss,
                        'train/loss/robot_1_blue_vy_loss': robot_1blue_vy_loss,
                        'train/loss/robot_1_blue_vw_loss': robot_1blue_vw_loss,
                        'train/loss/robot_2_blue_vx_loss': robot_2blue_vx_loss,
                        'train/loss/robot_2_blue_vy_loss': robot_2blue_vy_loss,
                        'train/loss/robot_2_blue_vw_loss': robot_2blue_vw_loss,
                        'train/loss/robot_0_yellow_vx_loss': robot_yellow_vx_loss,
                        'train/loss/robot_0_yellow_vy_loss': robot_yellow_vy_loss,
                        'train/loss/robot_0_yellow_vw_loss': robot_yellow_vw_loss,
                        'train/loss/robot_1_yellow_vx_loss': robot_1yellow_vx_loss,
                        'train/loss/robot_1_yellow_vy_loss': robot_1yellow_vy_loss,
                        'train/loss/robot_1_yellow_vw_loss': robot_1yellow_vw_loss,
                        'train/loss/robot_2_yellow_vx_loss': robot_2yellow_vx_loss,
                        'train/loss/robot_2_yellow_vy_loss': robot_2yellow_vy_loss,
                        'train/loss/robot_2_yellow_vw_loss': robot_2yellow_vw_loss,
                        })

        return general_loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        X, y = batch
        # ipdb.set_trace()

        _, pred = self.forward(X)

        pred = pred.squeeze()
        # pred = pred[:, :, -1]
        y = y.squeeze()

        y = y[:, :, self.indexes]

        general_loss = F.mse_loss(pred, y)

        robot_blue_vx_loss = F.mse_loss(pred[:, :,  0], y[:, :,  0])
        robot_blue_vy_loss = F.mse_loss(pred[:, :,  1], y[:, :,  1])
        robot_blue_vw_loss = F.mse_loss(pred[:, :,  2], y[:, :,  2])


        robot_1blue_vx_loss = F.mse_loss(pred[:, :, 3], y[:, :, 3])
        robot_1blue_vy_loss = F.mse_loss(pred[:, :, 4], y[:, :, 4])
        robot_1blue_vw_loss = F.mse_loss(pred[:, :, 5], y[:, :, 5])


        robot_2blue_vx_loss = F.mse_loss(pred[:, :, 6], y[:, :, 6])
        robot_2blue_vy_loss = F.mse_loss(pred[:, :, 7], y[:, :, 7])
        robot_2blue_vw_loss = F.mse_loss(pred[:, :, 8], y[:, :, 8])

        robot_yellow_vx_loss = F.mse_loss(pred[:, :, 9], y[:, :, 9])
        robot_yellow_vy_loss = F.mse_loss(pred[:, :, 10], y[:, :, 10])
        robot_yellow_vw_loss = F.mse_loss(pred[:, :, 11], y[:, :, 11])

        robot_1yellow_vx_loss = F.mse_loss(pred[:, :, 12], y[:, :, 12])
        robot_1yellow_vy_loss = F.mse_loss(pred[:, :, 13], y[:, :, 13])
        robot_1yellow_vw_loss = F.mse_loss(pred[:, :, 14], y[:, :, 14])


        robot_2yellow_vx_loss = F.mse_loss(pred[:, :, 15], y[:, :, 15])
        robot_2yellow_vy_loss = F.mse_loss(pred[:, :, 16], y[:, :, 16])
        robot_2yellow_vw_loss = F.mse_loss(pred[:, :, 17], y[:, :, 17])

        mean_robot_vx_loss = (robot_blue_vx_loss + robot_1blue_vx_loss + robot_2blue_vx_loss \
                                 + robot_yellow_vx_loss + robot_1yellow_vx_loss + robot_2yellow_vx_loss ) / 6

        mean_robot_vy_loss = (robot_blue_vy_loss + robot_1blue_vy_loss + robot_2blue_vy_loss \
                                    + robot_yellow_vy_loss + robot_1yellow_vy_loss + robot_2yellow_vy_loss ) / 6
        
        mean_robot_vw_loss = (robot_blue_vw_loss + robot_1blue_vw_loss + robot_2blue_vw_loss \
                                    + robot_yellow_vw_loss + robot_1yellow_vw_loss + robot_2yellow_vw_loss ) / 6
        

        self.log_dict({ 'val/loss/general_vel_pos': general_loss, 
                        'val/loss/mean_robot_vx_loss': mean_robot_vx_loss,
                        'val/loss/mean_robot_vy_loss': mean_robot_vy_loss,
                        'val/loss/mean_robot_vw_loss': mean_robot_vw_loss,
                        'val/loss/robot_0_blue_vx_loss': robot_blue_vx_loss,
                        'val/loss/robot_0_blue_vy_loss': robot_blue_vy_loss,
                        'val/loss/robot_0_blue_vw_loss': robot_blue_vw_loss,
                        'val/loss/robot_1_blue_vx_loss': robot_1blue_vx_loss,
                        'val/loss/robot_1_blue_vy_loss': robot_1blue_vy_loss,
                        'val/loss/robot_1_blue_vw_loss': robot_1blue_vw_loss,
                        'val/loss/robot_2_blue_vx_loss': robot_2blue_vx_loss,
                        'val/loss/robot_2_blue_vy_loss': robot_2blue_vy_loss,
                        'val/loss/robot_2_blue_vw_loss': robot_2blue_vw_loss,
                        'val/loss/robot_0_yellow_vx_loss': robot_yellow_vx_loss,
                        'val/loss/robot_0_yellow_vy_loss': robot_yellow_vy_loss,
                        'val/loss/robot_0_yellow_vw_loss': robot_yellow_vw_loss,
                        'val/loss/robot_1_yellow_vx_loss': robot_1yellow_vx_loss,
                        'val/loss/robot_1_yellow_vy_loss': robot_1yellow_vy_loss,
                        'val/loss/robot_1_yellow_vw_loss': robot_1yellow_vw_loss,
                        'val/loss/robot_2_yellow_vx_loss': robot_2yellow_vx_loss,
                        'val/loss/robot_2_yellow_vy_loss': robot_2yellow_vy_loss,
                        'val/loss/robot_2_yellow_vw_loss': robot_2yellow_vw_loss,
                        })

        return general_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = 1e-4)
