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

    def _get_noise(self, input_data: torch.Tensor, sigma=0.01, p=0.3):
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
                a_t, input_data[:, t, :].to(device))  # (bs * input_size)
            self.lstm.flatten_parameters()
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

class VaeAutoencoder(pl.LightningModule):
    def __init__(self, input_size: int = 40, hidden_size: int = 128,
                 window= 10, data_root ='', num_of_data_to_train = 256,
                 should_test_the_new_data_set = False, num_of_data_to_val = 128,
                 batch_size = 256, is_to_pred_pos = False, is_to_pred_vel = False,
                 seq_len: int = 10, output_size: int = 41, noise = False):
        """
        Initialize the network.
        Args:
            config:
        """

        super(VaeAutoencoder, self).__init__()
        
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.window = window
        self.input_size = input_size
        self.output_size = output_size
        self.data_root = data_root

        self.is_to_pred_pos = is_to_pred_pos
        self.is_to_pred_vel = is_to_pred_vel

        self.num_of_data_to_train = num_of_data_to_train
        self.num_of_data_to_val = num_of_data_to_val
        self.batch_size = batch_size
        self.should_test_the_new_data_set = should_test_the_new_data_set

        self.encoder = AttnEncoder(input_size, hidden_size, window, True)
        self.decoder = AttnDecoder(
            hidden_size, hidden_size, window, input_size, True)

        self.model = nn.ModuleDict({
            'fc_mu': nn.Linear(hidden_size, hidden_size),
            'fc_var': nn.Linear(hidden_size, hidden_size),
        })

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.automatic_optimization = False
        self.save_hyperparameters()
        self.opt = self.configure_optimizers()
        self.to(device)
        self.encoder.to(device)
        self.decoder.to(device)

    def train_dataloader(self):

        dataset = ConcatDataSetAutoencoder(
            root_dir=self.data_root,
            num_of_data_sets=self.num_of_data_to_train,
            window=self.window,
            should_test_the_new_data_set=self.should_test_the_new_data_set,
            type_of_data='train',
            horizon=1
        )

        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=6,
            pin_memory=True
        )

        return loader

    def val_dataloader(self):
        dataset = ConcatDataSetAutoencoder(
            root_dir=self.data_root + '-val',
            num_of_data_sets=self.num_of_data_to_val,
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

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=[1,2])

    def kl_divergence(self, z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)


        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(self, x):
        """
        Perform forward computation.
        Args:
            x: (torch.Tensor): input
            y_hist: (torch.Tensor): shifted target
        """
        x = x.to(device)
        _, x_encoded = self.encoder(x)
        # ipdb.set_trace()
        x_encoded = x_encoded.to(device)
        mu = self.model['fc_mu'](x_encoded)
        log_var = self.model['fc_var'](x_encoded)
        
        std = torch.exp(0.5 * log_var)
        q = torch.distributions.Normal(mu, std)

        z = q.rsample()

        x_hat = self.decoder(z, x)

        reccon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        kl = self.kl_divergence(z, mu, std)

        elbo = reccon_loss - kl

        elbo = elbo.mean()

        return elbo, x_hat

    def training_step(self, batch, batch_idx):
        self.train()
        X, y = batch

        # X = X.view((X.shape[0], 1 ,X.shape[1] * X.shape[2])).float()
        elbo, x_hat = self.forward(X)
        pred = x_hat.squeeze()
        y = y.squeeze()

        general_loss = F.mse_loss(pred, y)

        self.opt.zero_grad()
        self.manual_backward(elbo)
        self.opt.step()

        

        ball_x_loss = F.mse_loss(pred[:, :, 0], y[:, :, 0])
        ball_y_loss = F.mse_loss(pred[:, :, 1], y[:, :, 1])
        ball_vx_loss = F.mse_loss(pred[:, :, 2], y[:, :, 2])
        ball_vy_loss = F.mse_loss(pred[:, :, 3], y[:, :, 3])
        
        robot_blue_x_loss = F.mse_loss(pred[:, :, 4], y[:, :, 4])
        robot_blue_y_loss = F.mse_loss(pred[:, :, 5], y[:, :, 5])

        robot_blue_sin_loss = F.mse_loss(pred[:, :, 6], y[:, :, 6])
        robot_blue_cos_loss = F.mse_loss(pred[:, :, 7], y[:, :, 7])

        robot_blue_vx_loss = F.mse_loss(pred[:, :, 8], y[:, :, 8])
        robot_blue_vy_loss = F.mse_loss(pred[:, :, 9], y[:, :, 9])
        robot_blue_vw_loss = F.mse_loss(pred[:, :, 10], y[:, :, 10])

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

        reward_loss = F.mse_loss(pred[:, :, 40], y[:,:,40])


        self.log_dict({ 'train/loss/general_loss': general_loss,
                        'train/loss/elbo': elbo,
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

        return {'loss': general_loss}

    def validation_step(self, batch, batch_idx):
        self.eval()
        X, y = batch
        # ipdb.set_trace()
        # X = X.view((X.shape[0], 1 ,X.shape[1] * X.shape[2])).float()
        elbo, x_hat = self.forward(X)
        pred = x_hat.squeeze()
        y = y.squeeze()
        # ipdb.set_trace()

        general_loss = F.mse_loss(pred, y)

        ball_x_loss = F.mse_loss(pred[:, :, 0], y[:, :, 0])
        ball_y_loss = F.mse_loss(pred[:, :, 1], y[:, :, 1])
        ball_vx_loss = F.mse_loss(pred[:, :, 2], y[:, :, 2])
        ball_vy_loss = F.mse_loss(pred[:, :, 3], y[:, :, 3])

        robot_blue_x_loss = F.mse_loss(pred[:, :, 4], y[:, :, 4])
        robot_blue_y_loss = F.mse_loss(pred[:, :, 5], y[:, :, 5])

        robot_blue_sin_loss = F.mse_loss(pred[:, :, 6], y[:, :, 6])
        robot_blue_cos_loss = F.mse_loss(pred[:, :, 7], y[:, :, 7])

        robot_blue_vx_loss = F.mse_loss(pred[:, :, 8], y[:, :, 8])
        robot_blue_vy_loss = F.mse_loss(pred[:, :, 9], y[:, :, 9])
        robot_blue_vw_loss = F.mse_loss(pred[:, :, 10], y[:, :, 10])

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

        # ipdb.set_trace()

        reward_loss = F.mse_loss(pred[:,:,40], y[:,:,40])

        # robot_yellow_x_loss = F.mse_loss(pred[:,:,11], y[:,:,11])
        # robot_yellow_y_loss = F.mse_loss(pred[:,:,12], y[:,:,12])
        # robot_yellow_vx_loss = F.mse_loss(pred[:,:,13], y[:,:,13])
        # robot_yellow_vy_loss = F.mse_loss(pred[:,:,14], y[:,:,14])
        # robot_yellow_vw_loss = F.mse_loss(pred[:,:,15], y[:,:,15])



        self.log_dict({ 'val/loss/general_loss': general_loss, 
                        'train/loss/elbo': elbo,
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
                        'val/loss/reward_loss': reward_loss,
                        })

        return {'val/loss/general_loss': general_loss}

    def configure_optimizers(self):
        return Adam(self.parameters())
