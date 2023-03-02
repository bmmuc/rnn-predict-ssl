from copy import copy, deepcopy
import numpy as np
import torch
from torch import nn, no_grad
from torch.autograd import Variable
from torch.nn import functional as tf
import pytorch_lightning as pl
# from src.concat_data_set_autoencoder import ConcatDataSetAutoencoder
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import functional as F
from src.act_autoencoder import ActAutoEncoder
from src.pos_autoencoder import PositionAutoEncoder
import ipdb
from src.concat_data_set import ConcatDataSet
# from render import RCGymRender
import PIL
# import random
# from src.concat_autoencoder import ConcatAutoEncoder
# from handle_render import Handle_render

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


class PosLatent(pl.LightningModule):
    def __init__(self, window=10, input_size=10, hidden_size=256,
                 batch_size=32,
                 data_root='',
                 num_of_data_to_train=0, num_of_data_to_val=0,
                 output_size=20,
                 weights=(0.5, 0.5),
                 should_test_the_new_data_set=False,
                 act_path='',
                 pos_path='',
                 act_path_autoencoder='',
                 lr=1e-3,
                 ):
        super().__init__()
        self.batch_size = batch_size

        self.data_root = data_root
        self.should_test_the_new_data_set = should_test_the_new_data_set
        self.window = window
        self.output_size = output_size
        self.lr = lr
        self.num_of_data_to_train = num_of_data_to_train
        self.num_of_data_to_val = num_of_data_to_val
        self.automatic_optimization = False

        # self.encoder = ConcatAutoEncoder(window=window, input_size=input_size, hidden_size=hidden_size, output_size=output_size,
        # pos_path = './autoencoder/rnn-autoencoder-v3/2maykc2c/checkpoints/epoch=255-step=60927.ckpt',
        # act_path_autoencoder='./next_positions/rnn-predict-next-positions-with-v3/2a5vkx8o/checkpoints/epoch=255-step=60927.ckpt',)
        # self.encoder = self.encoder.load_from_checkpoint(act_path_autoencoder)

        self.pos_autoencoder = PositionAutoEncoder(
            window=window, input_size=input_size, hidden_size=hidden_size, output_size=22)
        self.pos_autoencoder = self.pos_autoencoder.load_from_checkpoint(
            pos_path)

        # self.act_forecast = ActAutoEncoder(window=self.window, input_size=input_size, hidden_size=hidden_size, output_size=18)
        # self.act_forecast = self.act_forecast.load_from_checkpoint(act_path)

        self.act_autoencoder = ActAutoEncoder(
            window=self.window, input_size=input_size, hidden_size=hidden_size, output_size=18)
        self.act_autoencoder = self.act_autoencoder.load_from_checkpoint(
            act_path)

        self.pos_autoencoder = self.pos_autoencoder.eval()
        self.act_autoencoder = self.act_autoencoder.eval()

        self.pred_pos = nn.ModuleDict({
            'linear1': nn.Linear(512, 1024),
            'linear2': nn.Linear(1024, 512),
            'linear3': nn.Linear(512, 512),
            'linear4': nn.Linear(512, 512),
            'dropout1': nn.Dropout(0.25),
            'linear5': nn.Linear(512, 512),
            'linear8': nn.Linear(512, 256),
            'linear9': nn.Linear(256, 256),
            'linear10': nn.Linear(256, 256),
            'dropout2': nn.Dropout(0.15),
            'linear11': nn.Linear(256, 256),
            'linear12': nn.Linear(256, 256),
            'linear13': nn.Linear(256, 256),
            'linear14': nn.Linear(256, 256),
            'linear15': nn.Linear(256, 256),
            'dropout3': nn.Dropout(0.5),
            'linear16': nn.Linear(256, 256),

        })

        self.pred_act = nn.ModuleDict({
            'linear1': nn.Linear(256, 1024),  # estrutura pos => (512, 1024)
            # 'linear1': nn.Linear(512, 1024), # estrutura pos + act => (512, 1024)
            'linear2': nn.Linear(1024, 512),
            'linear3': nn.Linear(512, 512),
            'linear4': nn.Linear(512, 512),
            'linear5': nn.Linear(512, 256),
            'linear6': nn.Linear(256, 256),
            'linear7': nn.Linear(256, 256),
            'linear8': nn.Linear(256, 256),
            'linear9': nn.Linear(256, 256),
            'linear10': nn.Linear(256, 256),
        })

        self.opt = self.configure_optimizers()  # testar 2 optimizers

        self.weitght_pos = weights[0]
        self.weitght_acts = weights[1]

        self.save_hyperparameters()
        self = self.to(torch.device('cuda'))

        torch.autograd.set_detect_anomaly(True)
        self.render = True

        self.indexes = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12,
                        13, 14, 18, 19, 20, 21, 25, 26, 30, 31, 35, 36]
        self.indexes_act = [8, 9, 10, 15, 16, 17, 22,
                            23, 24, 27, 28, 29, 32, 33, 34, 37, 38, 39]

        self.should_test_overffit = False

    def train_dataloader(self):
        dataset = ConcatDataSet(
            root_dir=self.data_root,
            num_of_data_sets=self.num_of_data_to_train,
            window=self.window,
            type_of_data='train',
            should_test_the_new_data_set=self.should_test_the_new_data_set,
            should_test_overffit=self.should_test_overffit,
            horizon=1
        )

        # self.num_of_windows = dataset.num_of_windows

        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=6,
            pin_memory=True
        )

        return loader

    def test_dataloader(self):
        dataset = ConcatDataSet(
            root_dir=self.data_root + '-val',
            num_of_data_sets=self.num_of_data_to_val,
            window=self.window,
            type_of_data='val',
            should_test_the_new_data_set=self.should_test_the_new_data_set,
            horizon=1
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True
        )

        return loader

    def run_pos(self, act_pos_hidden_concat):
        with torch.no_grad():

            out = self.pred_pos['linear1'](act_pos_hidden_concat)
            out = self.pred_pos['linear2'](out)
            out = self.pred_pos['linear3'](out)
            out = self.pred_pos['linear4'](out)
            out = self.pred_pos['dropout1'](out)
            out = self.pred_pos['linear5'](out)
            out = self.pred_pos['linear8'](out)
            out = self.pred_pos['linear9'](out)
            out = self.pred_pos['linear10'](out)
            out = self.pred_pos['dropout2'](out)
            out = self.pred_pos['linear11'](out)
            out = self.pred_pos['linear12'](out)
            out = self.pred_pos['linear13'](out)
            out = self.pred_pos['linear14'](out)
            out = self.pred_pos['linear15'](out)

            out_pred_pos = self.pred_pos['linear16'](out)

        return out_pred_pos

    def run_act(self, act_encoded):
        with torch.no_grad():

            out2 = self.pred_act['linear1'](act_encoded)
            out2 = self.pred_act['linear2'](out2)
            out2 = self.pred_act['linear3'](out2)
            out2 = self.pred_act['linear4'](out2)
            out2 = self.pred_act['linear5'](out2)
            out2 = self.pred_act['linear6'](out2)
            out2 = self.pred_act['linear7'](out2)
            out2 = self.pred_act['linear8'](out2)
            out2 = self.pred_act['linear9'](out2)
            out_pred_act = self.pred_act['linear10'](out2)

        return out_pred_act

    def predict_n_steps(self, x_hist, n_steps):
        # ipdb.set_trace()
        with torch.no_grad():
            x_hist = torch.FloatTensor(x_hist).to(torch.device('cuda'))
            x_hist = x_hist.view(1, x_hist.shape[0], x_hist.shape[1])
            y_hist = x_hist.clone()

            y_hist_pos = y_hist[:, :, self.indexes]
            y_hist_act = y_hist[:, :, self.indexes_act]

            act_encoded = self.act_autoencoder.encoding(x_hist)

            pos_encoded = self.pos_autoencoder.encoding(x_hist)

            act_pos_hidden_concat = torch.cat(
                (act_encoded, pos_encoded), dim=2)

            act_pos_hidden_concat = act_pos_hidden_concat.to(
                torch.device('cuda'))
            act_encoded = act_encoded.to(torch.device('cuda'))

            # next_act_hidden = act_encoded.clone()

            # for i in range(n_steps):
            #         # ipdb.set_trace()
            #         next_pos_hidden = self.run_pos(act_pos_hidden_concat)
            #         next_act_hidden = self.run_act(next_act_hidden)

            #         #comentar aq
            #         # if i != n_steps - 1:
            #         y_hist_pos = self.pos_autoencoder.decoding(next_pos_hidden, y_hist_pos)
            #         y_hist_act = self.act_autoencoder.decoding(act_encoded, y_hist_act)

            #         next_act_hidden = self.act_autoencoder.encoding(y_hist_act, False).to(torch.device('cuda'))

            #         next_pos_hidden = self.pos_autoencoder.encoding(y_hist_pos, False).to(torch.device('cuda'))

            #         act_pos_hidden_concat = torch.cat((next_act_hidden, next_pos_hidden), dim=2).to(torch.device('cuda'))

            #         # act_pos_hidden_concat = torch.cat((next_act_hidden, next_pos_hidden), dim=2)
            #         # act_encoded = next_act_hidden
            #         # ate aq

            for i in range(n_steps):
                next_pos_hidden = self.run_pos(act_pos_hidden_concat)
                # -> usar quando for a estrutura pos + act
                next_act_hidden = self.run_act(act_pos_hidden_concat)
                # next_act_hidden = self.run_act(act_encoded)

                act_pos_hidden_concat = torch.cat(
                    (next_act_hidden, next_pos_hidden), dim=2)
                act_encoded = next_act_hidden

                # y_hist_pos = self.pos_autoencoder.decoding(next_pos_hidden, y_hist_pos)
                # y_hist_act = self.act_autoencoder.decoding(act_encoded, y_hist_act)

            pos_decoded = self.pos_autoencoder.decoding(
                next_pos_hidden, y_hist_pos)
            act_decoded = self.act_autoencoder.decoding(
                next_act_hidden, y_hist_act)

        return pos_decoded, act_decoded

    def val_dataloader(self):
        dataset = ConcatDataSet(
            root_dir=self.data_root + '-val',
            num_of_data_sets=self.num_of_data_to_val,
            window=self.window,
            type_of_data='val',
            should_test_the_new_data_set=self.should_test_the_new_data_set,
            horizon=1
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True
        )

        return loader

    def forward(self, x):
        # ipdb.set_trace()
        x_copy = x.clone()

        # _, act_forecast = self.act_forecast(x)
        act_encoded = self.act_autoencoder.encoding(x)

        pos_encoded = self.pos_autoencoder.encoding(x)

        act_pos_hidden_concat = torch.cat((act_encoded, pos_encoded), dim=2)

        x_copy_pos = x_copy[:, :, self.indexes]
        x_copy_act = x_copy[:, :, self.indexes_act]

        # x_hist_copy = x_copy.clone()

        # act_forecast = act_forecast.view(act_forecast.shape[0], act_forecast.shape[2], act_forecast.shape[1])

        # x_copy = torch.cat((x_copy, act_forecast), dim=2)

        # encode = encode.to(device)
        act_pos_hidden_concat = act_pos_hidden_concat.to(torch.device('cuda'))
        act_encoded = act_encoded.to(torch.device('cuda'))

        out = self.pred_pos['linear1'](act_pos_hidden_concat)
        out = self.pred_pos['linear2'](out)
        out = self.pred_pos['linear3'](out)
        out = self.pred_pos['linear4'](out)
        out = self.pred_pos['dropout1'](out)
        out = self.pred_pos['linear5'](out)
        out = self.pred_pos['linear8'](out)
        out = self.pred_pos['linear9'](out)
        out = self.pred_pos['linear10'](out)
        out = self.pred_pos['dropout2'](out)
        out = self.pred_pos['linear11'](out)
        out = self.pred_pos['linear12'](out)
        out = self.pred_pos['linear13'](out)
        out = self.pred_pos['linear14'](out)
        out = self.pred_pos['linear15'](out)

        out_pred_pos = self.pred_pos['linear16'](out)

        out2 = self.pred_act['linear1'](act_encoded)  # -> versao sem concat
        # out2 = self.pred_act['linear1'](act_pos_hidden_concat)
        out2 = self.pred_act['linear2'](out2)
        out2 = self.pred_act['linear3'](out2)
        out2 = self.pred_act['linear4'](out2)
        out2 = self.pred_act['linear5'](out2)
        out2 = self.pred_act['linear6'](out2)
        out2 = self.pred_act['linear7'](out2)
        out2 = self.pred_act['linear8'](out2)
        out2 = self.pred_act['linear9'](out2)
        out_pred_act = self.pred_act['linear10'](out2)

        out_pred_pos = self.pos_autoencoder.decoding(out_pred_pos, x_copy_pos)

        out_pred_act = self.act_autoencoder.decoding(out_pred_act, x_copy_act)

        return out_pred_pos[:, -1, :], out_pred_act[:, -1, :]

    def encoding(self, x):
        # ipdb.set_trace()
        encoded = self.encoder.encoding(x)
        return encoded

    def get_actions(self, x):
        _, act_forecast = self.act_forecast(x)
        act_forecast = act_forecast.view(
            act_forecast.shape[0], act_forecast.shape[2], act_forecast.shape[1])

        return act_forecast

    def decoding(self, x, y_hist):
        decoded = self.encoder.decoding(x, y_hist)

        return decoded

    def training_step(self, batch, batch_idx):
        self.train()
        if self.current_epoch % 4 == 0:
            self.render = True
        # if self.global_step == 0:
        #     gen = self.validation_step(batch, batch_idx)
        #     return gen

        # ipdb.set_trace()
        X, y = batch

        y = y.squeeze()
        y_clone1 = y.clone()
        y_clone2 = y.clone()

        pred1, pred2 = self.forward(X)

        y_act = y_clone1[:, self.indexes_act].clone()
        y_pos = y_clone2[:, self.indexes].clone()

        loss_act = F.mse_loss(pred2, y_act)

        loss_pos = F.mse_loss(pred1, y_pos)

        general_loss = self.weitght_pos * loss_pos + self.weitght_acts * loss_act

        self.opt.zero_grad()
        self.manual_backward(general_loss)

        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.0175)
        ave_grads = []
        norm_grad = []
        total_norm = 0
        # layers = []
        for p in self.parameters():
            if (p.requires_grad):
                # layers.append(n)
                ave_grads.append(torch.max(torch.abs(p.grad)).item())
                param_norm = p.grad.data.norm(2)
                norm_grad.append(param_norm.item())
                total_norm += param_norm.item() ** 2
                norm_grad.append(torch.norm(p.grad).item())
        total_norm = total_norm ** (1. / 2)
        # ipdb.set_trace()
        self.opt.step()

        # pred_decoder = self.decoding(pred_copy, X)

        # pred_decoder = pred_decoder[:, -1, :]
        # y_true = y_true.squeeze()
        # general_loss_encoded = F.mse_loss(pred_decoder, y_true)
        # pred_decoder = pred_decoder.cpu().detach().numpy()
        # y_true = y_true.cpu().detach().numpy()
        # y_copy = y_copy.cpu().detach().numpy()

        self.log_dict({'train/loss/general_loss': general_loss,
                       'train/loss/loss_pos': loss_pos,
                       'train/loss/loss_act': loss_act,
                       'max_abs_gradients': max(ave_grads),
                       'max_norm_gradients': max(norm_grad),
                       'sum_norm_gradients': total_norm,
                       # 'train/loss/general_loss_encoded': general_loss_encoded,
                       })

        return general_loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        # ipdb.set_trace()

        X, y = batch

        pred1, pred2 = self.forward(X)
        pred1 = pred1.squeeze()
        pred2 = pred2.squeeze()

        y = y.squeeze()

        # y_copy = y_copy.cpu().detach().numpy()

        y_act = y[:, self.indexes_act]
        y_pos = y[:, self.indexes]

        loss_pos = F.mse_loss(pred1, y_pos)
        loss_act = F.mse_loss(pred2, y_act)

        # y = y[:, self.indexes]

        general_loss = self.weitght_pos * loss_pos + self.weitght_acts * loss_act

        # pred_decoder = self.decoding(pred_copy, X)

        # pred_decoder = pred_decoder[:, -1, :]
        # y_true = y_true.squeeze()
        # general_loss_encoded = F.mse_loss(pred_decoder, y_true)
        # pred_decoder = pred_decoder.cpu().detach().numpy()
        # y_true = y_true.cpu().detach().numpy()

        # pred_decoder = self.decoding(pred_copy, X)

        # pred_decoder = pred_decoder[:, -1, :]
        # y_true = y_true.squeeze()
        # general_loss_encoded = F.mse_loss(pred_decoder, y_true)
        # pred_decoder = pred_decoder.cpu().detach().numpy()
        # y_true = y_true.cpu().detach().numpy()
        # y_copy = y_copy.cpu().detach().numpy()

        if self.render:

            # render = RCGymRender(should_render_actual_ball = False, n_robots_blue = 3, n_robots_yellow = 3)
            Handle_render(self, False, name_predictor=f'pos_latent-eval-{self.global_step}',
                          num_of_features=self.output_size, path=f'./gifs/train/train-{self.global_step}-3v3.gif',
                          is_traning=True)\
                .render_n_steps_autoencoder3v3()

            self.render = False
            # frame = render.render_frame(
            #             # ball_actual_x = seq[-1][0], ball_actual_y = seq[-1][1],
            #             ball_true_x = y_copy[idx_to_use][0], ball_true_y = y_copy[idx_to_use][1],
            #             ball_pred_x = pred1[idx_to_use][0], ball_pred_y = pred1[idx_to_use][1],
            #             robot_actual_x= y_copy[idx_to_use][4], robot_actual_y = y_copy[idx_to_use][5],
            #             robot_blue_theta = y_copy[idx_to_use][6],
            #             robot2_actual_x= y_copy[idx_to_use][11], robot2_actual_y = y_copy[idx_to_use][12],
            #             robot2_blue_theta = y_copy[idx_to_use][13],
            #             robot3_actual_x= y_copy[idx_to_use][18], robot3_actual_y = y_copy[idx_to_use][19],
            #             robot3_blue_theta = y_copy[idx_to_use][20],
            #             robot_yellow_x= y_copy[idx_to_use][25], robot_yellow_y = y_copy[idx_to_use][26],
            #             robot_yellow_theta = y_copy[idx_to_use][27],
            #             robot2_yellow_x= y_copy[idx_to_use][30], robot2_yellow_y = y_copy[idx_to_use][31],
            #             robot2_yellow_theta = y_copy[idx_to_use][32],
            #             robot3_yellow_x= y_copy[idx_to_use][35], robot3_yellow_y = y_copy[idx_to_use][36],
            #             robot3_yellow_theta = y_copy[idx_to_use][37],
            #             return_rgb_array=True)

            # frame = PIL.Image.fromarray(frame)

            # del(render)
            # # ipdb.set_trace()

            # frame.save(
            #     fp=f'./gifs/train/global-step:{self.global_step}',
            #     format='GIF',
            #     # append_images,
            #     save_all=True,
            #     duration=40,
            #     loop=0
            #     )

        self.log_dict({'val/loss/general_loss': general_loss,
                       'val/loss/loss_pos': loss_pos,
                       'val/loss/loss_act': loss_act,
                       # 'train/loss/general_loss_encoded': general_loss_encoded,
                       })

        return general_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
