from copy import copy, deepcopy
import numpy as np
import torch
from torch import nn, no_grad

from torch.optim import Adam
from torch.nn import functional as F
from act_autoencoder import ActAutoEncoder
from pos_autoencoder import PositionAutoEncoder

from src.aux_idx import Aux

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PosLatent(nn.Module):
    def __init__(self, window=10, input_size=10,
                 hidden_size=256,
                 pos_hidden_size=256,
                 output_size=20,
                 weights=(0.5, 0.5),
                 act_path='',
                 pos_path='',
                 lr=1e-3,
                 ):
        super().__init__()

        self.window = window
        self.output_size = output_size
        self.lr = lr
        self.automatic_optimization = False
        self.device = device

        self.pos_autoencoder = PositionAutoEncoder(
            window=window, input_size=38, hidden_size=pos_hidden_size, output_size=38)
        self.pos_autoencoder.load_state_dict(torch.load(pos_path))

        self.act_autoencoder = ActAutoEncoder(
            window=self.window, input_size=36, hidden_size=hidden_size, output_size=36)
        self.act_autoencoder.load_state_dict(torch.load(act_path))

        self.pos_autoencoder = self.pos_autoencoder.eval()
        self.act_autoencoder = self.act_autoencoder.eval()

        self.pred_pos = nn.ModuleDict({
            'linear1': nn.Linear(pos_hidden_size + hidden_size, 1024),
            'linear2': nn.Linear(1024, 512),
            'linear3': nn.Linear(512, 512),
            'linear4': nn.Linear(512, 512),
            'linear5': nn.Linear(512, 512),
            'linear8': nn.Linear(512, 256),
            'linear9': nn.Linear(256, 256),
            # 'linear10': nn.Linear(256, 256),
            # 'linear11': nn.Linear(256, 256),
            # 'linear12': nn.Linear(256, 256),
            # 'linear13': nn.Linear(256, 256),
            # 'linear14': nn.Linear(256, 256),
            # 'linear15': nn.Linear(256, 256),
            'linear16': nn.Linear(256, pos_hidden_size),

        })

        self.pred_act = nn.ModuleDict({
            # estrutura pos => (512, 1024)
            'linear1': nn.Linear(hidden_size, 1024),
            # 'linear1': nn.Linear(512, 1024), # estrutura pos + act => (512, 1024)
            'linear2': nn.Linear(1024, 512),
            'linear3': nn.Linear(512, 512),
            'linear4': nn.Linear(512, 512),
            'linear5': nn.Linear(512, 256),
            'linear6': nn.Linear(256, 256),
            'linear7': nn.Linear(256, 256),
            'linear8': nn.Linear(256, 256),
            'linear9': nn.Linear(256, 256),
            'linear10': nn.Linear(256, hidden_size),
        })

        self.opt = self.configure_optimizers()  # testar 2 optimizers

        self.weitght_pos = weights[0]
        self.weitght_acts = weights[1]

        # self.save_hyperparameters()
        self = self.to(torch.device('cuda'))

        torch.autograd.set_detect_anomaly(True)
        self.render = False

        self.indexes_act = Aux.is_vel
        # self.indexes will be not self.indexes_act
        self.indexes = []

        for value in self.indexes_act:
            self.indexes.append(not value)

    def run_pos(self, act_pos_hidden_concat):
        with torch.no_grad():

            out = self.pred_pos['linear1'](act_pos_hidden_concat)
            out = self.pred_pos['linear2'](out)
            out = self.pred_pos['linear3'](out)
            out = self.pred_pos['linear4'](out)
            out = self.pred_pos['linear5'](out)
            out = self.pred_pos['linear8'](out)
            out = self.pred_pos['linear9'](out)
            out = self.pred_pos['linear10'](out)
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
        out = self.pred_pos['linear5'](out)
        out = self.pred_pos['linear8'](out)
        out = self.pred_pos['linear9'](out)
        # out = self.pred_pos['linear10'](out)
        # out = self.pred_pos['linear11'](out)
        # out = self.pred_pos['linear12'](out)
        # out = self.pred_pos['linear13'](out)
        # out = self.pred_pos['linear14'](out)
        # out = self.pred_pos['linear15'](out)
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

    def training_step(self, X, y):
        self.train()
        # if self.current_epoch % 4 == 0:
        #     self.render = True
        # if self.global_step == 0:
        #     gen = self.validation_step(batch, batch_idx)
        #     return gen

        # ipdb.set_trace()

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

        general_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.0175)

        self.opt.step()

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

        log_dict = {'train/loss/general_loss': general_loss,
                    'train/loss/loss_pos': loss_pos,
                    'train/loss/loss_act': loss_act,
                    'max_abs_gradients': max(ave_grads),
                    'max_norm_gradients': max(norm_grad),
                    'sum_norm_gradients': total_norm,
                    # 'train/loss/general_loss_encoded': general_loss_encoded,
                    }

        return log_dict

    def validation_step(self, X, y):
        self.eval()
        # ipdb.set_trace()

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

        # if self.render:

        #     # render = RCGymRender(should_render_actual_ball = False, n_robots_blue = 3, n_robots_yellow = 3)
        #     Handle_render(self, False, name_predictor=f'pos_latent-eval-{self.global_step}',
        #                     num_of_features=self.output_size, path=f'./gifs/train/train-{self.global_step}-3v3.gif',
        #                     is_traning=True)\
        #                         .render_n_steps_autoencoder3v3()

        #     self.render = False
        #     # frame = render.render_frame(
        #     #             # ball_actual_x = seq[-1][0], ball_actual_y = seq[-1][1],
        #     #             ball_true_x = y_copy[idx_to_use][0], ball_true_y = y_copy[idx_to_use][1],
        #     #             ball_pred_x = pred1[idx_to_use][0], ball_pred_y = pred1[idx_to_use][1],
        #     #             robot_actual_x= y_copy[idx_to_use][4], robot_actual_y = y_copy[idx_to_use][5],
        #     #             robot_blue_theta = y_copy[idx_to_use][6],
        #     #             robot2_actual_x= y_copy[idx_to_use][11], robot2_actual_y = y_copy[idx_to_use][12],
        #     #             robot2_blue_theta = y_copy[idx_to_use][13],
        #     #             robot3_actual_x= y_copy[idx_to_use][18], robot3_actual_y = y_copy[idx_to_use][19],
        #     #             robot3_blue_theta = y_copy[idx_to_use][20],
        #     #             robot_yellow_x= y_copy[idx_to_use][25], robot_yellow_y = y_copy[idx_to_use][26],
        #     #             robot_yellow_theta = y_copy[idx_to_use][27],
        #     #             robot2_yellow_x= y_copy[idx_to_use][30], robot2_yellow_y = y_copy[idx_to_use][31],
        #     #             robot2_yellow_theta = y_copy[idx_to_use][32],
        #     #             robot3_yellow_x= y_copy[idx_to_use][35], robot3_yellow_y = y_copy[idx_to_use][36],
        #     #             robot3_yellow_theta = y_copy[idx_to_use][37],
        #     #             return_rgb_array=True)

        #     # frame = PIL.Image.fromarray(frame)

        #     # del(render)
        #     # # ipdb.set_trace()

        #     # frame.save(
        #     #     fp=f'./gifs/train/global-step:{self.global_step}',
        #     #     format='GIF',
        #     #     # append_images,
        #     #     save_all=True,
        #     #     duration=40,
        #     #     loop=0
        #     #     )

        log_dict = {'val/loss/general_loss': general_loss,
                    'val/loss/loss_pos': loss_pos,
                    'val/loss/loss_act': loss_act,
                    # 'train/loss/general_loss_encoded': general_loss_encoded,
                    }

        return log_dict

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
