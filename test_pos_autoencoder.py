from pos_latent import PosLatent
from pos_autoencoder import PositionAutoEncoder
import torch
from render import RCGymRender
import pandas as pd
import copy
import ipdb
import PIL
import numpy as np
import math
from src.aux_idx import Aux
import matplotlib.pyplot as plt
import os

indexes_act = Aux.is_vel
# self.indexes will be not self.indexes_act
indexes = []

for value in indexes_act:
    indexes.append(not value)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
HIDDEN_SIZE = 256
POS_HIDDEN_SIZE = 256
WINDOW_SIZE = 10
INPUT_SIZE = 50
EPOCHS = 100
LR = 1e-4
NUM_WORKERS = 15
WEIGHTS = [0.9, 0.1]

ACT_PATH = '/home/bmmuc/Documents/robocin/rnn/rnn-predict-ssl/model_bom/model_10_act.pth'
POS_PATH = '/home/bmmuc/Documents/robocin/rnn/rnn-predict-ssl/model_bom/model_10_pos.pth'


def create_window(data, window_size, horizon=1):
    out = []
    L = len(data)

    for i in range(L - window_size - horizon):
        window = data[i: i+window_size, :]
        label = data[i+window_size: i+window_size+horizon, :]
        out.append((window, label))
    return out


def normalize_data(data, colmn_name):
    id_ = colmn_name.split('_')[-1]

    if id_ == 'x' and 'vel' not in colmn_name:
        print('atuando em: ', colmn_name)
        data[colmn_name] = data[colmn_name].replace(-99999, 10)
        data[colmn_name] = np.clip(data[colmn_name] / 9, -1.2, 1.2)

    elif id_ == 'y' and 'vel' not in colmn_name:
        data[colmn_name] = data[colmn_name].replace(-99999, 10)
        print('atuando em: ', colmn_name)

        data[colmn_name] = np.clip(data[colmn_name] / 6, -1.2, 1.2)

    elif id_ == 'x' and 'vel' in colmn_name:
        data[colmn_name] = data[colmn_name].replace(-99999, 9)
        print('atuando em: ', colmn_name)

        data[colmn_name] = np.clip(data[colmn_name] / 8, -1.2, 1.2)

    elif id_ == 'y' and 'vel' in colmn_name:
        data[colmn_name] = data[colmn_name].replace(-99999, 9)
        print('atuando em: ', colmn_name)

        data[colmn_name] = np.clip(data[colmn_name] / 8, -1.2, 1.2)

    elif id_ == 'orientation' and 'vel' not in colmn_name:
        # robot_yellow_{i}_orientation
        new_name = colmn_name.replace('orientation', 'sin')
        data[colmn_name] = data[colmn_name].replace(-99999, 4)
        print('atuando em: ', colmn_name)

        data[new_name] = np.sin(data[colmn_name])

        new_name = colmn_name.replace('orientation', 'cos')
        data[new_name] = np.cos(data[colmn_name])

    elif id_ == 'angular' and 'vel' in colmn_name:
        data[colmn_name] = data[colmn_name].replace(-99999, 9)
        print('atuando em: ', colmn_name)

        data[colmn_name] = np.clip(data[colmn_name] / 8, -1.2, 1.2)
    else:
        print('não atuando em: ', colmn_name)

    return data


model = PositionAutoEncoder(
    window=WINDOW_SIZE,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=INPUT_SIZE,
    lr=LR,
)

model.load_state_dict(torch.load(
    './modelo_autoencoder_pos.pth', map_location=torch.device('cpu')))


data = pd.read_csv(
    '/home/bmmuc/Documents/robocin/logs-unification/reader/output2_fixed/2021-06-21_10-22_OMID-vs-RobôCin.csv')

data.fillna(-99999, inplace=True)
columns_to_get = ['ball_x',
                  'ball_y',
                  # 'ball_vel_x',
                  # 'ball_vel_y'
                  ]

for i in range(6):
    columns_to_get.append(f'robot_blue_{i}_x')
    columns_to_get.append(f'robot_blue_{i}_y')
    columns_to_get.append(f'robot_blue_{i}_orientation')
    # columns_to_get.append(f'robot_blue_{i}_sin')
    # columns_to_get.append(f'robot_blue_{i}_cos')
    columns_to_get.append(f'robot_blue_{i}_vel_x')
    columns_to_get.append(f'robot_blue_{i}_vel_y')
    columns_to_get.append(f'robot_blue_{i}_vel_angular')

for i in range(6):
    columns_to_get.append(f'robot_yellow_{i}_x')
    columns_to_get.append(f'robot_yellow_{i}_y')
    columns_to_get.append(f'robot_yellow_{i}_orientation')
    # columns_to_get.append(f'robot_yellow_{i}_sin')
    # columns_to_get.append(f'robot_yellow_{i}_cos')
    columns_to_get.append(f'robot_yellow_{i}_vel_x')
    columns_to_get.append(f'robot_yellow_{i}_vel_y')
    columns_to_get.append(f'robot_yellow_{i}_vel_angular')

# ipdb.set_trace()
data.drop('ref_command', inplace=True, axis=1)


for col in columns_to_get:
    data = normalize_data(data, col)

for i, col in enumerate(columns_to_get):
    if 'orientation' in col:
        columns_to_get[i] = col.replace('orientation', 'sin')
        columns_to_get.insert(i+1, col.replace('orientation', 'cos'))

data = data[columns_to_get]
# ipdb.set_trace()
cols = data[columns_to_get].columns
data = data.values
mask = []
for col in cols:
    if col in columns_to_get:
        mask.append(True)
    else:
        mask.append(False)

files = os.listdir('/home/bmmuc/Documents/robocin/new_normalized/')
files = files[:10]
files.append('2021-06-21_10-22_OMID-vs-RobôCin-15.npy')
inputs_ball = []
outputs_ball = []

for f in files[0]:
    for i in range(2):
        data = np.load(
            f'/home/bmmuc/Documents/robocin/new_normalized/{f}', allow_pickle=True).astype(np.float32)
        # data = data[:, mask]

        data = create_window(data, 10, 1)
        render = RCGymRender(6, 6, simulator='ssl')
        frames = []
        for seq, y_true in data:
            seq2 = copy.deepcopy(seq)
            inputs_ball.append(seq2[0, :2])
            seq2 = torch.tensor(seq2, dtype=torch.float32).to(device)
            seq2 = seq2.view(1, seq2.shape[0], seq2.shape[1])
            # seq2 = torch.tensor(seq2, dtype=torch.float32).cuda()
            pos, _ = model(seq2)
            pos = pos.view(1, pos.shape[0], pos.shape[1])
            pos = pos.cpu().detach().numpy()
            # outputs_ball.append(pos[0, -1, :2])
            # print(pos[0, -1, :2])
            # ipdb.set_trace()
            # frame = render.render_frame(True, y_true[-1, 0], y_true[-1, 1], y_true[0, 0], y_true[0, 1],
            #                             preds=pos[0, -1, :], all_true=y_true[-1, indexes])
            frame = render.render_frame(True, y_true[-1, 0], y_true[-1, 1], y_true[0, 0], y_true[0, 1],
                                        preds=pos[0, -1, :], all_true=pos[:, -1, :])
            frame = PIL.Image.fromarray(frame)
            frames.append(frame)
            del pos
            del seq2
        name = 'real'

        if i == 0:
            name='autoencoder'

        frames[0].save(
        fp=f'./gifs/pos_autoencoder_dataset{f}_{name}.gif', 
        format='GIF', 
        append_images=frames[1:], 
        save_all=True,
        duration=40, 
        loop=0
        )

# inputs_ball = np.array(inputs_ball)
# outputs_ball = np.array(outputs_ball)

# np.save('inputs_ball.npy', inputs_ball)
# np.save('outputs_ball.npy', outputs_ball)

del render
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(10, 5))

# # ax1 is input X
# ax1.hist(inputs_ball[:, 0], bins=1000, label='input X')

# # ax2 is input Y
# ax2.hist(inputs_ball[:, 1], bins=1000, label='input Y')

# # ax3 is output X
# ax3.hist(outputs_ball[:, 0], bins=1000, label='output X')

# # ax4 is output Y
# ax4.hist(outputs_ball[:, 1], bins=1000, label='output Y')

plt.show()
