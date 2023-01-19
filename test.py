from IPython.core.pylabtools import figsize
from src.model import Predictor
import torch as th
import numpy as np
import ipdb
import matplotlib.pyplot as plt
import random
from utils.create_window import create_window

def test():
    WINDOW_SIZE = 10
    BATCH_SIZE = 128
    HIDDEN_SIZE = 256
    WEIGHT_DECAY = 0.00
    MAX_EPOCHS = 1
    HIDDEN_IS_WITH_NOISE = False
    NUM_OF_DATA_TO_TEST = 1000


#     name = f'Test with - batch:{BATCH_SIZE}-hidden_layer_size: {HIDDEN_SIZE}\
#             -weight_decay:{WEIGHT_DECAY}-epochs:{MAX_EPOCHS}\
#             -hidden_is_with_noise:{HIDDEN_IS_WITH_NOISE}'

#     wb_logger = WandbLogger(
#                 name=name,
#                 project='rnn-predict-next-positions', 
#                 log_model='all',
#                 )


    model = Predictor(
            input_size = 8, hidden_size = 256, out_size = 4,
            num_layers1= 1, num_layers2= 1,
            num_workers= 8, batch_size=128,
            num_of_data_to_train = 1, 
            num_of_data_to_val = 1, 
            weight_decay= 0.1, window=WINDOW_SIZE)

    model = model.load_from_checkpoint('./final_model.ckpt')
    model.to('cuda')
    model.eval()
    # img = plt.imread("./src/image/image_field.png")

    indexes = random.sample(range(0,NUM_OF_DATA_TO_TEST), 10)

    for i in indexes:
        figure, axis = plt.subplots(2, 2, figsize=(15,15), dpi=80)
        
        data = np.loadtxt(open('./data-v2-test' + f'/positions-{i}.txt'), dtype=np.float32)
        out_to_test = create_window(data, WINDOW_SIZE)
        
        preds_ball_x = list()
        preds_ball_y = list()
        preds_robot_x = list()
        preds_robot_y = list()
        
        true_ball_x = list()
        true_ball_y = list()
        true_robot_x = list()
        true_robot_y = list()

        for seq, y_true in out_to_test:
            model.hidden = (
                            th.zeros(1,1,HIDDEN_SIZE).to('cuda').float(),
                            th.zeros(1,1,HIDDEN_SIZE).to('cuda').float(),
                            )

            seq = th.FloatTensor(seq).to('cuda')
            seq = seq.view((1,1,seq.shape[0]*seq.shape[1])).float()

            y_pred = model(seq)
            
            y_pred = y_pred.squeeze()
            y_pred = y_pred.cpu().detach().numpy()

            true_ball_x.append(y_true[0][0])
            preds_ball_x.append(y_pred[0])

            true_ball_y.append(y_true[0][1])
            preds_ball_y.append(y_pred[1])

            true_robot_x.append(y_true[0][2])
            preds_robot_x.append(y_pred[2])

            true_robot_y.append(y_true[0][3])
            preds_robot_y.append(y_pred[3])

        x = [i  for i in range(len(preds_ball_x))]

        axis[0,0].plot(x, preds_ball_x, label = 'preds_ball_x', marker='o')
        axis[0,0].plot(x, true_ball_x, label ='real_ball_x', marker='x')
        
        axis[0,0].set_ylim([-1,1])

        axis[0,0].set_title("Pred ball_x vs Real ball_x")

        axis[0,1].plot(x, preds_ball_y, label = 'preds_ball_y', marker='o')
        axis[0,1].plot(x, true_ball_y, label = 'real_ball_y', marker='x')
        axis[0,1].set_ylim([-1,1])

        axis[0,1].set_title("Pred ball_y vs Real ball_y")

        axis[1, 0].plot(x, preds_robot_x, label = 'preds_robot_x', marker='o')
        axis[1, 0].plot(x, true_robot_x, label = 'real_robot_x', marker='x')
        
        axis[1,0].set_ylim([-1,1])

        axis[1,0].set_title("Pred robot_x vs Real robot_x")

        axis[1, 1].plot(x, preds_robot_y, label = 'preds_robot_y', marker='o')
        axis[1, 1].plot(x, true_robot_y, label = 'real_robot_y', marker='x')
        
        axis[1,1].set_ylim([-1,1])
        
        axis[1,1].set_title("Pred robot_y vs Real robot_y")


        plt.savefig(f'./graphs3/data_set{i}.png')

if __name__ == '__main__':
    test()
