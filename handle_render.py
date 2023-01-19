import rsoccer_gym
import torch as th
import numpy as np
from render import RCGymRender
from src.model import Predictor
from src.autoencoder import RecurrentAutoencoder
from src.new_autoencoder import RecurrentAutoencoder2
from src.predictor_with_multiple_hidden import PredictorEncoder
from src.new_pred import Act_pos_forecasting
from src.pos_latent import PosLatent
from src.concat_autoencoder import ConcatAutoEncoder
from torch.nn import functional as F
from src.model_with_encoder import Predictor_encoder
from utils.predict_n_future_steps import Pred_n_future
from utils.create_window import create_window
import random
import copy
import ipdb
import PIL
# from torch.nn import functional as F

WINDOW_SIZE = 10
HORIZON = 5
NUM_OF_FEATURES = 40
NUM_OF_DATA_TO_TEST = 1000
HIDDEN_SIZE = 256

class Handle_render:
    def __init__(self, model, should_render_actual_ball = True, 
                    name_predictor = '', name_autoenceoder = '',
                    num_of_features = 41, path='./gifs/gif-of-dataset-test-idx-{i}-horizon-{HORIZON}-3v3.gif',
                    is_traning = False) -> None:
        self.model = model
        self.traning = is_traning
        self.should_render_actual_ball = should_render_actual_ball
        self.path = path
        self.name_predictor = name_predictor
        self.name_autoenceoder = name_autoenceoder
        self.num_of_features = num_of_features

    def render_pos(self):
        render = RCGymRender(should_render_actual_ball=self.should_render_actual_ball)
        
        indexes = random.sample(range(0, NUM_OF_DATA_TO_TEST), 10)

        for i in indexes:
            data = np.loadtxt(open('../all_data/data-v4-test' + f'/positions-{i}.txt'), dtype=np.float32)
            out_to_test = create_window(data, WINDOW_SIZE, 1)
            frames = []
            for seq, y_true in out_to_test:
                self.model.hidden = (
                                th.zeros(1,1,HIDDEN_SIZE).to('cuda').float(),
                                th.zeros(1,1,HIDDEN_SIZE).to('cuda').float(),
                                )
                

                seq = th.FloatTensor(seq).to('cuda')
                seq = seq.view((1,1,seq.shape[0]*seq.shape[1])).float()
          

                y_pred = self.model(seq)
                y_pred = y_pred.squeeze()
                y_pred = y_pred.cpu().detach().numpy()
                # ipdb.set_trace()
                        
                frame = render.render_frame(
                    # ball_actual_x = seq[-1][0], ball_actual_y = seq[-1][1],
                    ball_true_x = y_true[-1][0], ball_true_y = y_true[-1][1],
                    ball_pred_x = y_pred[-1][0], ball_pred_y = y_pred[-1][1],
                    robot_actual_x= y_true[-1][4], robot_actual_y = y_true[-1][5],
                    robot_blue_theta = y_true[-1][6],
                    robot_yellow_x= y_true[-1][11], robot_yellow_y = y_true[-1][12],
                    # robot_yellow_theta = y_true[-1][15],
                    return_rgb_array=True
                )
                frame = PIL.Image.fromarray(frame)

                frames.append(frame)
        
                # time.sleep(0.1)
            frames[0].save(
                fp=f'./gifs/autoencoder-{i}.gif', 
                format='GIF', 
                append_images=frames[1:], 
                save_all=True,
                duration=90, 
                loop=0
            )
    
    def render_pos_autoencoder(self):
        render = RCGymRender(should_render_actual_ball=self.should_render_actual_ball,  n_robots_blue=3, n_robots_yellow=3)
        
        indexes = random.sample(range(0, NUM_OF_DATA_TO_TEST), 10)

        for i in indexes:

            data = np.loadtxt(open('../all_data/data-3v3-v2' + f'/positions-{i}.txt'), dtype=np.float32)
            if len(data) < 70:
                continue

            out_to_test = create_window(data, WINDOW_SIZE, 1)
            frames = []
            for seq, y_true in out_to_test:
                # self.model.hidden = (
                #                 th.zeros(1,1,HIDDEN_SIZE).to('cuda').float(),
                #                 th.zeros(1,1,HIDDEN_SIZE).to('cuda').float(),
                #                 )
                # ipdb.set_trace()

                seq = th.FloatTensor(seq).to('cuda')
                # seq = seq.view((1,1,seq.shape[0]*seq.shape[1])).float()
                # seq = seq.view((1,seq.shape[0],seq.shape[1])).float()
                seq = seq.view(1, 10 , 41).float()
                seq = seq[:, :, :40]

                y_pred = self.model(seq)
                # ipdb.set_trace()
                # _, y_pred = self.model(seq)
                # y_pred = y_pred.squeeze()
                y_pred = y_pred.cpu().detach().numpy()

                # ipdb.set_trace()
                        
                frame = render.render_frame(
                    # ball_actual_x = seq[-1][0], ball_actual_y = seq[-1][1],
                    ball_true_x = y_true[-1][0], ball_true_y = y_true[-1][1],
                    ball_pred_x = y_pred[-1][-1][0], ball_pred_y = y_pred[-1][-1][1],
                    robot_actual_x= y_true[-1][4], robot_actual_y = y_true[-1][5],
                    robot_blue_theta = y_true[-1][6],
                    robot2_actual_x= y_true[-1][11], robot2_actual_y = y_true[-1][12],
                    robot2_blue_theta = y_true[-1][13],
                    robot3_actual_x= y_true[-1][18], robot3_actual_y = y_true[-1][19],
                    robot3_blue_theta = y_true[-1][20],
                    robot_yellow_x= y_true[-1][25], robot_yellow_y = y_true[-1][26],
                    robot_yellow_theta = y_true[-1][27],
                    robot2_yellow_x= y_true[-1][30], robot2_yellow_y = y_true[-1][31],
                    robot2_yellow_theta = y_true[-1][32],
                    robot3_yellow_x= y_true[-1][35], robot3_yellow_y = y_true[-1][36],
                    robot3_yellow_theta = y_true[-1][37],
                    return_rgb_array=True)
                frame = PIL.Image.fromarray(frame)

                frames.append(frame)
        
                # time.sleep(0.1)
            frames[0].save(
                fp=f'./gifs/pred_lstm-{i}.gif', 
                format='GIF', 
                append_images=frames[1:], 
                save_all=True,
                duration=90, 
                loop=0
            )


    def render_n_steps(self):
        render = RCGymRender(should_render_actual_ball = False)
        indexes = random.sample(range(0, NUM_OF_DATA_TO_TEST), 10)
        get_preds = Pred_n_future(self.model, HORIZON)
        # testing = ConcatDataSet('../all_data/data-v2-test', NUM_OF_DATA_TO_TEST, 10, 5)
        

        for i in indexes:   
            total_error_x = 0
            total_error_y = 0
                     
            data = np.loadtxt(open('../all_data/data-v4-test' + f'/positions-{i}.txt'), dtype=np.float32)
            # data = np.loadtxt(open('../all_data/data-v4' + f'/positions-{0}.txt'), dtype=np.float32)
            if len(data) <= 30:
                continue

            out_to_test = create_window(data, WINDOW_SIZE, HORIZON)

            frames = []
            for seq, y_true in out_to_test:
                seq2 = copy.deepcopy(seq)
                test = get_preds.predict_n_future_steps(seq2)
                # ipdb.set_trace()
                # general_loss_x = abs(test[-1][0] - y_true[0][0])
                # total_error_x += general_loss_x 

                # general_loss_y = abs(test[-1][1] - y_true[0][1])
                # total_error_y += general_loss_y

                frame = render.render_frame(
                    # ball_actual_x = seq[-1][0], ball_actual_y = seq[-1][1],
                    ball_true_x = y_true[-1][0], ball_true_y = y_true[-1][1],
                    ball_pred_x = test[-1][0], ball_pred_y = test[-1][1],
                    robot_actual_x= y_true[-1][4], robot_actual_y = y_true[-1][5],
                    robot_blue_theta = y_true[-1][6],
                    robot_yellow_x= y_true[-1][11], robot_yellow_y = y_true[-1][12],
                    # robot_yellow_theta = y_true[-1][15],
                    return_rgb_array=True
                )

                frame = PIL.Image.fromarray(frame)
                frames.append(frame)
                del seq2
                del test


            # print('i:', i, 'mean_x': )
            frames[0].save(
                fp=f'./gifs/preds_to_discord/gif-of-dataset-test-idx-{i}.gif', 
                format='GIF', 
                append_images=frames[1:], 
                save_all=True,
                duration=40, 
                loop=0
            )

    def render_pos_autoencoder3v3(self):
        render = RCGymRender(should_render_actual_ball = False, n_robots_blue = 3, n_robots_yellow = 3)
        should_use_n_datasets = 10
        if self.traning:
            should_use_n_datasets = 1

        indexes = random.sample(range(0, NUM_OF_DATA_TO_TEST), should_use_n_datasets)

        for i in indexes:

            data = np.loadtxt(open('../all_data/data-3v3-v1' + f'/positions-{i}.txt'), dtype=np.float32)
            if len(data) < 70:
                continue
            out_to_test = create_window(data, WINDOW_SIZE, 1)
            frames = []
            for seq, y_true in out_to_test:
                # self.model.hidden = (
                #                 th.zeros(1,1,HIDDEN_SIZE).to('cuda').float(),
                #                 th.zeros(1,1,HIDDEN_SIZE).to('cuda').float(),
                #                 )
                # ipdb.set_trace()

                seq = th.FloatTensor(seq).to('cuda')
                # ipdb.set_trace()
                # seq = seq.view((1,1,seq.shape[0]*seq.shape[1])).float()
                # seq = seq.view((1,seq.shape[0],seq.shape[1])).float()
                seq = seq.view(1, 10 , 41).float()
            
                _, y_pred = self.model(seq)
                # ipdb.set_trace()
                # _, y_pred = self.model(seq)
                # y_pred = y_pred.squeeze()
                y_pred = y_pred.cpu().detach().numpy()
                # ipdb.set_trace()
                        
                frame = render.render_frame(
                    # ball_actual_x = seq[-1][0], ball_actual_y = seq[-1][1],
                    ball_true_x = y_true[-1][0], ball_true_y = y_true[-1][1],
                    ball_pred_x = y_pred[-1][-1][0], ball_pred_y = y_pred[-1][-1][1],
                    robot_actual_x= y_true[-1][4], robot_actual_y = y_true[-1][5],
                    robot_blue_theta = y_true[-1][6],
                    robot2_actual_x= y_true[-1][11], robot2_actual_y = y_true[-1][12],
                    robot2_blue_theta = y_true[-1][13],
                    robot3_actual_x= y_true[-1][18], robot3_actual_y = y_true[-1][19],
                    robot3_blue_theta = y_true[-1][20],
                    robot_yellow_x= y_true[-1][25], robot_yellow_y = y_true[-1][26],
                    robot_yellow_theta = y_true[-1][27],
                    robot2_yellow_x= y_true[-1][30], robot2_yellow_y = y_true[-1][31],
                    robot2_yellow_theta = y_true[-1][32],
                    robot3_yellow_x= y_true[-1][35], robot3_yellow_y = y_true[-1][36],
                    robot3_yellow_theta = y_true[-1][37],
                    return_rgb_array=True)
                frame = PIL.Image.fromarray(frame)

                frames.append(frame)
        
                # time.sleep(0.1)
            frames[0].save(
                fp=f'./gifs/pred_lstm-{i}.gif', 
                format='GIF', 
                append_images=frames[1:], 
                save_all=True,
                duration=90, 
                loop=0
            )
    

    def render_n_steps_autoencoder(self):
        render = RCGymRender(should_render_actual_ball = False)
        indexes = random.sample(range(0, NUM_OF_DATA_TO_TEST), 10)
        get_preds = Pred_n_future(self.model, HORIZON)
        # testing = ConcatDataSet('../all_data/data-v2-test', NUM_OF_DATA_TO_TEST, 10, 5)
        

        for i in indexes:   
            total_error_x = 0
            total_error_y = 0
                     
            data = np.loadtxt(open('../all_data/data-v6-val' + f'/positions-{i}.txt'), dtype=np.float32)
            # data = np.loadtxt(open('../all_data/data-v4' + f'/positions-{0}.txt'), dtype=np.float32)
            if len(data) <= 30:
                continue

            out_to_test = create_window(data, WINDOW_SIZE, HORIZON)

            frames = []
            for seq, y_true in out_to_test:
                seq2 = copy.deepcopy(seq)
                test = get_preds.predict_n_future_steps2(seq2)
                # ipdb.set_trace()
                # general_loss_x = abs(test[-1][0] - y_true[0][0])
                # total_error_x += general_loss_x 

                # general_loss_y = abs(test[-1][1] - y_true[0][1])
                # total_error_y += general_loss_y
                # ipdb.set_trace()

                frame = render.render_frame(
                    # ball_actual_x = seq[-1][0], ball_actual_y = seq[-1][1],
                    ball_true_x = y_true[-1][0], ball_true_y = y_true[-1][1],
                    ball_pred_x = test[-1][0], ball_pred_y = test[-1][1],
                    robot_actual_x= y_true[-1][4], robot_actual_y = y_true[-1][5],
                    robot_blue_theta = y_true[-1][6],
                    robot_yellow_x= y_true[-1][11], robot_yellow_y = y_true[-1][12],
                    # robot_yellow_theta = y_true[-1][15],
                    return_rgb_array=True
                )

                frame = PIL.Image.fromarray(frame)
                frames.append(frame)
                del seq2
                del test
            print('i: ', i,'error_x: ',total_error_x)
            print('i: ', i,'error_y: ',total_error_y)
            # print('i:', i, 'mean_x': )
            frames[0].save(
            fp=f'./gifs/gif-of-dataset-test-idx-{i}-horizon-{HORIZON}.gif', 
            format='GIF', 
            append_images=frames[1:], 
            save_all=True,
            duration=40, 
            loop=0
            )

    def render_n_steps_autoencoder_hidden_3v3(self):
        render = RCGymRender(should_render_actual_ball = False, n_robots_blue = 3, n_robots_yellow = 3)
        indexes = random.sample(range(0, NUM_OF_DATA_TO_TEST), 10)
        get_preds = Pred_n_future(self.model, HORIZON)
        # testing = ConcatDataSet('../all_data/data-v2-test', NUM_OF_DATA_TO_TEST, 10, 5)
        

        for i in indexes:   
            total_error_x = 0
            total_error_y = 0
                     
            data = np.loadtxt(open('../all_data/data-3v3-v1' + f'/positions-{i}.txt'), dtype=np.float32)
            # data = np.loadtxt(open('../all_data/data-v4' + f'/positions-{0}.txt'), dtype=np.float32)
            if len(data) <= 30:
                continue

            out_to_test = create_window(data, WINDOW_SIZE, HORIZON)

            frames = []
            for seq, y_true in out_to_test:
                seq = th.FloatTensor(seq).to('cuda')
           
                seq = seq.view(1, 10 , 41).float()
                seq2 = copy.deepcopy(seq)
                seq2 = seq2.reshape(1, 41 , 10).float()
                test = model.predict_t_steps(seq2, HORIZON)
                ipdb.set_trace()

                frame = render.render_frame(
                    # ball_actual_x = seq[-1][0], ball_actual_y = seq[-1][1],
                    ball_true_x = y_true[-1][0], ball_true_y = y_true[-1][1],
                    ball_pred_x = test[-1][-1][0], ball_pred_y = test[-1][-1][1],
                    robot_actual_x= y_true[-1][4], robot_actual_y = y_true[-1][5],
                    robot_blue_theta = y_true[-1][6],
                    robot2_actual_x= y_true[-1][11], robot2_actual_y = y_true[-1][12],
                    robot2_blue_theta = y_true[-1][13],
                    robot3_actual_x= y_true[-1][18], robot3_actual_y = y_true[-1][19],
                    robot3_blue_theta = y_true[-1][20],
                    robot_yellow_x= y_true[-1][25], robot_yellow_y = y_true[-1][26],
                    robot_yellow_theta = y_true[-1][27],
                    robot2_yellow_x= y_true[-1][30], robot2_yellow_y = y_true[-1][31],
                    robot2_yellow_theta = y_true[-1][32],
                    robot3_yellow_x= y_true[-1][35], robot3_yellow_y = y_true[-1][36],
                    robot3_yellow_theta = y_true[-1][37],
                    return_rgb_array=True)

                frame = PIL.Image.fromarray(frame)
                frames.append(frame)
                del seq2
                del test
            print('i: ', i,'error_x: ',total_error_x)
            print('i: ', i,'error_y: ',total_error_y)
            # print('i:', i, 'mean_x': )
            frames[0].save(
            fp=f'./gifs/gif-of-dataset-test-idx-{i}-horizon-{HORIZON}--hidden_method-3v3.gif', 
            format='GIF', 
            append_images=frames[1:], 
            save_all=True,
            duration=40, 
            loop=0
            )
    
    def test_pos_latent(self):
        render = RCGymRender(should_render_actual_ball = False, n_robots_blue = 3, n_robots_yellow = 3)
        indexes = random.sample(range(0, NUM_OF_DATA_TO_TEST), 10)
        get_preds = Pred_n_future(self.model, HORIZON)
        # testing = ConcatDataSet('../all_data/data-v2-test', NUM_OF_DATA_TO_TEST, 10, 5)
        # ipdb.set_trace()
        # file = open('./gifs/params.txt', 'w')
        # file.write(f'predictor: {self.name_predictor}\n')
        # file.write(f'autoencoder: {self.name_autoenceoder}')
        # file.close()

        for i in indexes:   
            total_error_x = 0
            total_error_y = 0
                     
            data = np.loadtxt(open('../all_data/data-3v3-v2' + f'/positions-{i}.txt'), dtype=np.float32)
            # data = np.loadtxt(open('../all_data/data-v4' + f'/positions-{0}.txt'), dtype=np.float32)
            if len(data) <= 30:
                continue

            out_to_test = create_window(data, WINDOW_SIZE, HORIZON)

            frames = []
            for seq, y_true in out_to_test:
                '''
                    seq.shape -> (10, 41)
                    y_true.shape -> (1, 41)
                '''
                seq2 = copy.deepcopy(seq)
                seq2 = seq2[:, :-1]
                seq2 = th.FloatTensor(seq2).cuda()
                
                seq2 = seq2.reshape(1, 10 , seq2.shape[1]).float()

                ipdb.set_trace()
                indexes = [0,1,2,3,4,5,6,7,11,12,13,14,18,19,20,21,27,28,30,31,35,36]
                test = model(seq2)
                seq2 = seq[:, indexes]
                seq2 = th.FloatTensor(seq2).cuda()
                seq2 = seq2.reshape(1, 10 , seq2.shape[1]).float()
                test = model.decoding(test, seq2)
                # test = get_preds.test_pos_latent(seq2, self.num_of_features)
                # test = seq2
                # general_loss_x = abs(test[-1][0] - y_true[0][0])
                # total_error_x += general_loss_x 

                # general_loss_y = abs(test[-1][1] - y_true[0][1])
                # total_error_y += general_loss_y
                # ipdb.set_trace()
                # 4 + 14 =  
                frame = render.render_frame(
                    # ball_actual_x = seq[-1][0], ball_actual_y = seq[-1][1],
                    ball_true_x = y_true[-1][0], ball_true_y = y_true[-1][1],
                    ball_pred_x = test[-1][-1][0], ball_pred_y = test[-1][-1][1],
                    robot_actual_x= y_true[-1][4], robot_actual_y = y_true[-1][5],
                    robot_blue_theta = y_true[-1][6],
                    robot2_actual_x= y_true[-1][11], robot2_actual_y = y_true[-1][12],
                    robot2_blue_theta = y_true[-1][13],
                    robot3_actual_x= y_true[-1][18], robot3_actual_y = y_true[-1][19],
                    robot3_blue_theta = y_true[-1][20],
                    robot_yellow_x= y_true[-1][25], robot_yellow_y = y_true[-1][26],
                    robot_yellow_theta = y_true[-1][27],
                    robot2_yellow_x= y_true[-1][30], robot2_yellow_y = y_true[-1][31],
                    robot2_yellow_theta = y_true[-1][32],
                    robot3_yellow_x= y_true[-1][35], robot3_yellow_y = y_true[-1][36],
                    robot3_yellow_theta = y_true[-1][37],
                    return_rgb_array=True)

                frame = PIL.Image.fromarray(frame)
                frames.append(frame)
                del seq2
                del test
            print('i: ', i,'error_x: ',total_error_x)
            print('i: ', i,'error_y: ',total_error_y)
            # print('i:', i, 'mean_x': )
            frames[0].save(
            fp=f'./gifs/gif-of-dataset-test-idx-{i}-horizon-{HORIZON}-3v3.gif', 
            format='GIF', 
            append_images=frames[1:], 
            save_all=True,
            duration=40, 
            loop=0
            )
  
    def render_n_steps_autoencoder3v3(self, is_testing= False , is_to_save = True):
        render = RCGymRender(should_render_actual_ball = False, n_robots_blue = 3, n_robots_yellow = 3)
        get_preds = Pred_n_future(self.model, HORIZON)

        should_use_n_datasets = 10
        # ipdb.set_trace()
        if self.traning:
            should_use_n_datasets = 1

        indexes = random.sample(range(0, NUM_OF_DATA_TO_TEST), should_use_n_datasets)
        indexes = [0, 50, 100, 250, 450, 600, 700, 800, 900, 950]
        if is_testing:
            # indexes = [i for i in range(NUM_OF_DATA_TO_TEST)]
            indexes = [0]

        arr_loss_pos = []
        arr_loss_act = []
        arr_general_loss = []
        for i in indexes: 
            total_error_x = 0
            total_error_y = 0


            data = np.loadtxt(open('../all_data/data-3v3-v1-val' + f'/positions-{i}.txt'), dtype=np.float32)
            # data = np.loadtxt(open('../all_data/data-v4' + f'/positions-{0}.txt'), dtype=np.float32)
            if len(data) <= 30:
                continue

            out_to_test = create_window(data, WINDOW_SIZE, HORIZON)

            frames = []
            for seq, y_true in out_to_test:
                '''
                    seq.shape -> (10, 41)
                    y_true.shape -> (1, 41)
                '''
                seq2 = copy.deepcopy(seq)
                seq2 = seq2[:, :-1]
                pos, act = self.model.predict_n_steps(seq, HORIZON)
                # test = get_preds.pred_two_models(seq2, self.num_of_features)
                # pos = pos[0][-1][:]
                # act = act[0][-1][:]
                # y_copy = th.FloatTensor(y_true[-1])
                # y_copy = y_copy.to('cuda')
                # y_pos = y_copy[[0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 21, 25, 26, 30, 31, 35, 36]]
                # y_act = y_copy[[8, 9, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 32,33, 34, 37, 38, 39]]

            
                # loss_pos = F.mse_loss(pos, y_pos)
                # loss_act = F.mse_loss(act, y_act)

                # arr_loss_pos.append(loss_pos)
                # arr_loss_act.append(loss_act)
                #  # y = y[:, self.indexes]

                # general_loss = 0.7 * loss_pos +  0.3 * loss_act
                # arr_general_loss.append(general_loss)

                frame = render.render_frame(
                    ball_true_x = y_true[-1][0], ball_true_y = y_true[-1][1],
                    # ball_pred_x = test[-1][0], ball_pred_y = test[-1][1],
                    ball_pred_x = pos[0][-1][0], ball_pred_y = pos[0][-1][1],
                    # ball_pred_x = pos[0], ball_pred_y = pos[1], # using when testing
                    robot_actual_x= y_true[-1][4], robot_actual_y = y_true[-1][5],
                    robot_blue_theta = y_true[-1][6],
                    robot2_actual_x= y_true[-1][11], robot2_actual_y = y_true[-1][12],
                    robot2_blue_theta = y_true[-1][13],
                    robot3_actual_x= y_true[-1][18], robot3_actual_y = y_true[-1][19],
                    robot3_blue_theta = y_true[-1][20],
                    robot_yellow_x= y_true[-1][25], robot_yellow_y = y_true[-1][26],
                    robot_yellow_theta = y_true[-1][27],
                    robot2_yellow_x= y_true[-1][30], robot2_yellow_y = y_true[-1][31],
                    robot2_yellow_theta = y_true[-1][32],
                    robot3_yellow_x= y_true[-1][35], robot3_yellow_y = y_true[-1][36],
                    robot3_yellow_theta = y_true[-1][37],
                    return_rgb_array=True)

                frame = PIL.Image.fromarray(frame)
                frames.append(frame)
                del seq2
                del pos
                del act
                # del test

            # print('i:', i, 'mean_x': )
            if is_to_save:
                frames[0].save(
                fp=f'./gifs/gif-of-dataset-test-idx-{i}.gif',
                # fp=f'./gifs/old_way-{i}.gif',
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=40,
                loop=0
                )
        # return loss_act, loss_pos, general_loss
        del render


    def render_n_steps_one_robot_pred_3v3(self):
        render = RCGymRender(should_render_actual_ball = False, n_robots_blue = 3, n_robots_yellow = 3)
        indexes = random.sample(range(0, NUM_OF_DATA_TO_TEST), 10)
        get_preds = Pred_n_future(self.model, HORIZON)
        # testing = ConcatDataSet('../all_data/data-v2-test', NUM_OF_DATA_TO_TEST, 10, 5)
        

        for i in indexes:   
            total_error_x = 0
            total_error_y = 0
                     
            data = np.loadtxt(open('../all_data/data-3v3-v1' + f'/positions-{i}.txt'), dtype=np.float32)
            # data = np.loadtxt(open('../all_data/data-v4' + f'/positions-{0}.txt'), dtype=np.float32)
            if len(data) <= 30:
                continue

            out_to_test = create_window(data, WINDOW_SIZE, HORIZON)

            frames = []
            for seq, y_true in out_to_test:
                seq2 = copy.deepcopy(seq)

if __name__ == '__main__':
    MAX_EPOCHS = 128
    VAL_INTERVAL = 10
    WINDOW_SIZE = 10
    BATCH_SIZE = 128
    FEATURES = 16

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
		# num_of_data_to_val = 256,
		num_of_data_to_val = 1000,
    )

    # model = model.load_from_checkpoint('./next_positions/rnn-predict-next-positions-with-v3/37j6va2n/checkpoints/epoch=127-step=30463.ckpt')
    model = model.load_from_checkpoint('./next_positions-old/rnn-predict-next-positions-with-v3/2zxd4xe1/checkpoints/epoch=127-step=60927.ckpt')

    render = Handle_render(model, False, name_predictor='3fltdh5d/checkpoints/epoch=127-step=30463.ckpt',
                            name_autoenceoder='2maykc2c/checkpoints/epoch=255-step=60927.ckpt', 
                            num_of_features=NUM_OF_FEATURES).render_n_steps_autoencoder3v3()

