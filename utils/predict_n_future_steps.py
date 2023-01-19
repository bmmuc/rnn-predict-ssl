import numpy as np
import torch as th
import ipdb
from utils.convert_pred_obs import convert_pred_to_obs
class Pred_n_future:
    def __init__(self, model, n_steps) -> None:
        self.model = model
        self.n_steps = n_steps

    
    
    def predict_n_future_steps(
        self,
        positions = None
        ):
        preds = []
        self.model.hidden = (   0.01 * th.rand(1, 1, 512).float().cuda(),
                                0.01 * th.rand(1, 1, 512).float().cuda(),
                            )

        with th.no_grad():
            for _ in range(self.n_steps):

                positions = th.FloatTensor(positions).cuda()
                    
                new_pos = positions.view((1,1,positions.shape[0]*positions.shape[1])).float()

                out = self.model(new_pos)
                    
                positions = positions.cpu().detach().squeeze().numpy()
                # ipdb.set_trace()
                positions = self.update_list(positions, out.cpu().detach().numpy())
                # out = self.undo_standardization(out.cpu().detach().numpy())
                # out = np.squeeze(out)
                preds.append(out.squeeze().detach().cpu().numpy())
                # preds.append(out.tolist())
        return preds


    def predict_n_future_steps2(
        self,
        positions = None
        ):
        preds = []

        with th.no_grad():
            for _ in range(self.n_steps):

                positions = th.FloatTensor(positions).cuda()
                # new_pos = positions.view((1,1,positions.shape[0]*positions.shape[1])).float()
                
                # ipdb.set_trace()
                positions = positions.view(1, 16 , 10).float()
                # positions = positions.view((positions.shape[0],positions.shape[1],1)).float()
                
                
                out = self.model(positions)
                positions = positions.view(10, 16)
                positions = positions.cpu().detach().squeeze().numpy()
                # ipdb.set_trace()

                positions = self.update_list2(positions, out.cpu().detach().numpy())
                
                # positions = out.cpu().detach().numpy()
                preds.append(out.squeeze().detach().cpu().numpy())

        return preds

    def predict_n_future_steps3(
        self,
        positions = None,
        n_features = 41,
        ):
        preds = []

        with th.no_grad():
            for _ in range(self.n_steps):

                positions = th.FloatTensor(positions).cuda()
                # new_pos = positions.view((1,1,positions.shape[0]*positions.shape[1])).float()
                
                # ipdb.set_trace()
                # if positions.shape[1] == 41:
                    # positions = positions[:, :-1]
                    # positions = positions.view(1, 16 , 10).float()
                positions = positions.reshape(1, n_features , 10).float()
                # positions = positions.view((positions.shape[0],positions.shape[1],1)).float()
                
                
                out = self.model(positions)
                positions = positions.view(10, n_features)
                positions = positions.cpu().detach().squeeze().numpy()
                # ipdb.set_trace()

                positions = self.update_list2(positions, out.cpu().detach().numpy())
                
                # positions = out.cpu().detach().numpy()
                preds.append(out.squeeze().detach().cpu().numpy())

        return preds
    
    def test_pos_latent(self, positions = None, n_features=41):
        preds=[]
        indexes = [0,1,2,3,4,5,6,7,11,12,13,14,18,19,20,21,27,28,30,31,35,36]
        with th.no_grad():
            for _ in range(self.n_steps):
                # ipdb.set_trace()
                positions = th.FloatTensor(positions).cuda()
                positions = positions.reshape(1, 10 , n_features).float()
                out = self.model(positions)
                x_copy_act = positions.clone()
                x_copy_act = x_copy_act[:, :, indexes]
                out = self.model.decoding(out, x_copy_act)

                positions = positions.view(10, n_features)
                positions = positions.cpu().detach().squeeze().numpy()

                # positions = self.update_list2(positions, out.cpu().detach().numpy())

                preds.append(out.squeeze().detach().cpu().numpy())
        
        return preds

    def new_preds(self, positions = None, n_features=41):
        preds=[]
        with th.no_grad():
            for _ in range(self.n_steps):
                # ipdb.set_trace()
                positions = th.FloatTensor(positions).cuda()
                positions = positions.reshape(1, 10 , n_features).float()
                out = self.model(positions)
                acts = self.model.get_actions(positions)
                acts = acts.squeeze()
                acts = acts[-1,:]
                acts = acts.reshape(1,-1).float()
                out = th.cat((out, acts), dim=1)

                positions = positions.view(10, n_features)
                positions = positions.cpu().detach().squeeze().numpy()

                positions = self.update_list2(positions, out.cpu().detach().numpy())

                preds.append(out.squeeze().detach().cpu().numpy())

        return preds

    def pred_two_models(self, positions = None, n_features=40):
        preds=[]
        with th.no_grad():
            for _ in range(self.n_steps):
                # ipdb.set_trace()
                positions = th.FloatTensor(positions).cuda()
                positions = positions.reshape(1, 10 , n_features).float()
                
                pred_pos, pred_act = self.model(positions)
                pred_pos = pred_pos.squeeze().detach().cpu().numpy()
                pred_act = pred_act.squeeze().detach().cpu().numpy()
                obs = convert_pred_to_obs(pred_pos, pred_act)

                # out = th.cat((pred_pos, pred_act), dim=1)
                positions = positions.view(10, n_features)
                positions = positions.cpu().detach().squeeze().numpy()
                # obs_ = th.FloatTensor(obs).cuda()
                positions = self.update_list3(positions, obs)

                preds.append(obs)

        return preds

    def predict_n_future_steps_one_robot(
        self,
        positions = None
        ):
        preds = []
        test = []
        with th.no_grad():
            ipdb.set_trace()
            for _ in range(self.n_steps):
                    # ball_values = copy.deepcopy(X)
                    # robot_to_use = randint(0, 2)

                    # robot_values = copy.deepcopy(X)

                    # aux = robot_values[:, 4:11, :]

                    # robot_values[:, 4:11, :] = robot_values[:, robot_to_use:robot_to_use + 7, :]
                    # robot_values[:, robot_to_use + 7: robot_to_use + 14, :] = aux

                    # # X = th.cat((, robot_values), dim=1)


                positions = positions.cpu().detach().squeeze().numpy()
                # ipdb.set_trace()
                positions = self.update_list(positions, out.cpu().detach().numpy())
                # out = self.undo_standardization(out.cpu().detach().numpy())
                # out = np.squeeze(out)
                # preds.append(out.squeeze().detach().cpu().numpy())
        #TODO
    def update_list3(self, old_list, new_element):
        # ipdb.set_trace()
        for i in range(0, len(old_list) - 1):
            old_list[i] = old_list[i + 1]
        # ipdb.set_trace()
        # ipdb.set_trace()
        old_list[-1] = new_element
        
        return old_list
    def update_list2(self, old_list, new_element):
        ipdb.set_trace()
        for i in range(0, len(old_list) - 1):
            old_list[i] = old_list[i + 1]
        # ipdb.set_trace()
        # ipdb.set_trace()
        old_list[-1] = new_element[-1,:]
        
        return old_list


    def update_list(self, old_list, new_element):
        # ipdb.set_trace()

        for i in range(0, len(old_list) - 1):
            old_list[i] = old_list[i + 1]
        # ipdb.set_trace()
        old_list[-1] = new_element
        
        return old_list
