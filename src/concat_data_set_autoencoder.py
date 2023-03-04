from torch.utils.data import Dataset
import numpy as np
# from utils import *
import sys
import random
import tqdm
from src.aux_idx import Aux
import ipdb
# from create_window import create_window


def create_window(data, window_size, horizon=1):
    out = []
    L = len(data)

    for i in range(L - window_size - horizon):
        window = data[i: i+window_size, :]
        label = data[i+window_size: i+window_size+horizon, :]
        out.append((window, label))
    return out


class ConcatDataSetAutoencoder(Dataset):
    """Multi-variate Time-Series ConcatDataset for *.txt file
    This Dataset concatentes anothers datasets by idx of the dataset
    then creates a batch from this dataset.

    Co-Author: @goncamateus
    Returns:
        [sample, next_sample]
    """

    def __init__(self,
                 root_dir='',
                 num_of_data_sets=50000,
                 window=5,
                 type_of_data='train',
                 horizon=1,
                 is_pos=True,
                 should_test_overffit=False,
                 should_test_the_new_data_set=True,
                 is_to_debug=False):

        self.root_dir = root_dir
        self.num_of_data_sets = num_of_data_sets
        self.window = window
        self.is_pos = is_pos
        self.indexes_pos = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12,
                            13, 14, 18, 19, 20, 21, 25, 26, 30, 31, 35, 36]
        aux_index = Aux.is_vel
        self.bool_indexes = []

        for value in aux_index:
            self.bool_indexes.append(not value)

        self.actions_indexes = []
        for i in aux_index:
            self.actions_indexes.append(i)

        self.should_test_overffit = should_test_overffit
        self.horizons = horizon
        self.should_test_the_new_data_set = should_test_the_new_data_set
        self.is_to_debug = is_to_debug
        self.data_windows = list()
        self.data_labels = list()
        self.type = type_of_data

        print(root_dir)

        self.init_dataset()

    def _shuffle(self, X, y):
        joined_lists = list(zip(X, y))
        random.shuffle(joined_lists)

        X, y = zip(*joined_lists)

        return X, y

    def init_dataset(self):
        print(f'Loading the {self.type} data set...')
        # if self.is_pos:
        #     self.data_windows = np.load(f'./datasets/{self.type}_data_windows.npy', allow_pickle=True)
        # else:
        self.data_windows = np.load(
            f'./datas/{self.window}/{self.type}_data_windows_full_games_reduced.npy', allow_pickle=True)
        # ipdb.set_trace()
        print(f'Loaded the {self.type} data set')

    def __getitem__(self, idx):
        # ipdb.set_trace()
        # ipdb.set_trace()
        if (self.is_pos):
            # X = self.data_windows[idx, :, self.bool_indexes]
            # X = X.reshape(-1, X.shape[0]).astype(np.float32)

            # y = self.data_windows[idx, :, self.bool_indexes]
            # y = y.reshape(-1, y.shape[0]).astype(np.float32)
            X = self.data_windows[idx, :, self.bool_indexes]
            # X = X.reshape(-1, 38).astype(np.float32)

            # y = self.data_windows[idx, :, :]
            # y = y.reshape(-1, y.shape[0]).astype(np.float32)
        else:
            # ipdb.set_trace()
            X = self.data_windows[idx, :, self.actions_indexes]
            # X = X.reshape(-1, X.shape[0]).astype(np.float32)

            # y = self.data_windows[idx, :, self.actions_indexes]
            # y = y.reshape(-1, y.shape[0]).astype(np.float32)

        # X = self.data_windows[idx]
        # y = self.data_windows[idx]
        # return 0
        # print(X.shape, y.shape)
        X = X.reshape(-1, X.shape[0]).astype(np.float32)
        return [X, X]

    def __len__(self):
        return len(self.data_windows)


if __name__ == '__main__':

    test = ConcatDataSetAutoencoder(root_dir='/home/bmmuc/Documents/robocin/rnn/all_data/data-v4', num_of_data_sets=100, horizon=1, is_to_debug=True,
                                    is_pos=False)
    test = ConcatDataSetAutoencoder(root_dir='/home/bmmuc/Documents/robocin/rnn/all_data/data-v4', num_of_data_sets=100, horizon=1, is_to_debug=True,
                                    is_pos=True)
    test.__getitem__(0)
    # test.__getitem__(40602)lf.root_dir + f'/positions-{i}.txt'), dtype=np.float32)
