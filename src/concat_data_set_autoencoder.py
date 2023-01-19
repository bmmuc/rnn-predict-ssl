from torch.utils.data import Dataset
import numpy as np
# from utils import *
import sys
import random
import tqdm

# from create_window import create_window
def create_window(data, window_size, horizon = 1):
        out = []
        L = len(data)

        for i in range(L - window_size - horizon):
            window = data[i : i+window_size, :]
            label = data[i+window_size : i+window_size+horizon, :]
            out.append((window,label))
        return out
import ipdb

class ConcatDataSetAutoencoder(Dataset):
    """Multi-variate Time-Series ConcatDataset for *.txt file
    This Dataset concatentes anothers datasets by idx of the dataset
    then creates a batch from this dataset.

    Co-Author: @goncamateus
    Returns:
        [sample, next_sample]
    """
    def __init__(self, 
                root_dir =  '', 
                num_of_data_sets = 50000,
                window = 5,
                type_of_data = 'train',
                horizon = 1,
                is_pos = True,
                should_test_overffit = False,
                should_test_the_new_data_set = True,
                is_to_debug = False):

        self.root_dir = root_dir
        self.num_of_data_sets = num_of_data_sets
        self.window = window
        self.is_pos = is_pos
        self.indexes_pos = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 18, 19, 20, 21, 25, 26, 30, 31, 35, 36]
        self.bool_indexes = [i in self.indexes_pos for i in range(42)]
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
        if not self.should_test_the_new_data_set:
            for i in tqdm.tqdm(range(self.num_of_data_sets), desc=f'Creating {self.type}_DataSet'):
                data = np.loadtxt(open(self.root_dir + f'/positions-{i}.txt'), dtype=np.float32)
                data = create_window(data, self.window, 1)
                for window, _ in data:
                    if(len(window[0]) == 41):
                        # ipdb.set_trace()
                        self.data_windows.append(window[:, :-1])
                        self.data_labels.append(window[:, :-1])
                        # assert False, f'error no tamanho, {len(window[0])}, no data_set: de id {i}'
                    else:
                        self.data_windows.append(window)
                        self.data_labels.append(window)


            _data_windows, self.data_labels = self._shuffle(self.data_windows, self.data_labels)

            self.data_windows = np.array(_data_windows)
            self.data_labels = np.array(self.data_labels)

            del data

        else:
            print(f'Loading the {self.type} data set...')

            self.data_windows = np.load(f'./{self.type}_data_windows.npy')

            print(f'Loaded the {self.type} data set')

    def __getitem__(self, idx):
        if(self.is_pos):
            X = self.data_windows[idx, :, self.bool_indexes]
            X = X.reshape(-1, 22)

            y = self.data_windows[idx, :, self.bool_indexes]
            y = y.reshape(-1, 22)
        else:
            # ipdb.set_trace()
            X = self.data_windows[idx, :, 40:42]
            y = self.data_windows[idx, :, 40:42]


        # X = self.data_windows[idx]
        # y = self.data_windows[idx]
        # return 0
        return [X, y]    

    def __len__(self):
        return self.num_of_data_sets


if __name__ == '__main__':

    test = ConcatDataSetAutoencoder(root_dir= '/home/bmmuc/Documents/robocin/rnn/all_data/data-v4', num_of_data_sets= 100, horizon=1, is_to_debug=True,
        is_pos=False)
    test.__getitem__(0)
    # test.__getitem__(40602)lf.root_dir + f'/positions-{i}.txt'), dtype=np.float32)
