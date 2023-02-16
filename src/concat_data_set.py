from torch.utils.data import Dataset
import numpy as np
# from utils import *
import sys
import random
import tqdm

sys.path.insert(1, './utils')
from create_window import create_window

import ipdb
class ConcatDataSet(Dataset):
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
                should_test_overffit = False,
                should_test_the_new_data_set = False,
                is_to_debug = False):

        self.root_dir = root_dir
        self.num_of_data_sets = num_of_data_sets
        self.window = window
        self.should_test_the_new_data_set = should_test_the_new_data_set
        self.should_test_overffit = should_test_overffit
        self.horizons = horizon
        self.is_to_debug = is_to_debug
        self.data_windows = list()
        self.data_labels = list()   
        self.type = type_of_data
        # self.num_of_windows = 0


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
                data = create_window(data, self.window, self.horizons)

                for window, label in data:
                    self.data_windows.append(window)
                    self.data_labels.append(label)


                # self.data_set.append((window, label))

        # random.shuffle(self.data_set)

            self.data_windows, self.data_labels = self._shuffle(self.data_windows, self.data_labels)
            
            self.data_windows = np.array(self.data_windows)
            self.data_labels = np.array(self.data_labels)
            del data

        else:
            print(f'Loading the {self.type} data set...')

            self.data_windows = np.load(f'./{self.type}_data_windows_no_reward.npy')
            self.data_labels = np.load(f'./{self.type}_data_labels_no_reward.npy')
            
            print(f'Loaded the {self.type} data set')

    def __getitem__(self, idx):

        X = self.data_windows[idx]
        y = self.data_labels[idx]

        return [X, y]

    def __len__(self):
        return self.num_of_data_sets


if __name__ == '__main__':

    test = ConcatDataSet(root_dir= '../all_data/data-v4', num_of_data_sets= 100, horizon=1, is_to_debug=True)
    test.__getitem__(0)
    # test.__getitem__(40602)lf.root_dir + f'/positions-{i}.txt'), dtype=np.float32)
