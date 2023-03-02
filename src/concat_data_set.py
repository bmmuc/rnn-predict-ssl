from torch.utils.data import Dataset
import numpy as np
# from utils import *
import random

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
                root_dir =  './datasets_full_games/', 
                type_of_data = 'train',
                ):

        self.root_dir = root_dir
        self.data_windows = list()
        self.data_labels = list()   
        self.type = type_of_data

        self.init_dataset()

    def _shuffle(self, X, y):
        joined_lists = list(zip(X, y))
        random.shuffle(joined_lists)
        
        X, y = zip(*joined_lists)
        
        return X, y

    def init_dataset(self):
        print(f'Loading the {self.type} data set...')

        self.data_windows = np.load(f'{self.root_dir}/{self.type}_data_windows_full_games.npy')
        self.data_labels = np.load(f'{self.root_dir}/{self.type}_data_labels_full_games.npy')

        print(f'Loaded the {self.type} data set')

    def __getitem__(self, idx):

        X = self.data_windows[idx]
        y = self.data_labels[idx]

        return [X, y]

    def __len__(self):
        return len(self.data_windows)


if __name__ == '__main__':

    test = ConcatDataSet(root_dir= '../all_data/data-v4', num_of_data_sets= 100, horizon=1, is_to_debug=True)
    test.__getitem__(0)
    # test.__getitem__(40602)lf.root_dir + f'/positions-{i}.txt'), dtype=np.float32)
