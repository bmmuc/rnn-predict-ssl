import numpy as np
import sys
import random
import tqdm

sys.path.insert(1, './utils')
def create_window(data, window_size, horizon = 1):
        out = []
        L = len(data)

        for i in range(L - window_size - horizon):
            window = data[i : i+window_size, :]
            label = data[i+window_size : i+window_size+horizon, :]
            out.append((window,label))
        return out

class CreateDataSetFile():

    def __init__(self, 
                root_dir =  '', 
                num_of_data_sets = 50000,
                window = 10,
                type_of_data = 'train',
                horizon = 1,
                should_test_overffit = False,
                is_to_debug = False):

        self.root_dir = root_dir
        self.num_of_data_sets = num_of_data_sets
        self.window = window
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

        for i in tqdm.tqdm(range(self.num_of_data_sets), desc=f'Creating {self.type}_DataSet'):
            data = np.loadtxt(open(self.root_dir + f'/positions-{1}.txt'), dtype=np.float32, delimiter=',')
            data = create_window(data, self.window, self.horizons)

            for window, label in data:
                if(len(window[0]) == 41):
                    # ipdb.set_trace()
                    self.data_windows.append(window[:, :-1])
                    self.data_labels.append(label[:, :-1])
                    # assert False, f'error no tamanho, {len(window[0])}, no data_set: de id {i}'
                else:
                    self.data_windows.append(window)
                    self.data_labels.append(label)

        self.data_windows, self.data_labels = self._shuffle(self.data_windows, self.data_labels)

        self.data_windows = np.array(self.data_windows)
        self.data_labels = np.array(self.data_labels)

        np.save(f'{self.type}_data_windows.npy', self.data_windows, allow_pickle=False)
        np.save(f'{self.type}_data_labels_.npy', self.data_labels, allow_pickle=False)

        del data

if __name__ == '__main__':

    test = CreateDataSetFile(root_dir= '../all_data/data-v5', 
                                num_of_data_sets = 1, 
                                window= 10,
                                type_of_data = 'val',
                                horizon=1)
    # test.__getitem__(0)
    # test.__getitem__(40602)lf.root_dir + f'/positions-{i}.txt'), dtype=np.float32)
