import numpy as np
import tqdm
import os
import random
import ipdb
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
                window = 50,
                type_of_data = 'train',
                horizon = 1,
                ):

        self.root_dir = root_dir
        self.window = window
        self.horizons = horizon



        self.type = type_of_data

        # if type_of_data == 'train':
        self.files_to_read = os.listdir(self.root_dir)
        # else:
            # self.files_to_read = os.listdir(self.root_dir)[len(os.listdir(self.root_dir))//2:]

        self.data_windows = list()
        self.data_labels = list()
        self.init_dataset()

    def init_dataset(self):
        loaded_datas = []
        for file in tqdm.tqdm(self.files_to_read, desc=f'Creating {self.type}_DataSet'):
            data = np.load(f'{self.root_dir}/{file}', allow_pickle=True)
            loaded_datas.append(data)
        # ipdb.set_trace()
        loaded_datas = np.concatenate(loaded_datas, axis=0)
        if self.type == 'train':
            loaded_datas = loaded_datas[:int(len(loaded_datas) * 0.8)]
        else:
            loaded_datas = loaded_datas[int(len(loaded_datas) * 0.8):]
        # loaded_datas = np.concatenate(loaded_datas, axis=0)
        data = create_window(loaded_datas, self.window, self.horizons)

        for window, label in data:
            # if(len(window[0]) == 41):
            # ipdb.set_trace()
            self.data_windows.append(window)
            self.data_labels.append(label)
                # assert False, f'error no tamanho, {len(window[0])}, no data_set: de id {i}'
            # else:
            #     self.data_windows.append(window)
            #     self.data_labels.append(label)

        self.data_windows, self.data_labels = self._shuffle(self.data_windows, self.data_labels)

        self.data_windows = np.array(self.data_windows)
        self.data_labels = np.array(self.data_labels)
        print(f'Creating {self.type}_DataSet: {self.data_windows.shape}, {self.data_labels.shape}')
        np.save(f'{self.type}_data_windows.npy', self.data_windows, allow_pickle=True)
        np.save(f'{self.type}_data_labels.npy', self.data_labels, allow_pickle=True)

        del data

    def _shuffle(self, X, y):
        joined_lists = list(zip(X, y))
        random.shuffle(joined_lists)
        
        X, y = zip(*joined_lists)
        
        return X, y

if __name__ == '__main__':
    root_dir = '/home/bmmuc/Documents/robocin/rnn/rnn-predict-ssl/log_raw'
    CreateDataSetFile(root_dir, window=50, type_of_data='train', horizon=1)
    CreateDataSetFile(root_dir, window=50, type_of_data='val', horizon=1)
