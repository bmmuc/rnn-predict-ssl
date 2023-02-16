from torch.utils.data import Dataset
import numpy as np


class ConcatDataSetSsl(Dataset):
    """Multi-variate Time-Series ConcatDataset for *.txt file
    This Dataset concatentes anothers datasets by idx of the dataset
    then creates a batch from this dataset.

    Co-Author: @goncamateus
    Returns:
        [sample, next_sample]
    """
    def __init__(self, 
                type_of_data = 'train',
                is_autoencoder = False):
        self.data_windows = list()
        self.data_labels = list()   
        self.type = type_of_data
        self.is_autoencoder = is_autoencoder
        # self.num_of_windows = 0


        self.init_dataset()

    def init_dataset(self):

        print(f'Loading the {self.type} data set...')

        self.data_windows = np.load(f'./{self.type}_data_windows.npy', allow_pickle=True)
        self.data_labels = np.load(f'./{self.type}_data_labels.npy', allow_pickle=True)
        # change type of data to float32
        self.data_windows = self.data_windows.astype(np.float32)
        if not self.is_autoencoder:
            self.data_labels = self.data_labels.astype(np.float32)

        print(f'Loaded the {self.type} data set')

    def __getitem__(self, idx):

        X = self.data_windows[idx]
        # X = np.array(X)
        if self.is_autoencoder:
            y = X
        else:
            y = self.data_labels[idx][:2]
            # y = np.array(y)
        return [X, y]

    def __len__(self):
        return len(self.data_windows)


if __name__ == '__main__':
    dataset = ConcatDataSetSsl(type_of_data='train')
    item = dataset[0]

    print(item[0])
    print(item[1])
    print('------------------')
    print(item[0].dtype)
    print(item[1].dtype)
    print('------------------')
    print(item[0].shape)
    print(item[1].shape)
    print('------------------')
    print(len(dataset))
