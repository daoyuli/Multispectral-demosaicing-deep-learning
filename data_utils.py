import torch
from os import listdir
from os.path import join
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np


class dataset(Dataset):

    def __init__(self, input_dir, truth_dir):
        super(dataset, self).__init__()
        self.input_filenames = [join(input_dir, x) for x in listdir(input_dir)]
        self.truth_filenames = [join(truth_dir, x) for x in listdir(truth_dir)]

    def __getitem__(self, index):
        input = sio.loadmat(self.input_filenames[index])['meas']
        truth = sio.loadmat(self.truth_filenames[index])['orig_subscene']
        input_t = torch.from_numpy(input.astype('float32')).unsqueeze(0)
        truth_n = np.zeros((truth.shape[-1], truth.shape[0], truth.shape[1]))
        for it in range(truth.shape[-1]):
            truth_n[it, :, :] = truth[:, :, it]
        truth_t = torch.from_numpy(truth_n.astype('float32'))
        return input_t, truth_t

    def __len__(self):
        return len(self.truth_filenames)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    input = sio.loadmat('./data/input_9/MosaicScene_9channels_1.mat')['meas']
    truth = sio.loadmat('./data/output_9/OrigScene_9channels_1.mat')['orig_subscene']
    plt.imshow(input, 'gray')
    plt.show()
    input_t = torch.from_numpy(input.astype('float32')).unsqueeze(0)
    truth_n = np.zeros((truth.shape[-1], truth.shape[0], truth.shape[1]))
    for it in range(truth.shape[-1]):
        truth_n[it, :, :] = truth[:, :, it]
    truth_t = torch.from_numpy(truth_n.astype('float32'))
    print(truth_t)
