from torch.utils.data import Dataset
import numpy as np
import os
import torchvision.transforms as trnsfs
import torchvision.transforms as transforms
import torch

class MovingMnist(Dataset):

    def __init__(self, data_path, downsampled=False):

        super(MovingMnist, self).__init__()
        # data-shape: [frame, seq, W, H]
        self.data_path = data_path
        self.downsampled = True
        if not downsampled:
            self.data = np.load(os.path.join(self.data_path, 'mnist_test_seq.npy'))
        else:
            self.data = np.load(os.path.join(self.data_path, 'mnist_test_seq_28.npy'))

        self.train_data, self.valid_data, self.test_data = self.create_train_valid_test_split(self.data)
        self.transform = trnsfs.Resize(28)
        self.toPil = transforms.ToPILImage()
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, item):

        img = np.squeeze(self.train_data[item['rand_img'], item['rand_seq_img'], :, :])
        seq = np.squeeze(self.train_data[:, item['rand_seq'], :, :])
        # downsample to 28x28
        return {'img': self.toTensor(self.transform(self.toPil(img))), 'seq': seq}

    def __len__(self):
        return self.train_data.shape[0]*self.train_data.shape[1]

    def create_train_valid_test_split(self, data):

        train_data_len = int(data.shape[1]*0.8)
        valid_data_len = int(data.shape[1]*0.1)
        test_data_len = int(data.shape[1]*0.1)

        # ToDO add random sequences
        train_data = data[:, 0:train_data_len, :, :]
        valid_data = data[:, train_data_len:(valid_data_len+train_data_len), :, :]
        test_data = data[:, (train_data_len+valid_data_len):(train_data_len+valid_data_len+test_data_len), :, :]
        return train_data, valid_data, test_data

    def get_valid_batch(self, batch_size):

        assert self.downsampled is True

        seq_length = self.valid_data.shape[0]
        num_seq = self.valid_data.shape[1]

        rand_imgs = np.random.randint(0, seq_length-1, batch_size)
        rand_seq = np.random.randint(0, num_seq-1, batch_size)
        return torch.tensor(self.valid_data[rand_imgs, rand_seq, :, :])

    def get_test_batch(self, batch_size):
        pass

    def get_num_sequences(self):
        return self.train_data.shape[1]

    def get_seq_length(self):
        return self.train_data.shape[0]

