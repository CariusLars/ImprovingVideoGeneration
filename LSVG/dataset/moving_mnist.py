from torch.utils.data import Dataset
import numpy as np
import os
import torchvision.transforms as trnsfs
import torchvision.transforms as transforms
import torch


class MovingMnist(Dataset):

    def __init__(self, data_path, downsampled=False):
        """
        :param data_path: path to .npy file containing the dataset
        :param downsampled: if the dataset is already downsampled. If not, get_valid_batch is not available
        """
        super(MovingMnist, self).__init__()
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
        """
        Return one item, i.e. one image, of the dataset
        :param item: {rand_img:int, rand_seq_img:int, rand_seq:int}
        :return: returns one image as tensor and downsampled to 28x28 pixels
        """
        img = np.squeeze(self.train_data[item['rand_img'], item['rand_seq_img'], :, :])
        seq = np.squeeze(self.train_data[:, item['rand_seq'], :, :])

        return {'img': self.toTensor(self.transform(self.toPil(img))), 'seq': seq}

    def __len__(self):
        return self.train_data.shape[0]*self.train_data.shape[1]

    def create_train_valid_test_split(self, data):
        """
        Splits the data into train, validation and test data
        :param data: the data as np.array
        :return: returns train_data, valid_data, test_data as np.arrays
        """
        train_data_len = int(data.shape[1]*0.8)
        valid_data_len = int(data.shape[1]*0.1)
        test_data_len = int(data.shape[1]*0.1)

        train_data = data[:, 0:train_data_len, :, :]
        valid_data = data[:, train_data_len:(valid_data_len+train_data_len), :, :]
        test_data = data[:, (train_data_len+valid_data_len):(train_data_len+valid_data_len+test_data_len), :, :]

        return train_data, valid_data, test_data

    def get_valid_batch(self, batch_size):
        """
        This creates one validation batch. It cannot downsample images, therefore downsample=True
        :param batch_size: size of the batch
        :return: batch [batch_size, 28, 28] as np.array
        """
        assert self.downsampled is True

        seq_length = self.valid_data.shape[0]
        num_seq = self.valid_data.shape[1]

        rand_imgs = np.random.randint(0, seq_length-1, batch_size)
        rand_seq = np.random.randint(0, num_seq-1, batch_size)
        return torch.tensor(self.valid_data[rand_imgs, rand_seq, :, :])

    def get_num_sequences(self):
        return self.train_data.shape[1]

    def get_seq_length(self):
        return self.train_data.shape[0]
