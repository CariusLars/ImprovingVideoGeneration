from torch.utils.data import Sampler
import numpy as np


class MnistSampler(Sampler):

    def __init__(self, mnist_dataset):
        """
        :param mnist_dataset: mnist dataset object
        """
        super(MnistSampler, self).__init__(mnist_dataset)
        self.dataset = mnist_dataset

    def __iter__(self):
        """
        This yields one random index of the dataset.
        :return: {rand_seq_img: int, rand_img:int, rand_seq:int}
        """
        rand_sequences_img = np.random.randint(0, self.dataset.get_num_sequences(), size=len(self.dataset))
        rand_sequences = np.random.randint(0, self.dataset.get_num_sequences(), size=len(self.dataset))
        rand_img = np.random.randint(0, self.dataset.get_seq_length(), size=len(self.dataset))

        for i in range(rand_img.shape[0]):

            yield {'rand_seq_img': rand_sequences_img[i], 'rand_img': rand_img[i], 'rand_seq': rand_sequences[i]}

    def __len__(self):
        return len(self.dataset)
