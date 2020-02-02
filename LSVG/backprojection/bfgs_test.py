import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from backprojection.bfgs import BFGSProjector
import torchvision.datasets as dset
import torchvision.transforms as transforms
from dataset.moving_mnist import MovingMnist
from dataset.moving_mnist_sampler import MnistSampler
from torch.utils.data import DataLoader
import numpy as np
DIM = 64


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, 1, 28, 28)


#dataset = dset.MNIST(root='./mnist/', download=True,
#                         transform=transforms.Compose([
#                            transforms.Resize(28),
#                            transforms.ToTensor()
#                           ]))
def create_train_valid_test_split(data):
    train_data_len = int(data.shape[1]*0.8)
    valid_data_len = int(data.shape[1]*0.1)
    test_data_len = int(data.shape[1]*0.1)
    train_data = data[:, 0:train_data_len, :, :]
    valid_data = data[:, train_data_len:(valid_data_len+train_data_len), :, :]
    test_data = data[:, (train_data_len+valid_data_len):(train_data_len+valid_data_len+test_data_len), :, :]
    return train_data, valid_data, test_data

data = MovingMnist('/Users/michaelgentnermac/Documents/ADL4CV')
sampler = MnistSampler(data)

loader = DataLoader(data, 1, sampler=sampler)

if __name__ == '__main__':

    #netG = Generator()
    #netG.double()
    #netG.load_state_dict(torch.load('/Users/michaelgentnermac/Downloads/g.pt', map_location='cpu'))
    data = np.load('/Users/michaelgentnermac/Documents/ADL4CV/mnist_test_seq.npy')
    train_data, valid_data, test_data = create_train_valid_test_split(data)
    transform = transforms.Resize(28)
    toPil = transforms.ToPILImage()
    toTensor = transforms.ToTensor()
    netG = torch.load('/Users/michaelgentnermac/Documents/ADL4CV/10.01.2020-23_47_23/models/g.pt', map_location=torch.device('cpu'))
    netG.double()
    #loader = DataLoader(dataset, shuffle=True, batch_size=1, drop_last=True)
    #img = next(iter(loader))['img'].view(1, 1, 28, 28)
    img = toTensor(transform(toPil(test_data[0, 400, :, :]))).view(1, 1, 28, 28)
    print(img)
    print(img.shape)
    projector = BFGSProjector(netG, None, 256)

    z = projector.project(img[0].double())
    res_img = netG(torch.from_numpy(z.x))

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(res_img.detach().squeeze())
    fig.add_subplot(1, 2, 2)
    plt.imshow(img[0].squeeze())
    plt.show()
    print(z)
