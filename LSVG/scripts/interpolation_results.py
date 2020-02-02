import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from backprojection.bfgs import BFGSProjector
from interpolate_latent.functions import latentVectorExtrapolateForward
import torchvision.transforms as transforms
import numpy as np

SEQ = 554
DIM = 64 # Model dimensionality
latent_dim = 256

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(latent_dim, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 2 * DIM, 3, padding=1),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
            nn.Conv2d(DIM, DIM, 3, padding=1),
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
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        return output.view(-1, 1, 28, 28)


def create_train_valid_test_split(data):
    train_data_len = int(data.shape[1]*0.8)
    valid_data_len = int(data.shape[1]*0.1)
    test_data_len = int(data.shape[1]*0.1)
    train_data = data[:, 0:train_data_len, :, :]
    valid_data = data[:, train_data_len:(valid_data_len+train_data_len), :, :]
    test_data = data[:, (train_data_len+valid_data_len):(train_data_len+valid_data_len+test_data_len), :, :]
    return train_data, valid_data, test_data



if __name__ == '__main__':

    data = np.load('/Users/michaelgentnermac/Documents/ADL4CV/mnist_test_seq.npy')
    train_data, valid_data, test_data = create_train_valid_test_split(data)
    transform = transforms.Resize(28)
    toPil = transforms.ToPILImage()
    toTensor = transforms.ToTensor()
    netG = torch.load('/Users/michaelgentnermac/Documents/ADL4CV/10.01.2020-23_47_23/models/g.pt',
                      map_location=torch.device('cpu'))
    netG.double()
    projector = BFGSProjector(netG, None, 256)
    sequence = test_data[:, SEQ, :, :]

    seq_length = sequence.shape[0]
    projected_images = []
    for i in range(2):

        img = toTensor(transform(toPil(sequence[i, :, :]))).double()
        z = projector.project(img)
        projected_images.append(z.x)
        print(i)

    projected_images = np.array(projected_images)
    print(projected_images.shape)

    projected_images = toTensor(projected_images)

    res_imgs = netG(projected_images).detach().numpy()

    res_imgs = res_imgs.squeeze()

    print(res_imgs.shape)

    fig = plt.figure(figsize=(25, 3))
    columns = 2
    rows = 1
    for i in range(0, 2):
        img = res_imgs[i, :, :]
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    plt.show()
    # Choose first and last frame of video
    #input_frames = [toTensor(transform(toPil(test_data[0, SEQ, :, :]))),
    #                toTensor(transform(toPil(test_data[9, SEQ, :, :])))]

