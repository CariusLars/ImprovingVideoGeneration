import torch.nn as nn
import torch

DIM = 128
latent_dim = 16


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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.InstanceNorm2d(DIM, affine=True),
            #nn.ReLU(True),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, DIM, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            #nn.Conv2d(2*DIM, 2*DIM, 3, stride=1, padding=1),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.InstanceNorm2d(DIM*2, affine=True),
            #nn.ReLU(True),
            nn.LeakyReLU(),
            nn.Conv2d(2*DIM, 2*DIM, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            #nn.Conv2d(4*DIM, 4*DIM, 3, stride=1, padding=1),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.InstanceNorm2d(4*DIM, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(4*DIM, 4*DIM, 3, stride=1, padding=1),
            nn.LeakyReLU()
            #nn.ReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1)


def weight_init(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        net.weight.data.normal_(0.0, 0.02)
        net.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        net.weight.data.normal_(1.0, 0.02)
        net.bias.data.fill_(0)

