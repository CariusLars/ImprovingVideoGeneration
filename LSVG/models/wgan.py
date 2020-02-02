import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, latent_dim, dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.dim = dim

        preprocess = nn.Sequential(
            nn.Linear(self.latent_dim, 4*4*4*self.dim),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*self.dim, 2*self.dim, 5),
            nn.ReLU(True),
            nn.Conv2d(2*self.dim, 2 * self.dim, 3, padding=1),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*self.dim, self.dim, 5),
            nn.ReLU(True),
            nn.Conv2d(self.dim, self.dim, 3, padding=1),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(self.dim, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.preprocess(x)
        output = output.view(-1, 4*self.dim, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)

        return output.view(-1, 1, 28, 28)


class Discriminator(nn.Module):

    def __init__(self, latent_dim, dim):
        super(Discriminator, self).__init__()

        self.latent_dim = latent_dim
        self.dim = dim

        main = nn.Sequential(
            nn.Conv2d(1, self.dim, 5, stride=2, padding=2),
            nn.InstanceNorm2d(self.dim, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(self.dim, self.dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.dim, 2*self.dim, 5, stride=2, padding=2),
            nn.InstanceNorm2d(self.dim*2, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(2*self.dim, 2*self.dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2*self.dim, 4*self.dim, 5, stride=2, padding=2),
            nn.InstanceNorm2d(4*self.dim, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(4*self.dim, 4*self.dim, 3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.main = main
        self.output = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        out = self.main(x)
        out = out.view(-1, 4*4*4*self.dim)
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
