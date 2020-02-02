from dataset.moving_mnist import MovingMnist
from dataset.moving_mnist_sampler import MnistSampler
from models.wgan import WGAN_D, WGAN_G, weight_init
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--nf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

opt = parser.parse_args()

# Dataset loading and preparation
if opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                         transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                           ]))
    sampler = None
    shuff = True
elif opt.dataset == 'moving-mnist':

    dataset = MovingMnist('/Users/michaelgentnermac/Documents/ADL4CV/')

    sampler = MnistSampler(dataset)
    shuff = False

else:
    dataset = None
    sampler = None
    shuff = False

# cudnn stuff
cudnn.benchmark = True

# Set up models & init weights
device = torch.device("cuda:0" if opt.cuda else "cpu")
wgan_d = WGAN_D(1, int(opt.nf), 10, device)
wgan_g = WGAN_G(int(opt.nz), int(opt.nf), 1, device)
wgan_g.to(device)
wgan_d.to(device)
wgan_d.apply(weight_init)
wgan_g.apply(weight_init)

optim_d = optim.Adam(wgan_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optim_g = optim.Adam(wgan_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Set up tensorboard
writer = SummaryWriter('./tb/')

loader = DataLoader(dataset, opt.batchSize, sampler=sampler, shuffle=shuff, num_workers=4, pin_memory=True)

valid_noise = torch.randn(9, opt.nz, 1, 1, device=device)

for epoch in range(opt.niter):

    for i, batch in enumerate(loader):

        x_real = batch[0].cuda()
        # ToDO replace this with sampling on unit sphere
        z = torch.rand((opt.batchSize, opt.nz, 1, 1), device=device)


        x_generated = wgan_g(z)
        d_loss, grad_pen = wgan_d.wasserstein_loss(x_real, x_generated)

        d_loss.backward()
        optim_d.step()
        writer.add_scalar('D/loss', d_loss, i + len(loader) * epoch)
        writer.add_scalar('D/grad_pen', grad_pen, i + len(loader)*epoch)

        if i%3 == 0:

            wgan_g.zero_grad()
            z = torch.rand((opt.batchSize, opt.nz, 1, 1), device=device)
            x_generated = wgan_g(z)
            d_generated = wgan_d(x_generated)

            g_loss = wgan_g.wasserstein_loss(d_generated)
            g_loss.backward()
            optim_g.step()

            writer.add_scalar('G/loss', g_loss, i + len(loader) * epoch)

        if i%50 == 0:

            valid_x = wgan_g(valid_noise)
            writer.add_image('valid_image', torchvision.utils.make_grid(valid_x, nrow=3))

        print('[{}/{}][{}/{}]'.format(epoch, opt.niter, i, len(loader)))

