import os, sys
#sys.path.append(os.getcwd())
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from models.wgan import Generator, Critic, BGCritic, BGGenerator
from functions.functions import calc_gradient_penalty
import argparse
from datetime import datetime


# Argparse
help = 'This script trains a WGAN model according to the specified optional arguments.'
parser = argparse.ArgumentParser(description=help)

parser.add_argument('--dataset', type=str, default='', help='path to the dataset folder')
parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train')
parser.add_argument('--random_seed', type=int, default=1, help='number of epochs to train')
parser.add_argument('--log_dir', type=str, default='./tb/', help='number of epochs to train')
parser.add_argument('--c_scaling', type=int, default=64, help='scaling factor for critic filters')
parser.add_argument('--g_scaling', type=int, default=64, help='scaling factor for generator filters')
parser.add_argument('--c_iters', type=int, default=5, help='number of iterations to train the critic '
                                                           'before the generator is trained')
parser.add_argument('--g_iters', type=int, default=1, help='generator iterations after critic is trained')
parser.add_argument('--lam', type=float, default=10, help='factor for gradient penalty')
parser.add_argument('--batchSize', type=int, default=256, help='batch size')
parser.add_argument('--latent_size', type=int, default=128, help='size of the latent-vector')
parser.add_argument('--Clrate', type=float, default=1e-4, help='learning rate for critic')
parser.add_argument('--Glrate', type=float, default=1e-4, help='learning rate for generator')
config = parser.parse_args()
timestamp = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

# Save training configuration
if not os.path.exists(config.log_dir + '/' + timestamp):
    os.makedirs(config.log_dir + '/' + timestamp)

with open(config.log_dir + '/' + timestamp + '/train_config.txt', 'w') as f:
    print('------------- training configuration -------------', file=f)
    for k, v in vars(config).items():
        print(('{}: {}').format(k, v), file=f)

    print('Saved training configuration to {}'.format(f))


torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
writer = SummaryWriter('./tb/' + timestamp + '/')

dataset = dset.MNIST(root='./mnist/', download=True,
                        transform=transforms.Compose([
                            transforms.Resize(32),
                            transforms.ToTensor()
                           ]))

#netG = Generator(config.g_scaling, config.latent_size)
#netD = Critic(config.c_scaling)
netG = BGGenerator(config.g_scaling, config.latent_size)
netD = BGCritic(config.c_scaling)
netG = netG.to(device)
netD = netD.to(device)

optimizerD = optim.Adam(netD.parameters(), lr=config.Clrate, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=config.Glrate, betas=(0.5, 0.9))

# Tensors for loss function decomposition
one = torch.tensor([1], dtype=torch.float)
one = one.squeeze().to(device)
mone = one * -1
mone = mone.squeeze().to(device)
valid_noise = torch.randn((9, config.latent_size), device=device)

for epoch in range(config.num_epochs):

    loader = DataLoader(dataset, shuffle=True, batch_size=config.batchSize, drop_last=True)
    print("Starting epoch with {} iterations".format(len(loader)))
    iterator = iter(loader)

    for iteration in range(int(len(loader)/5)):

        # Training the critic network
        print("[{}/{}][{}/{}]".format(epoch, config.num_epochs, iteration, int(len(loader)/5)))
        for p in netD.parameters():
            p.requires_grad = True

        for iter_d in range(config.c_iters):
            real_data = next(iterator)[0]
            real_data = real_data.to(device)
            real_data_v = nn.Parameter(real_data)

            netD.zero_grad()

            # train with real
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)

            # train with fake
            fake = netG.generate_images(config.batchSize, device)

            D_fake = netD(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data, config.batchSize, config.lam, device)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

            if iter_d == 0:

                writer.add_scalar('D/fake', D_fake, iteration + int(len(loader)/5)*epoch)
                writer.add_scalar('D/GP', gradient_penalty, iteration + int(len(loader)/5)*epoch)
                writer.add_scalar('D/real', D_real, iteration+int(len(loader)/5)*epoch)
                writer.add_scalar('D/cost', D_cost, iteration + int(len(loader)/5)*epoch)
                writer.add_scalar('D/wasserstein', Wasserstein_D, iteration + int(len(loader)/5)*epoch)

        # Train generator network
        for i in range(1):
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            fake = netG.generate_images(config.batchSize, device)
            G = netD(fake)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            optimizerG.step()

            writer.add_scalar('G/cost', G_cost, iteration + int(len(loader)/5)*epoch)

            if iteration%20 == 0:

                valid_x = netG(valid_noise)
                writer.add_image('valid_image', torchvision.utils.make_grid(valid_x, nrow=3), global_step=iteration)


