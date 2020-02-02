import os
import sys
sys.path.append(os.getcwd())
from datetime import datetime
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import argparse
from tqdm import tqdm
from LSVG.dataset.moving_mnist import MovingMnist
from LSVG.dataset.moving_mnist_sampler import MnistSampler
from LSVG.models.wgan import Generator, Discriminator


# Parsing arguments
help_desc = 'This script trains a WGAN model according to the specified optional arguments.'
parser = argparse.ArgumentParser(description=help_desc)

parser.add_argument('--dataset', type=str, default='', help='path to the dataset folder')
parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train')
parser.add_argument('--random_seed', type=int, default=1, help='number of epochs to train')
parser.add_argument('--log_dir', type=str, default='./tb/', help='number of epochs to train')
parser.add_argument('--model_dir', type=str, default=None, help='path to d and g model')
parser.add_argument('--dim', type=int, default=128, help='scaler for filters in D and G')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--c_iters', type=int, default=5, help='critic updates before one generator update')
parser.add_argument('--lambd', type=float, default=10, help='gradient penalty term')
parser.add_argument('--latent_dim', type=int, default=16, help='dimensionality of latent space')
config = parser.parse_args()

DIM = config.dim
BATCH_SIZE = config.batch_size
CRITIC_ITERS = config.c_iters
LAMBDA = config.lambd
latent_dim = config.latent_dim


# Function for calculating gradient penalty
def calc_gradient_penalty(discriminator, real, fake):

    alpha = torch.rand(BATCH_SIZE, 1, 1, 1)
    alpha = alpha.expand(real.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real + ((torch.tensor([1], dtype=torch.float).to(device) - alpha) * fake)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = nn.Parameter(interpolates)

    disc_interpolates = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    grad_pen = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return grad_pen


# Save training configuration
timestamp = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
if not os.path.exists(config.log_dir + '/' + timestamp):
    os.makedirs(config.log_dir + '/' + timestamp)

with open(config.log_dir + '/' + timestamp + '/train_config.txt', 'w') as f:
    print('------------- training configuration -------------', file=f)
    for k, v in vars(config).items():
        print('{}: {}'.format(k, v), file=f)

    print('Saved training configuration to {}'.format(f))


# Setting manual seeds
torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)


# Cuda and logging
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
writer = SummaryWriter(config.log_dir + '/' + timestamp + '/tb')


# Creating folders for models and tensorboard
if not os.path.exists(config.log_dir + '/' + timestamp + '/tb'):
    os.makedirs(config.log_dir + '/' + timestamp + '/tb')
if not os.path.exists(config.log_dir + '/' + timestamp + '/models'):
    os.makedirs(config.log_dir + '/' + timestamp + '/models')


# Load saved models to continue training
if config.model_dir is not None:

    netG = torch.load(os.path.join(config.model_dir, 'g.pt'), map_location=device)
    netD = torch.load(os.path.join(config.model_dir, 'd.pt'), map_location=device)
else:

    netG = Generator(latent_dim, DIM).to(device)
    netD = Discriminator(latent_dim, DIM).to(device)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))


# Load dataset
dataset = MovingMnist(config.dataset)


# save initial model structure to file
with open(config.log_dir + '/' + timestamp + '/model_structure.txt', 'w') as f:
    print('--------------------------------------------------', file=f)
    print('Sequences in Dataset: ', len(dataset), ', Batch size: ', BATCH_SIZE, file=f)
    print('--------------------------------------------------', file=f)
    print('Generator structure: ', file=f)
    print(netG, file=f)
    print('--------------------------------------------------', file=f)
    print('Discriminator structure: ', file=f)
    print(netD, file=f)
    print('--------------------------------------------------', file=f)
    print(' Initial model strutures saved to {}'.format(f))


# Variables for computing the gradients of the total loss
one = torch.tensor([1], dtype=torch.float)
one = one.squeeze().to(device)
mone = one * -1
mone = mone.squeeze().to(device)

# Static latent vectors for validation
valid_noise = torch.randn((9, latent_dim), device=device)


for epoch in range(config.num_epochs):

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True, sampler=MnistSampler(dataset))
    print("Starting epoch {} of {} with {} iterations".format(epoch+1, config.num_epochs, len(loader)))
    iterator = iter(loader)

    for iteration in tqdm(range(int(len(loader)/CRITIC_ITERS))):

        for p in netD.parameters():
            p.requires_grad = True

        # Train critic
        for iter_d in range(CRITIC_ITERS):
            real_data = next(iterator)['img']
            real_data = real_data.to(device)
            real_data_v = nn.Parameter(real_data)

            netD.zero_grad()

            # train with real
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)
            if iter_d == 0:
                writer.add_scalar('D/real', D_real, iteration+int(len(loader)/CRITIC_ITERS)*epoch)

            # train with fake
            noise = torch.randn(BATCH_SIZE, latent_dim)
            noise = noise.to(device)
            noisev = nn.Parameter(noise)
            fake = nn.Parameter(netG(noisev))
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)
            if iter_d == 0:
                writer.add_scalar('D/fake', D_fake, iteration + int(len(loader)/CRITIC_ITERS)*epoch)

            # Apply gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            gradient_penalty.backward()
            if iter_d == 0:
                writer.add_scalar('D/GP', gradient_penalty, iteration + int(len(loader)/CRITIC_ITERS)*epoch)
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()
            if iter_d == 0:
                writer.add_scalar('D/cost', D_cost, iteration + int(len(loader)/5)*epoch)
                writer.add_scalar('D/wasserstein', Wasserstein_D, iteration + int(len(loader)/5)*epoch)

        # Setting parameters to false to avoid computation
        for p in netD.parameters():
            p.requires_grad = False

        netG.zero_grad()

        noise = torch.randn(BATCH_SIZE, latent_dim)
        noise = noise.to(device)
        noisev = nn.Parameter(noise)
        fake = netG(noisev)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()
        writer.add_scalar('G/cost', G_cost, iteration + int(len(loader)/5)*epoch)
        if iteration % 20 == 0:
            valid_x = netG(valid_noise)
            writer.add_image('valid_image', torchvision.utils.make_grid(valid_x, nrow=3), global_step=iteration)

print("Saving models")
torch.save(netG, config.log_dir + '/' + timestamp + '/models' + '/g.pt')
torch.save(netD, config.log_dir + '/' + timestamp + '/models' + '/d.pt')
print("Saving models - done")
