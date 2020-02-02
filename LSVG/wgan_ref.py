import os, sys
sys.path.append(os.getcwd())

import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import torchvision.transforms as transforms
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import argparse
from dataset.moving_mnist import MovingMnist
from dataset.moving_mnist_sampler import MnistSampler
from tqdm import tqdm


help = 'This script trains a WGAN model according to the specified optional arguments.'
parser = argparse.ArgumentParser(description=help)

parser.add_argument('--dataset', type=str, default='', help='path to the dataset folder')
parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train')
parser.add_argument('--random_seed', type=int, default=1, help='number of epochs to train')
parser.add_argument('--log_dir', type=str, default='./tb/', help='number of epochs to train')
parser.add_argument('--model_dir', type=str, default=None, help='path to d and g model')

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

torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
    print("Using GPU to train model")

DIM = 128 # Model dimensionality
BATCH_SIZE = 256 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
latent_dim = 16#256



if not os.path.exists(config.log_dir + '/' + timestamp + '/tb'):
    os.makedirs(config.log_dir + '/' + timestamp + '/tb')
writer = SummaryWriter(config.log_dir + '/' + timestamp + '/tb')
if not os.path.exists(config.log_dir + '/' + timestamp + '/models'):
    os.makedirs(config.log_dir + '/' + timestamp + '/models')
device = torch.device("cuda:0" if use_cuda else "cpu")

# ==================Definition Start======================


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


def generate_image(frame, netG):
    noise = torch.randn(BATCH_SIZE, latent_dim)
    if use_cuda:
        noise = noise.cuda()
    noisev = nn.Parameter(noise)
    samples = netG(noisev)
    samples = samples.view(BATCH_SIZE, 28, 28)
    # print samples.size()

    samples = samples.cpu().data.numpy()


dataset = MovingMnist(config.dataset)

def calc_gradient_penalty(netD, real_data, fake_data):

    alpha = torch.rand(BATCH_SIZE, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((torch.tensor([1], dtype=torch.float).to(device) - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = nn.Parameter(interpolates)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================


if config.model_dir is not None:

    netG = torch.load(os.path.join(config.model_dir, 'g.pt'), map_location=device)
    netD = torch.load(os.path.join(config.model_dir, 'd.pt'), map_location=device)
else:

    netG = Generator()
    netD = Discriminator()


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


if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.tensor([1], dtype=torch.float)
one = one.squeeze()
#one = torch.ones((BATCH_SIZE, 1, 1, 1), dtype=torch.float)
mone = one * -1
mone = mone.squeeze()
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

valid_noise = torch.randn((9, latent_dim), device=device)
#matplotlib.use('TKAgg',warn=False, force=True)
for epoch in range(config.num_epochs):

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True, sampler=MnistSampler(dataset))
    print("Starting epoch {} of {} with {} iterations".format(epoch+1, config.num_epochs, len(loader)))
    iterator = iter(loader)

    for iteration in tqdm(range(int(len(loader)/5))):
        start_time = time.time()
         ############################
         # (1) Update D network
         ###########################
        #print("[{}/{}][{}/{}]".format(epoch, 25, iteration, int(len(loader)/5)))
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in range(CRITIC_ITERS):
            real_data = next(iterator)['img']
            if use_cuda:
                real_data = real_data.cuda()
            real_data_v = nn.Parameter(real_data)

            netD.zero_grad()
            #print(real_data.shape)
            #print(real_data[0, 0, :, :])
            #plt.imshow(real_data[0, 0, :, :])
            #plt.show()
            # train with real
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            # print D_real
            D_real.backward(mone)
            if iter_d == 0:
                writer.add_scalar('D/real', D_real, iteration+int(len(loader)/5)*epoch)
            # train with fake
            noise = torch.randn(BATCH_SIZE, latent_dim)
            if use_cuda:
                noise = noise.cuda()
            noisev = nn.Parameter(noise)
            fake = nn.Parameter(netG(noisev))
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)
            if iter_d == 0:
                writer.add_scalar('D/fake', D_fake, iteration + int(len(loader)/5)*epoch)
            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            gradient_penalty.backward()
            if iter_d == 0:
                writer.add_scalar('D/GP', gradient_penalty, iteration + int(len(loader)/5)*epoch)
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()
            if iter_d == 0:
                writer.add_scalar('D/cost', D_cost, iteration + int(len(loader)/5)*epoch)
                writer.add_scalar('D/wasserstein', Wasserstein_D, iteration + int(len(loader)/5)*epoch)
            ############################
        # (2) Update G network
        ###########################
        for i in range(1):
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            noise = torch.randn(BATCH_SIZE, latent_dim)
            if use_cuda:
                noise = noise.cuda()
            noisev = nn.Parameter(noise)
            fake = netG(noisev)
            G = netD(fake)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            optimizerG.step()
            writer.add_scalar('G/cost', G_cost, iteration + int(len(loader)/5)*epoch)

            if iteration%20 == 0:

                #print(valid_noise.shape)
                valid_x = netG(valid_noise)
                writer.add_image('valid_image', torchvision.utils.make_grid(valid_x, nrow=3), global_step=iteration)

print("Saving models")
torch.save(netG,config.log_dir + '/' + timestamp + '/models' + '/g.pt')
torch.save(netD, config.log_dir + '/' + timestamp + '/models' + '/d.pt')
print("Saving models - done")