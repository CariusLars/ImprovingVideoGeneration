import os, sys
sys.path.append('/root/video_interpolation/')
sys.path.append('/Users/michaelgentnermac/Documents/ADL4CV/video_interpolation/')
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from models.pnet import PNet
from functions.functions import calc_gradient_penalty
import argparse
from datetime import datetime
from dataset.moving_mnist import MovingMnist
from dataset.moving_mnist_sampler import MnistSampler


# Argparse
help = 'This script trains a WGAN model according to the specified optional arguments.'
parser = argparse.ArgumentParser(description=help)

parser.add_argument('--dataset', type=str, default='/Users/michaelgentnermac/Documents/ADL4CV/', help='path to the dataset folder')
parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train')
parser.add_argument('--random_seed', type=int, default=1, help='number of epochs to train')
parser.add_argument('--log_dir', type=str, default='./tb/', help='number of epochs to train')
parser.add_argument('--batchSize', type=int, default=256, help='batch size')
parser.add_argument('--lrate', type=float, default=1e-4, help='learning rate for critic')
parser.add_argument('--gpath', type=str, default='/Users/michaelgentnermac/Documents/ADL4CV/model/g.pt')
parser.add_argument('--checkpoint_dir', type=str, default=None)

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

if not os.path.exists(config.log_dir + '/' + timestamp + '/tb'):
    os.makedirs(config.log_dir + '/' + timestamp + '/tb')
writer = SummaryWriter(config.log_dir + '/' + timestamp + '/tb')
if not os.path.exists(config.log_dir + '/' + timestamp + '/models'):
    os.makedirs(config.log_dir + '/' + timestamp + '/models')

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

latent_dim = 16
DIM = 128


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


dataset = MovingMnist(config.dataset, downsampled=True)

if config.checkpoint_dir is None:
    pnet = PNet()
else:
    pnet = torch.load(os.path.join(config.checkpoint_dir, 'pnet.pt'), map_location=device)

netG = torch.load(config.gpath, map_location=device)
pnet = pnet.to(device)

for parameter in netG.parameters():
    parameter.requires_grad = False

optimizer = optim.Adam(pnet.parameters(), lr=config.lrate)

dataloader = DataLoader(dataset, batch_size=config.batchSize, drop_last=False, sampler=MnistSampler(dataset))

#criterion = nn.MSELoss()
criterion = nn.BCELoss()

static_valid_imgs = dataset.get_valid_batch(2)
static_valid_imgs = static_valid_imgs.to(device)

static_train_imgs = next(iter(dataloader))['img'][0:2, :, :, :]
static_train_imgs = static_train_imgs.to(device)

for epoch in range(config.num_epochs):

    for i, batch in enumerate(dataloader):

        pnet.zero_grad()
        imgs = batch['img']
        imgs = imgs.to(device)
        latent = pnet(imgs)
        generated_imgs = netG(latent)
        loss = criterion(generated_imgs, imgs)
        loss.backward()
        optimizer.step()
        if i%10 == 0:

            print("[{}/{}][{}/{}]".format(epoch, config.num_epochs, i, len(dataloader)))
            writer.add_scalar('loss/MSE', loss, global_step=i+len(dataloader)*epoch)

        if i%200 == 0:

            valid_batch = dataset.get_valid_batch(config.batchSize)
            valid_batch = valid_batch.to(device)

            valid_latent = pnet(valid_batch)
            valid_imgs = netG(valid_latent)
            valid_loss = criterion(valid_imgs, valid_batch)

            writer.add_scalar('valid_loss/MSE', valid_loss, global_step=i+len(dataloader)*epoch)

            static_valid_out = netG(pnet(static_valid_imgs))
            static_train_out = netG(pnet(static_train_imgs))

            grid_valid = torch.cat([static_valid_imgs.unsqueeze(1), static_valid_out], dim=0)
            grid_train = torch.cat([static_train_imgs, static_train_out], dim=0)

            writer.add_image('img/valid', torchvision.utils.make_grid(grid_valid, nrow=2), global_step=i+len(dataloader)*epoch)
            writer.add_image('img/traind', torchvision.utils.make_grid(grid_train, nrow=2), global_step=i+len(dataloader)*epoch)


torch.save(pnet, config.log_dir + timestamp + '/models' + '/pnet.pt')

    # validation

"""
        writer.add_scalar('D/fake', D_fake, iteration + int(len(loader)/5)*epoch)
        writer.add_scalar('D/GP', gradient_penalty, iteration + int(len(loader)/5)*epoch)
        writer.add_scalar('D/real', D_real, iteration+int(len(loader)/5)*epoch)
        writer.add_scalar('D/cost', D_cost, iteration + int(len(loader)/5)*epoch)
        writer.add_scalar('D/wasserstein', Wasserstein_D, iteration + int(len(loader)/5)*epoch)


        writer.add_scalar('G/cost', G_cost, iteration + int(len(loader)/5)*epoch)

        if iteration%20 == 0:

            valid_x = netG(valid_noise)
            writer.add_image('valid_image', torchvision.utils.make_grid(valid_x, nrow=3), global_step=iteration)
"""

