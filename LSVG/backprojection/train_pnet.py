import os
import sys
sys.path.append('/root/video_interpolation/')
sys.path.append('/Users/michaelgentnermac/Documents/ADL4CV/video_interpolation/')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from LSVG.models.pnet import PNet
import argparse
from datetime import datetime
from LSVG.dataset.moving_mnist import MovingMnist
from LSVG.dataset.moving_mnist_sampler import MnistSampler
from LSVG.models.wgan import Generator


# Argparse
help = 'This script trains a WGAN model according to the specified optional arguments.'
parser = argparse.ArgumentParser(description=help)

parser.add_argument('--dataset', type=str, default='/Users/michaelgentnermac/Documents/ADL4CV/',
                    help='path to the dataset folder')
parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs to train')
parser.add_argument('--log_dir', type=str, default='./tb/', help='directory where the logs will be placed')
parser.add_argument('--batchSize', type=int, default=256, help='batch size')
parser.add_argument('--lrate', type=float, default=1e-4, help='learning rate for critic')
parser.add_argument('--gpath', type=str, default='/Users/michaelgentnermac/Documents/ADL4CV/model/g.pt')
parser.add_argument('--checkpoint_dir', type=str, default=None)
parser.add_argument('--dim', type=int, default=64, help='scaling factor for filters in Pnet')
config = parser.parse_args()

timestamp = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")


# Save training configuration
if not os.path.exists(config.log_dir + '/' + timestamp):
    os.makedirs(config.log_dir + '/' + timestamp)

with open(config.log_dir + '/' + timestamp + '/train_config.txt', 'w') as f:
    print('------------- training configuration -------------', file=f)
    for k, v in vars(config).items():
        print('{}: {}'.format(k, v), file=f)

    print('Saved training configuration to {}'.format(f))

if not os.path.exists(config.log_dir + '/' + timestamp + '/tb'):
    os.makedirs(config.log_dir + '/' + timestamp + '/tb')
if not os.path.exists(config.log_dir + '/' + timestamp + '/models'):
    os.makedirs(config.log_dir + '/' + timestamp + '/models')


# Cuda and logging
torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
writer = SummaryWriter(config.log_dir + '/' + timestamp + '/tb')


# Load models and dataset
dataset = MovingMnist(config.dataset, downsampled=True)
if config.checkpoint_dir is None:
    pnet = PNet(config.dim).to(device)
else:
    pnet = torch.load(os.path.join(config.checkpoint_dir, 'pnet.pt'), map_location=device)
netG = torch.load(config.gpath, map_location=device)


# Disable gradient computation for generator
for parameter in netG.parameters():
    parameter.requires_grad = False


# Setup training
criterion = nn.BCELoss()
optimizer = optim.Adam(pnet.parameters(), lr=config.lrate)
dataloader = DataLoader(dataset, batch_size=config.batchSize, drop_last=False, sampler=MnistSampler(dataset))


# Get two validation images
static_valid_imgs = dataset.get_valid_batch(2).to(device)


# Get two images from the training dataset
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

        if i % 10 == 0:
            print("[{}/{}][{}/{}]".format(epoch, config.num_epochs, i, len(dataloader)))
            writer.add_scalar('loss/MSE', loss, global_step=i+len(dataloader)*epoch)

        # Validation run on images of training set and validation set
        if i % 200 == 0:

            # Calculate validation error
            valid_batch = dataset.get_valid_batch(config.batchSize).to(device)
            valid_latent = pnet(valid_batch)
            valid_imgs = netG(valid_latent)
            valid_loss = criterion(valid_imgs, valid_batch)
            writer.add_scalar('valid_loss/MSE', valid_loss, global_step=i+len(dataloader)*epoch)

            # Create validation images and train images
            static_valid_out = netG(pnet(static_valid_imgs))
            static_train_out = netG(pnet(static_train_imgs))
            grid_valid = torch.cat([static_valid_imgs.unsqueeze(1), static_valid_out], dim=0)
            grid_train = torch.cat([static_train_imgs, static_train_out], dim=0)
            writer.add_image('img/valid', torchvision.utils.make_grid(grid_valid, nrow=2),
                             global_step=i+len(dataloader)*epoch)
            writer.add_image('img/traind', torchvision.utils.make_grid(grid_train, nrow=2),
                             global_step=i+len(dataloader)*epoch)


torch.save(pnet, config.log_dir + timestamp + '/models' + '/pnet.pt')
