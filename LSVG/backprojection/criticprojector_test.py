import torch.nn as nn
import torch
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from models.wgan import Generator, Discriminator
from backprojection.criticprojector import CriticProjector

#latent_dim = 16
#DIM = 128
latent_path = '/Users/michaelgentnermac/Documents/ADL4CV/final_model/latent_video_linear.pkl'
model_path = '/Users/michaelgentnermac/Documents/ADL4CV/final_model'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

"""
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
"""

netG = torch.load(os.path.join(model_path, 'g.pt'), map_location=device)
netD = torch.load(os.path.join(model_path, 'd.pt'), map_location=device)
#netG.double()
#netD.double()

#for parameter in netG.parameters():
#    parameter.requires_grad = False

#for parameter in netD.parameters():
#    parameter.requires_grad = False


x0 = pickle.load(open(latent_path, 'rb'))
gt = torch.tensor(x0[0]).unsqueeze(0).to(device)
old = torch.tensor(x0[9]).unsqueeze(0).to(device)
x0 = torch.tensor(x0[9]).unsqueeze(0).to(device)

#x0.requires_grad = True
#optim = torch.optim.SGD([x0], lr=1e-4)
#one = torch.tensor([1], dtype=torch.float)
#one = torch.ones((BATCH_SIZE, 1, 1, 1), dtype=torch.float)
#mone = one * -1
#one = one.double()
#mone = mone.double()

proj = CriticProjector(netG, netD, device)

x0 = proj.project(x0, 100)
#for i in range(200):

#    if x0.grad is not None:
#        x0.grad.zero_()

#    loss = netD(netG(x0))
#    print(loss)

#    loss.backward(mone)
#    optim.step()

print(old-x0)
fig = plt.figure(figsize=(8, 8))
fig.add_subplot(1, 3, 1)
plt.imshow(netG(x0).detach().squeeze().numpy())
fig.add_subplot(1, 3, 2)
plt.imshow(netG(old).detach().squeeze().numpy())
fig.add_subplot(1, 3, 3)
plt.imshow(netG(gt).detach().squeeze().numpy())
plt.show()

