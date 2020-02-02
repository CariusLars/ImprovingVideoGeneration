import scipy.optimize as opt
import torch
import torch.nn as nn
import numpy as np


class BFGSProjector:

    def __init__(self, generator, latent_size, loss, method, device, tol):

        self.generator = generator
        self.loss = loss
        self.device = device
        self.method = method
        self.latent_size = latent_size
        self.tol = tol

    def project(self, img, x0=None):

        if x0 is None:
            x0 = np.random.rand(self.latent_size)

        res = opt.minimize(_forward, x0, args=(img, self.generator, self.loss, self.device), jac=_backward, method=self.method, tol=self.tol)
        return res


def _forward(x, y, wgan_G, loss, device):

    x = torch.from_numpy(x)
    x = x.to(device)
    x.requires_grad = True
    out = loss(wgan_G(x), y)
    return out.detach().numpy()


def _backward(x, y, wgan_G, loss, device):

    wgan_G.zero_grad()
    x = torch.from_numpy(x)
    x = x.to(device)
    x.requires_grad = True
    out = loss(wgan_G(x), y)
    out.backward()
    return x.grad.detach().numpy()