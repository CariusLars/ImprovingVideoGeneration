import scipy.optimize as opt
import torch
import numpy as np


class BFGSProjector:

    def __init__(self, generator, latent_size, loss, method, device, tol):
        """
        :param generator: object ot generator
        :param latent_size: size of the latent space
        :param loss: object of loss function to use (e.g. torch BCE)
        :param method: str, which optimization method to apply (see scipy docs for more info)
        :param device: torch.device, where to place the operations
        :param tol: tolerance for termination of optimization
        """
        self.generator = generator
        self.loss = loss
        self.device = device
        self.method = method
        self.latent_size = latent_size
        self.tol = tol

    def project(self, img, x0=None):
        """
        This function backprojects image img into latent space using x0 as initialization
        :param img: target image as torch.tensor
        :param x0: initialization for optimization of size [latent_space]
        :return: optimization result (see scipy docs), res.x contains the latent vector of the backprojected image
        """
        if x0 is None:
            x0 = np.random.rand(self.latent_size)

        res = opt.minimize(_forward, x0, args=(img, self.generator, self.loss, self.device),
                           jac=_backward, method=self.method, tol=self.tol)
        return res


def _forward(x, y, wgan_G, loss, device):
    """
    This computes the forwardpass through the generator and returns the loss between generated and target image y
    :param x: input latent vector as numpy array
    :param y: target image as torch.tensor
    :param wgan_G: object of the generator
    :param loss: loss function to apply
    :param device: torch.device, where to place the operations
    :return: loss(wgan_G(x), y)
    """
    x = torch.from_numpy(x)
    x = x.to(device)
    x.requires_grad = True
    out = loss(wgan_G(x), y)
    return out.detach().numpy()


def _backward(x, y, wgan_G, loss, device):
    """
    This function computes the gradient of the loss between the generated image wgan_G(x) and the target image y w.r.t.
    the input x.
    :param x: input latent vector as numpy array
    :param y: target image as torch.tensor
    :param wgan_G: object of the generator
    :param loss: loss function to apply
    :param device: torch.device, where to place the operations
    :return: gradient of loss(wgan_G(x), y) w.r.t. x
    """
    wgan_G.zero_grad()
    x = torch.from_numpy(x)
    x = x.to(device)
    x.requires_grad = True
    out = loss(wgan_G(x), y)
    out.backward()
    return x.grad.detach().numpy()
