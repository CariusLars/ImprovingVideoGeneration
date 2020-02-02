import torch
import torch.nn as nn
import torch.autograd as autograd


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, gp_lambda, device):

    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((torch.tensor([1], dtype=torch.float).to(device) - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = nn.Parameter(interpolates)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


