import torch


class CriticProjector:

    def __init__(self, generator, critic, device, learning_rate=1e-4, verbose=False):

        self.netG = generator
        self.netC = critic
        self.device = device
        self.optim = torch.optim.SGD
        self.verbose = verbose
        self.lr = learning_rate

        self.netG = self.netG.double().to(device)
        self.netC = self.netC.double().to(device)

        for parameter in self.netG.parameters():
            parameter.requires_grad = False

        for parameter in self.netC.parameters():
            parameter.requires_grad = False

    def project(self, x0, steps):

        x0 = torch.tensor(x0).to(self.device)
        x0.requires_grad = True

        optim = self.optim([x0], lr=self.lr)
        mone = (-1)*torch.tensor([1], dtype=torch.float)
        mone = mone.double().to(self.device)

        for i in range(steps):

            if x0.grad is not None:
                x0.grad.zero_()

            loss = self.netC(self.netG(x0))

            if self.verbose:
                print("[{}/{}]: {}".format(i, steps, loss))

            loss.backward(mone)
            optim.step()

        return x0.detach()