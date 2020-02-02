import torch.nn as nn


class PNet(nn.Module):
    def __init__(self, dim):
        super(PNet, self).__init__()

        self.dim = dim

        main = nn.Sequential(
            nn.Conv2d(1, self.dim, 5, stride=2, padding=2),
            nn.BatchNorm2d(self.dim),
            nn.LeakyReLU(),
            nn.Conv2d(self.dim, self.dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim),
            nn.LeakyReLU(),
            nn.Conv2d(self.dim, 2*self.dim, 5, stride=2, padding=2),
            nn.BatchNorm2d(2*self.dim),
            nn.LeakyReLU(),
            nn.Conv2d(2*self.dim, 2*self.dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(2*self.dim),
            nn.LeakyReLU(),
            nn.Conv2d(2*self.dim, 4*self.dim, 5, stride=2, padding=2),
            nn.BatchNorm2d(4*self.dim),
            nn.LeakyReLU(),
            nn.Conv2d(4*self.dim, 4*self.dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(4*self.dim),
            nn.LeakyReLU()
        )
        self.main = main
        self.output = nn.Sequential(nn.Linear(4*4*4*self.dim, self.dim),
                                    nn.BatchNorm1d(self.dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(self.dim, 16))

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        out = self.main(x)
        out = out.view(-1, 4*4*4*self.dim)
        out = self.output(out)
        return out.view(-1, 16)
