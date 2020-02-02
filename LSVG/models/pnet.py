import torch.nn as nn

DIM = 64


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            #nn.InstanceNorm2d(DIM, affine=True),
            nn.BatchNorm2d(DIM),
            #nn.ReLU(True),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(DIM),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.BatchNorm2d(2*DIM),
            nn.LeakyReLU(),
            nn.Conv2d(2*DIM, 2*DIM, 3, stride=1, padding=1),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            #nn.InstanceNorm2d(DIM*2, affine=True),
            nn.BatchNorm2d(2*DIM),
            #nn.ReLU(True),
            nn.LeakyReLU(),
            #nn.Conv2d(2*DIM, 2*DIM, 3, stride=1, padding=1),
            #nn.LeakyReLU(),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.BatchNorm2d(4*DIM),
            nn.LeakyReLU(),
            nn.Conv2d(4*DIM, 4*DIM, 3, stride=1, padding=1),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            #nn.InstanceNorm2d(4*DIM, affine=True),
            nn.BatchNorm2d(4*DIM),
            nn.LeakyReLU()
            #nn.Conv2d(4*DIM, 4*DIM, 3, stride=1, padding=1),
            #nn.LeakyReLU()
            #nn.ReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Sequential(nn.Linear(4*4*4*DIM, DIM),
                                    nn.BatchNorm1d(DIM),
                                    nn.LeakyReLU(),
                                    nn.Linear(DIM, 16))

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1, 16)
