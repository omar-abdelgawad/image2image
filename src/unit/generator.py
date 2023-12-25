from torch import nn


class VAEGen(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
