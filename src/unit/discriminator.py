from torch import nn


class MsImageDis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
