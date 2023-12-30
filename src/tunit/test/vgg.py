# pylint: skip-file
import torch
from torch import nn

from tunit.cfg import CHANNELS_MULTIPLIER


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity(),
        )

    def forward(self, x):
        return self.model(x)


class VGG(nn.Module):
    def __init__(
        self,
        in_channels=3,
        channels_multiplier=CHANNELS_MULTIPLIER,
        out_channels=CHANNELS_MULTIPLIER * 8,
    ):
        super().__init__()
        self.model = nn.Sequential(
            VGGBlock(
                in_channels=in_channels,
                out_channels=channels_multiplier,
                pool=True,
            ),
            VGGBlock(
                in_channels=channels_multiplier,
                out_channels=channels_multiplier * 2,
                pool=True,
            ),
            VGGBlock(
                in_channels=channels_multiplier * 2,
                out_channels=channels_multiplier * 4,
                pool=False,
            ),
            VGGBlock(
                in_channels=channels_multiplier * 4,
                out_channels=channels_multiplier * 4,
                pool=True,
            ),
            VGGBlock(
                in_channels=channels_multiplier * 4,
                out_channels=channels_multiplier * 8,
                pool=False,
            ),
            VGGBlock(
                in_channels=channels_multiplier * 8,
                out_channels=channels_multiplier * 8,
                pool=True,
            ),
            VGGBlock(
                in_channels=channels_multiplier * 8,
                out_channels=channels_multiplier * 8,
                pool=False,
            ),
            VGGBlock(
                in_channels=channels_multiplier * 8,
                out_channels=out_channels,
                pool=True,
            ),
        )

    def forward(self, x):
        return self.model(x)
