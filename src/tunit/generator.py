# pylint: skip-file

import torch
from torch import nn

from tunit.blocks import GenConvBlock
from tunit.blocks import GenResBlock

CHANNELS_MULTIPLIER = 64


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=CHANNELS_MULTIPLIER,
                kernel_size=7,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(num_features=CHANNELS_MULTIPLIER),
            GenConvBlock(
                in_channels=CHANNELS_MULTIPLIER,
                out_channels=CHANNELS_MULTIPLIER * 2,
                kernel_size=4,
                stride=1,
                ins=True,
            ),
            nn.InstanceNorm2d(num_features=CHANNELS_MULTIPLIER * 2),
            GenConvBlock(
                in_channels=CHANNELS_MULTIPLIER * 2,
                out_channels=CHANNELS_MULTIPLIER * 4,
                kernel_size=4,
                stride=2,
                ins=True,
            ),
            nn.InstanceNorm2d(num_features=CHANNELS_MULTIPLIER * 4),
            GenConvBlock(
                in_channels=CHANNELS_MULTIPLIER * 4,
                out_channels=CHANNELS_MULTIPLIER * 8,
                kernel_size=4,
                stride=2,
                ins=True,
            ),
            nn.InstanceNorm2d(num_features=CHANNELS_MULTIPLIER * 8),
        )

        self.layer_2 = nn.Sequential(
            GenResBlock(
                in_channels=CHANNELS_MULTIPLIER * 8,
                out_channels=CHANNELS_MULTIPLIER * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                ins=True,
            ),
            GenResBlock(
                in_channels=CHANNELS_MULTIPLIER * 8,
                out_channels=CHANNELS_MULTIPLIER * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                ins=True,
            ),
            GenResBlock(
                in_channels=CHANNELS_MULTIPLIER * 8,
                out_channels=CHANNELS_MULTIPLIER * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                ins=False,
            ),
            GenResBlock(
                in_channels=CHANNELS_MULTIPLIER * 8,
                out_channels=CHANNELS_MULTIPLIER * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                ins=False,
            ),
        )

        self.layer_3 = nn.Sequential(
            GenConvBlock(
                in_channels=CHANNELS_MULTIPLIER * 8,
                out_channels=CHANNELS_MULTIPLIER * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                ins=False,
                up=True,
            ),
            GenConvBlock(
                in_channels=CHANNELS_MULTIPLIER * 4,
                out_channels=CHANNELS_MULTIPLIER * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                ins=False,
                up=True,
            ),
            GenConvBlock(
                in_channels=CHANNELS_MULTIPLIER * 2,
                out_channels=CHANNELS_MULTIPLIER,
                kernel_size=3,
                stride=1,
                padding=1,
                ins=False,
                up=True,
            ),
            nn.Conv2d(
                in_channels=CHANNELS_MULTIPLIER,
                out_channels=3,
                kernel_size=7,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
