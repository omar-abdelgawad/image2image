# pylint: skip-file

import torch
from torch import nn

from tunit.blocks import DiscResBlock

CHANNELS_MULTIPLIER = 64
K = 9


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=CHANNELS_MULTIPLIER,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            DiscResBlock(
                in_channels=CHANNELS_MULTIPLIER,
                out_channels=CHANNELS_MULTIPLIER,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            DiscResBlock(
                in_channels=CHANNELS_MULTIPLIER,
                out_channels=CHANNELS_MULTIPLIER * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            DiscResBlock(
                in_channels=CHANNELS_MULTIPLIER * 2,
                out_channels=CHANNELS_MULTIPLIER * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            DiscResBlock(
                in_channels=CHANNELS_MULTIPLIER * 2,
                out_channels=CHANNELS_MULTIPLIER * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            DiscResBlock(
                in_channels=CHANNELS_MULTIPLIER * 4,
                out_channels=CHANNELS_MULTIPLIER * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            DiscResBlock(
                in_channels=CHANNELS_MULTIPLIER * 4,
                out_channels=CHANNELS_MULTIPLIER * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            DiscResBlock(
                in_channels=CHANNELS_MULTIPLIER * 8,
                out_channels=CHANNELS_MULTIPLIER * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            DiscResBlock(
                in_channels=CHANNELS_MULTIPLIER * 8,
                out_channels=CHANNELS_MULTIPLIER * 16,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            DiscResBlock(
                in_channels=CHANNELS_MULTIPLIER * 16,
                out_channels=CHANNELS_MULTIPLIER * 16,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            DiscResBlock(
                in_channels=CHANNELS_MULTIPLIER * 16,
                out_channels=CHANNELS_MULTIPLIER * 16,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
        )

        self.layer_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=CHANNELS_MULTIPLIER * 16,
                out_channels=CHANNELS_MULTIPLIER * 16,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Conv2d(
            in_channels=CHANNELS_MULTIPLIER * 16,
            out_channels=K,
            kernel_size=1,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        return self.classifier(self.layer_2(self.layer_1(x)))
