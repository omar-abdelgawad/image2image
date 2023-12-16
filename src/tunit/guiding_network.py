# pylint: skip-file

import torch
from torch import nn

from tunit.blocks import GuidingNetworkConvBlock

CHANNELS_MULTIPLIER = 64
K = 9


class GuidingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Sequential(
            GuidingNetworkConvBlock(
                in_channels=3,
                out_channels=CHANNELS_MULTIPLIER,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            GuidingNetworkConvBlock(
                in_channels=CHANNELS_MULTIPLIER,
                out_channels=CHANNELS_MULTIPLIER * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            GuidingNetworkConvBlock(
                in_channels=CHANNELS_MULTIPLIER * 2,
                out_channels=CHANNELS_MULTIPLIER * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            GuidingNetworkConvBlock(
                in_channels=CHANNELS_MULTIPLIER * 4,
                out_channels=CHANNELS_MULTIPLIER * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            GuidingNetworkConvBlock(
                in_channels=CHANNELS_MULTIPLIER * 4,
                out_channels=CHANNELS_MULTIPLIER * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            GuidingNetworkConvBlock(
                in_channels=CHANNELS_MULTIPLIER * 8,
                out_channels=CHANNELS_MULTIPLIER * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            GuidingNetworkConvBlock(
                in_channels=CHANNELS_MULTIPLIER * 8,
                out_channels=CHANNELS_MULTIPLIER * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            GuidingNetworkConvBlock(
                in_channels=CHANNELS_MULTIPLIER * 8,
                out_channels=CHANNELS_MULTIPLIER * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Linear(
                in_features=CHANNELS_MULTIPLIER * 8,
                out_features=128,
            ),
            nn.Linear(in_features=128, out_features=K),
        )

    def forward(self, x):
        return self.classifier(self.layer_1(x))
