# pylint: skip-file

import torch
from torch import nn

from tunit.utils import AdaptiveInstanceNorm2d
from tunit.utils import FRN


class GenConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        ins=True,
        up=False,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.InstanceNorm2d(num_features=in_channels)
            if ins
            else AdaptiveInstanceNorm2d(num_features=in_channels),
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            if up
            else None,
        )

    def forward(self, x):
        return self.model(x)


class GuidingNetworkConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        pool=True,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=in_channels),
            nn.MaxPool2d(kernel_size=2) if pool else None,
        )

    def forward(self, x):
        return self.model(x)


class DiscConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        pool=True,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            FRN(num_features=out_channels),
            nn.AvgPool2d(kernel_size=2) if pool else None,
        )

    def forward(self, x):
        return self.model(x)


class GenResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        ins=True,
    ):
        super().__init__()

        self.model = nn.Sequential(
            GenConvBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                ins,
            ),
            GenConvBlock(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                ins,
            ),
        )

    def forward(self, x):
        x_org = x
        residual = self.model(x)
        out = x_org + 0.1 * residual
        return out


class DiscResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        pool=True,
    ):
        super().__init__()

        self.model = nn.Sequential(
            DiscConvBlock(
                in_channels, out_channels, kernel_size, stride, padding, pool
            ),
            DiscConvBlock(
                out_channels, out_channels, kernel_size, stride, padding, pool
            ),
        )

    def forward(self, x):
        x_org = x
        residual = self.model(x)
        out = x_org + 0.1 * residual
        return out
