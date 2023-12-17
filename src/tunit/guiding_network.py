# pylint: skip-file

import torch
from torch import nn
import torch.nn.functional as F

from blocks import GuidingNetworkConvBlock


class GuidingNetwork(nn.Module):
    def __init__(
        self, in_channels: int, channels_multiplier: int, out_channels: int
    ) -> None:
        super().__init__()

        self.layer_1 = nn.Sequential(
            GuidingNetworkConvBlock(
                in_channels=in_channels,
                out_channels=channels_multiplier,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            GuidingNetworkConvBlock(
                in_channels=channels_multiplier,
                out_channels=channels_multiplier * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            GuidingNetworkConvBlock(
                in_channels=channels_multiplier * 2,
                out_channels=channels_multiplier * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            GuidingNetworkConvBlock(
                in_channels=channels_multiplier * 4,
                out_channels=channels_multiplier * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            GuidingNetworkConvBlock(
                in_channels=channels_multiplier * 4,
                out_channels=channels_multiplier * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            GuidingNetworkConvBlock(
                in_channels=channels_multiplier * 8,
                out_channels=channels_multiplier * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            GuidingNetworkConvBlock(
                in_channels=channels_multiplier * 8,
                out_channels=channels_multiplier * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            GuidingNetworkConvBlock(
                in_channels=channels_multiplier * 8,
                out_channels=channels_multiplier * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
        )

        self.features = self.layer_1

        self.classifier_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(
                in_features=channels_multiplier * 8,
                out_features=128,
            ),
        )

        self.cont = self.classifier_1

        self.classifier_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=channels_multiplier * 8, out_features=out_channels),
        )

        self.disc = self.classifier_2

        self._initialize_weights()

    def forward(self, x: torch.Tensor, style: bool = False) -> torch.Tensor:
        if style:
            return self.classifier_1(self.layer_1(x))
        else:
            x = self.layer_1(x)
            return self.classifier_1(x), self.classifier_2(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def moco(self, x):
        return self.cont(self.features(x))

    def iic(self, x):
        return self.disc(self.features(x))


if __name__ == "__main__":
    import torch

    C = GuidingNetwork(in_channels=3, channels_multiplier=64, out_channels=9)
    x_in = torch.randn(32, 3, 128, 128)
    sty = C.moco(x_in)
    print(sty.shape)
    cls = C.iic(x_in)
    print(sty.shape, cls.shape)
    x, y = C(x_in, style=False)
    print(x.shape, y.shape)
