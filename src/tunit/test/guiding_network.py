# pylint: skip-file
import torch
from torch import nn

from vgg import VGG
from tunit.cfg import CHANNELS_MULTIPLIER, K


class GuidingNetwork(nn.Module):
    def __init__(
        self, in_channels=3, channels_multiplier=CHANNELS_MULTIPLIER, out_channels=K
    ):
        super().__init__()

        self.model = nn.Sequential(
            VGG(
                in_channels=in_channels,
                channels_multiplier=channels_multiplier,
                out_channels=channels_multiplier * 8,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )

        self.disc = nn.Linear(
            in_features=channels_multiplier * 8,
            out_features=out_channels,
        )

        self.gen = nn.Linear(
            in_features=channels_multiplier * 8,
            out_features=128,
        )

        self._initialize_weights()

    def forward(self, x, sty=False):
        if sty:
            return self.gen(self.model(x))
        return self.gen(self.model(x)), self.disc(self.model(x))

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
        return self.gen(self.model(x))

    def iic(self, x):
        return self.disc(self.model(x))


if __name__ == "__main__":
    x = torch.randn(1, 3, 128, 128)
    model = GuidingNetwork()
    x, y = model(x)
    print(x.shape, y.shape)
