# pylint: skip-file

import torch
from torch import nn
from torch.nn import init

import math

from blocks import DiscResBlock


class Discriminator(nn.Module):
    def __init__(
        self, in_channels: int, channels_multiplier: int, out_channels: int
    ) -> None:
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels_multiplier,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            DiscResBlock(
                in_channels=channels_multiplier,
                out_channels=channels_multiplier,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            DiscResBlock(
                in_channels=channels_multiplier,
                out_channels=channels_multiplier * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            DiscResBlock(
                in_channels=channels_multiplier * 2,
                out_channels=channels_multiplier * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            DiscResBlock(
                in_channels=channels_multiplier * 2,
                out_channels=channels_multiplier * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            DiscResBlock(
                in_channels=channels_multiplier * 4,
                out_channels=channels_multiplier * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            DiscResBlock(
                in_channels=channels_multiplier * 4,
                out_channels=channels_multiplier * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            DiscResBlock(
                in_channels=channels_multiplier * 8,
                out_channels=channels_multiplier * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            DiscResBlock(
                in_channels=channels_multiplier * 8,
                out_channels=channels_multiplier * 16,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
            DiscResBlock(
                in_channels=channels_multiplier * 16,
                out_channels=channels_multiplier * 16,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=False,
            ),
            DiscResBlock(
                in_channels=channels_multiplier * 16,
                out_channels=channels_multiplier * 16,
                kernel_size=3,
                stride=1,
                padding=1,
                pool=True,
            ),
        )

        self.layer_2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=channels_multiplier * 16,
                out_channels=channels_multiplier * 16,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Conv2d(
            in_channels=channels_multiplier * 16,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.apply(weights_init("kaiming"))

    def forward(self, x, y):
        out = self.classifier(self.layer_2(self.layer_1(x)))
        feat = out
        out = out.view(out.size(0), -1)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]
        return out, feat

    def _initialize_weights(self, mode="fan_in"):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()


def weights_init(init_type="gaussian"):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            if init_type == "gaussian":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "default":
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


if __name__ == "__main__":
    D = Discriminator(in_channels=3, channels_multiplier=64, out_channels=10)
    x_in = torch.randn(32, 3, 128, 128)
    y_in = torch.randint(0, 10, size=(32,))
    out, feat = D(x_in, y_in)
    print(out.shape, feat.shape)
