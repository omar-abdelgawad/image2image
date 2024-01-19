"""Discriminator model for the pix2pix GAN."""
import torch
from torch import nn


class CNNBlock(nn.Module):
    """Block that consists of a Conv2d layer, BatchNorm2d layer and LeakyReLU activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for conv kernel. Defaults to 2.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out: torch.Tensor = self.conv(x)
        return out


class Discriminator(nn.Module):
    """Discriminator model for the pix2pix GAN.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        features (tuple, optional): Sequence of feature channels. Defaults to (64, 128, 256, 512).
    """

    # channels are 3 because they are always going to be rgb
    def __init__(
        self, in_channels: int = 3, features=(64, 128, 256, 512)
    ) -> None:  # 256x256->30*30
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )
        layers: list[nn.Module] = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(
                    in_channels, feature, stride=1 if feature == features[-1] else 2
                )
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass for discriminator of Pix2Pix GAN.

        Args:
            x (torch.Tensor): Input image in first domain.
            y (torch.Tensor): Output image in second domain.

        Returns:
            torch.Tensor: Output tensor for probability for every patch.
        """
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


def test() -> None:
    """Test function for finding shape of output tensor."""
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x, y)
    print(preds)
    print(preds.shape)


if __name__ == "__main__":
    test()
