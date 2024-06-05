"""Discriminator model for the pix2pix GAN."""

import torch
from torch import nn

from img2img.cfg.pix2pix import ActivationType, NormalizationType, PaddingType
from img2img.nn.blocks import ConvBlock


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

        self.initial = ConvBlock(
            in_channels=in_channels * 2,
            out_channels=features[0],
            kernel_size=4,
            stride=2,
            padding=1,
            normalization_type=NormalizationType.NONE,
            padding_type=PaddingType.REFLECT,
            activation_type=ActivationType.LEAKY_RELU,
            bias=True,
        )
        layers: list[nn.Module] = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=feature,
                    kernel_size=4,
                    stride=1 if feature == features[-1] else 2,
                    padding=1,
                    normalization_type=NormalizationType.BATCH,
                    padding_type=PaddingType.REFLECT,
                    activation_type=ActivationType.LEAKY_RELU,
                    bias=False,
                ),
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
