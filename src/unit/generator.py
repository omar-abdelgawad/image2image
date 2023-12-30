"""Generator for the GAN."""
from typing import Tuple

import torch
from torch import nn

from unit.blocks import ResBlocks, ConvBlock


class Generator(nn.Module):
    """Generator model for the unit GAN.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        out_channels (int, optional): Number of output channels. Defaults to 3.
        layer_multiplier (int, optional): Number of channels multiplier. Defaults to 64.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        layer_multiplier: int = 64,
    ):
        super().__init__()

        self.enc = Encoder(
            in_channels=in_channels,
            out_channels=256,
            layer_multiplier=layer_multiplier,
        )

        self.dec = Decoder(
            in_channels=256,
            out_channels=out_channels,
            repeat_num=4,
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the generator.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Generated images and encoded images.
        """
        encoded_images = self.enc(images)

        if self.training:
            noise = torch.randn(encoded_images.size()).cuda(
                encoded_images.detach().get_device()
            )
            gen_images = self.decode(encoded_images + noise)
        else:
            gen_images = self.decode(encoded_images)

        return gen_images, encoded_images

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the input images.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Encoded images and noise.
        """
        encoded_images = self.enc(images)
        noise = torch.randn(encoded_images.size()).cuda(
            encoded_images.detach().get_device(),
        )
        return encoded_images, noise

    def decode(self, encoded_images: torch.Tensor) -> torch.Tensor:
        """Decode the encoded images.

        Args:
            encoded_images (torch.Tensor): Encoded images.

        Returns:
            torch.Tensor: Decoded images.
        """
        images = self.dec(encoded_images)
        return images


class Encoder(nn.Module):
    """Encoder for the GAN.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        out_channels (int, optional): Number of output channels. Defaults to 512.
        layer_multiplier (int, optional): Number of channels multiplier. Defaults to 64.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 512,
        layer_multiplier: int = 64,
    ):
        super().__init__()

        self.layers = []

        self.layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_multiplier,
                kernel_size=7,
                stride=1,
                padding=3,
            )
        )

        n = layer_multiplier

        for _ in range(2):
            self.layers.append(
                ConvBlock(
                    in_channels=n,
                    out_channels=n * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            n *= 2

        self.layers.append(
            ResBlocks(channels=n, repeat_num=4),
        )

        self.model = nn.Sequential(*self.layers)
        self.output_dim = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Encoded images.
        """
        return self.model(x)


class Decoder(nn.Module):
    """Decoder for the GAN.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 512.
        out_channels (int, optional): Number of output channels. Defaults to 3.
        repeat_num (int, optional): Number of times to repeat the ResBlocks. Defaults to 4.
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 3,
        repeat_num: int = 4,
    ):
        super().__init__()

        self.layers = []

        self.layers.append(ResBlocks(channels=in_channels, repeat_num=repeat_num))

        n = in_channels

        for _ in range(2):
            self.layers += [
                nn.Upsample(scale_factor=2),
                ConvBlock(n, n // 2, 5, 1, 2),
            ]

            n //= 2

        self.layers += [
            ConvBlock(
                in_channels=n,
                out_channels=out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Decoded images.
        """
        return self.model(x)


if __name__ == "__main__":
    x_test = torch.randn(size=(1, 3, 256, 256)).to("cuda")
    y_test = torch.randn(size=(32, 512, 256, 256))
    # D = Decoder(512, 3, 4).to("cuda")
    G = Generator(3, 3, 64).to("cuda")

    print(G(x_test)[0].shape, G(x_test)[1].shape)
