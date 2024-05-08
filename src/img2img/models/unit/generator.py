"""Generator for the UNIT architecture."""

import torch
from torch import nn

from img2img.nn import ResBlocks, ConvBlock
from img2img.cfg import PaddingType, ActivationType, NormalizationType


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
        dim: int = 64,
        n_downsample: int = 2,
        n_res: int = 4,
        activ: ActivationType = ActivationType.RELU,
        pad_type: PaddingType = PaddingType.REFLECT,
        norm: NormalizationType = NormalizationType.INSTANCE,
    ):
        super().__init__()

        self.enc = Encoder(
            in_channels=in_channels,
            dim=dim,
            n_downsample=n_downsample,
            n_res=n_res,
            norm=norm,
            activ=activ,
            pad_type=pad_type,
        )

        self.dec = Decoder(
            in_channels=256,
            out_channels=in_channels,
            n_upsample=n_downsample,
            n_res=n_res,
            res_norm=norm,
            activ=activ,
            pad_type=pad_type,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the generator.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Generated images and encoded images.
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

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input images.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Encoded images and noise.
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
        images: torch.Tensor = self.dec(encoded_images)
        return images


class Encoder(nn.Module):
    """Encoder for the GAN. out_channels is 256.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        out_channels (int, optional): Number of output channels. Defaults to 512.
        layer_multiplier (int, optional): Number of channels multiplier. Defaults to 64.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 64,
        n_downsample: int = 2,
        n_res: int = 4,
        norm: NormalizationType = NormalizationType.INSTANCE,
        activ: ActivationType = ActivationType.RELU,
        pad_type: PaddingType = PaddingType.REFLECT,
    ):
        super().__init__()

        self.layers: list[nn.Module] = []

        self.layers.append(
            ConvBlock(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=7,
                stride=1,
                padding=3,
                normalization_type=norm,
                padding_type=pad_type,
                activation_type=activ,
            )
        )

        for _ in range(n_downsample):
            self.layers.append(
                ConvBlock(
                    in_channels=dim,
                    out_channels=dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    normalization_type=norm,
                    padding_type=pad_type,
                    activation_type=activ,
                )
            )
            dim *= 2

        self.layers.append(
            ResBlocks(
                channels=dim,
                num_blocks=n_res,
                normalization_type=norm,
                padding_type=pad_type,
                activation_type=activ,
            )
        )

        self.model = nn.Sequential(*self.layers)
        self.output_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Encoded images.
        """
        x = self.model(x)
        return x


class Decoder(nn.Module):
    """Decoder for the GAN.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 512.
        out_channels (int, optional): Number of output channels. Defaults to 3.
        repeat_num (int, optional): Number of times to repeat the ResBlocks. Defaults to 4.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 3,
        n_upsample: int = 2,
        n_res: int = 4,
        res_norm: NormalizationType = NormalizationType.INSTANCE,
        activ: ActivationType = ActivationType.RELU,
        pad_type: PaddingType = PaddingType.REFLECT,
    ):
        super().__init__()

        self.layers: list[nn.Module] = []

        self.layers.append(
            ResBlocks(
                channels=in_channels,
                num_blocks=n_res,
                normalization_type=res_norm,
                padding_type=pad_type,
                activation_type=activ,
            )
        )
        # FIXME: replace the upsampling with a convtranspose layer
        for _ in range(n_upsample):
            self.layers += [
                nn.Upsample(scale_factor=2),
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    normalization_type=NormalizationType.INSTANCE,  # FIXME: original implementation uses layer norm
                    padding_type=pad_type,
                    activation_type=activ,
                ),
            ]
            in_channels //= 2

        self.layers.append(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                normalization_type=NormalizationType.NONE,
                padding_type=pad_type,
                activation_type=ActivationType.TANH,
            ),
        )

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Decoded images.
        """
        # print(f"from decoder x.shape is {x.shape}")
        x = self.model(x)
        return x


# TODO: generate a test for checking output shapes of this module
def test():
    x_test = torch.randn(size=(1, 3, 256, 256))
    y_test = torch.randn(size=(32, 512, 256, 256))
    # D = Decoder(512, 3, 4).to("cuda")
    gen = Generator(3, 64)
    res = gen(x_test)
    print(res[0].shape, res[1].shape)


if __name__ == "__main__":
    """Nothing here"""
    # test()
