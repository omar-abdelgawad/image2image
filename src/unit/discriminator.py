"""Discriminator for the GAN."""
import torch
from torch import nn

from unit.blocks import ConvBlock, ConvBlocks
from unit.cfg import NormalizationType, PaddingType, ActivationType


class Discriminator(nn.Module):
    """Discriminator model for the unit GAN.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        out_channels (int, optional): Number of output channels. Defaults to 1.
        layer_multiplier (int, optional): Number of channels multiplier. Defaults to 64.
        max_layer_multiplier (int, optional): Max number of channels multiplier. Defaults to 1024.
        normalization_type (NormalizationType, optional): Norm. Defaults to NormalizationType.NONE.
        padding_type (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
        activation_type (ActivationType, optional): Act type. Defaults to ActivationType.LEAKY_RELU.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        layer_multiplier: int = 64,
        max_layer_multiplier: int = 1024,
        normalization_type: NormalizationType = NormalizationType.NONE,
        padding_type: PaddingType = PaddingType.REFLECT,
        activation_type: ActivationType = ActivationType.LEAKY_RELU,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=layer_multiplier,
                kernel_size=4,
                stride=2,
                padding=1,
                normalization_type=NormalizationType.NONE,
                padding_type=padding_type,
                activation_type=activation_type,
            ),
            ConvBlocks(
                layer_multiplier,
                max_layer_multiplier,
                kernel_size=4,
                stride=2,
                padding=1,
                normalization_type=normalization_type,
                padding_type=padding_type,
                activation_type=activation_type,
            ),
            nn.Conv2d(
                in_channels=max_layer_multiplier,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the discriminator.

        Args:
            x (torch.Tensor): forward pass input.

        Returns:
            torch.Tensor: forward pass output.
        """
        x = self.model(x)
        return x

    # def calc_dis_loss(self, input_fake, input_real):
    #     # calculate the loss to train D
    #     outs0 = self.forward(input_fake)
    #     outs1 = self.forward(input_real)
    #     loss = 0

    #     for it, (out0, out1) in enumerate(zip(outs0, outs1)):
    #         if self.gan_type == 'lsgan':
    #             loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
    #         elif self.gan_type == 'nsgan':
    #             all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
    #             all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
    #             loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
    #                                F.binary_cross_entropy(F.sigmoid(out1), all1))
    #         else:
    #             assert 0, "Unsupported GAN type: {}".format(self.gan_type)
    #     return loss

    # def calc_gen_loss(self, input_fake):
    #     # calculate the loss to train G
    #     outs0 = self.forward(input_fake)
    #     loss = 0
    #     for it, (out0) in enumerate(outs0):
    #         if self.gan_type == 'lsgan':
    #             loss += torch.mean((out0 - 1)**2) # LSGAN
    #         elif self.gan_type == 'nsgan':
    #             all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
    #             loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
    #         else:
    #             assert 0, "Unsupported GAN type: {}".format(self.gan_type)
    #     return loss


if __name__ == "__main__":
    x_test = torch.randn(size=(32, 3, 256, 256))
    D = Discriminator(
        3,
        1,
        64,
        1024,
        normalization_type=NormalizationType.NONE,
        activation_type=ActivationType.LEAKY_RELU,
        padding_type=PaddingType.REFLECT,
    )

    print(D(x_test).shape)
