"""Generator model for Pix2Pix."""

import torch
from torch import nn


class Block(nn.Module):
    """Block for Pix2Pix Generator model. Consists of Conv2d/ConvTranspose2d, BatchNorm2d,
    and ReLU/LeakyReLU.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        down (bool, optional): Determines if block is at first or second half of
        U-net. Defaults to True.
        act (str, optional): Activation function type. Defaults to "relu".
        use_dropout (bool, optional): Whether to use dropout or not. Defaults to False.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            down: bool = True,
            act: str = "relu",
            use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"
            )
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


# TODO: make it a multi-modal generator by adding a random noise
#  vector z sampled from a normal distribution.
# TODO: Apply Diversity-Sensitive Conditional GANs DSGANs to the generator.
class Generator(nn.Module):
    """Generator Class for Pix2Pix model.

    Args:
        in_channels (int, optional): _description_. Defaults to 3.
        features (int, optional): _description_. Defaults to 64.
    """

    def __init__(self, in_channels=3, features=64) -> None:
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(
            features, features * 2, down=True, act="leaky", use_dropout=False
        )
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                features * 8, features * 8, kernel_size=4, stride=2, padding=1
            ),  # 4x4
            nn.ReLU(),
        )
        self.up1 = Block(
            features * 8, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(
            features * 2 * 2, features, down=False, act="relu", use_dropout=False
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, in_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Pix2Pix's U-net generator. Downsamples the image then
        upsamples using convtranspose2d.

        Args:
            x (torch.Tensor): Batched input Image(s) tensor.

        Returns:
            torch.Tensor: Batched output Image(s) tensor
        """
        d1: torch.Tensor = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        d1 = self.final_up(torch.cat([up7, d1], 1))
        return d1
