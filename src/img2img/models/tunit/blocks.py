"""Blocks for TUNIT"""

import torch
from torch import nn
from torch.nn import functional as F

from img2img.utils.tunit import FRN, AdaptiveInstanceNorm2d


class GenConvBlock(nn.Module):
    """Convolutional block for the generator.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size for conv kernel. Defaults to 3.
        stride (int, optional): Stride for conv kernel. Defaults to 1.
        padding (int, optional): Padding for conv kernel. Defaults to 0.
        padd_type (str, optional): Padding type for conv kernel. Defaults to "zero".
        ins (bool, optional): Whether to use instance norm. Defaults to True.
        up (bool, optional): Whether to use upsampling. Defaults to False.
        use_bias (bool, optional): Whether to use bias. Defaults to True.
        use_sn (bool, optional): Whether to use spectral norm. Defaults to False.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            padd_type: str = "zero",
            ins: bool = True,
            up: bool = False,
            use_bias: bool = True,
            use_sn: bool = False,
    ) -> None:
        super().__init__()

        self.use_bias = use_bias

        if padd_type == "reflect":
            self.padd = nn.ReflectionPad2d(padding)
        elif padd_type == "replicate":
            self.padd = nn.ReplicationPad2d(padding)
        elif padd_type == "zero":
            self.padd = nn.ZeroPad2d(padding)
        else:
            assert 0, f"Unsupported padding type: {padd_type}"

        self.model = nn.Sequential()
        if up:
            self.model.add_module("upsample", nn.Upsample(scale_factor=2))
        self.model.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=self.use_bias,
            ),
        )
        if use_sn:
            self.model.add_module("sn", nn.utils.spectral_norm())

        if ins:
            self.model.add_module(
                "norm",
                nn.InstanceNorm2d(num_features=out_channels),
            )
        else:
            self.model.add_module(
                "norm",
                AdaptiveInstanceNorm2d(num_features=out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Genertor's Convolutional Block.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.padd(x)
        return self.model(x)


class GuidingNetworkConvBlock(nn.Module):
    """Guiding Network's block for the generator.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size for conv kernel. Defaults to 3.
        stride (int, optional): Stride for conv kernel. Defaults to 1.
        padding (int, optional): Padding for conv kernel. Defaults to 0.
        pool (bool, optional): Whether to use max pooling. Defaults to True.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            pool: bool = True,
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
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=False),
        )
        if pool:
            self.model.add_module(
                "norm",
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Guiding's Convolutional Block.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
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

        self.pool = pool

        self.layer_1 = nn.Sequential(
            FRN(num_features=in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

        self.layer_2 = nn.Sequential(
            FRN(num_features=in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        x = self.layer_1(x)
        if self.pool:
            x = F.avg_pool2d(x, kernel_size=2)
        x = self.layer_2(x)
        return x


class GenResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            ins=True,
            padd_type="zero",
            use_sn=False,
    ):
        super().__init__()

        self.model = nn.Sequential(
            GenConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ins=ins,
                padd_type=padd_type,
                use_sn=use_sn,
            ),
            nn.ReLU(inplace=True),
            GenConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ins=ins,
                padd_type=padd_type,
                use_sn=use_sn,
            ),
            nn.ReLU(inplace=True),
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


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm="none", act="relu", use_sn=False):
        super().__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

        # initialize normalization
        norm_dim = out_dim
        if norm == "bn":
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, f"Unsupported normalization: {norm}"

        # initialize activation
        if act == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == "tanh":
            self.activation = nn.Tanh()
        elif act == "none":
            self.activation = None
        else:
            assert 0, f"Unsupported activation: {act}"

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out
