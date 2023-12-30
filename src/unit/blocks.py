"""Blocks for the generator and discriminator."""
import torch
from torch import nn

from unit.cfg import NormalizationType, PaddingType, ActivationType


class ConvBlock(nn.Module):
    """Convolutional block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size.
        stride (int): Stride.
        padding (int, optional): Padding. Defaults to 0.
        norm (NormalizationType, optional): Normalization type. Defaults to NormalizationType.NONE.
        padding_type (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
        activation_type (ActivationType, optional): Act type. Defaults to ActivationType.RELU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        normalization_type: NormalizationType = NormalizationType.NONE,
        padding_type: PaddingType = PaddingType.ZERO,
        activation_type: ActivationType = ActivationType.RELU,
    ) -> None:
        super().__init__()

        self.padding = padding
        self.norm_dim = out_channels

        self.padding_layer = self._padding_type_selector(padding_type)
        self.normalization_layer = self._normalization_selector(normalization_type)
        self._activation_layer = self._activation_layer_selector(activation_type)

        self.model = nn.Sequential(
            self.padding_layer if self.padding_layer is not None else nn.Identity(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            self.normalization_layer
            if self.normalization_layer is not None
            else nn.Identity(),
            self._activation_layer
            if self._activation_layer is not None
            else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def _padding_type_selector(self, padding_type: PaddingType) -> nn.Module:
        """Selects the padding type.

        Args:
            padding_type (PaddingType): Padding type.

        Raises:
            NotImplementedError: If the padding type is not implemented.

        Returns:
            nn.Module: Padding layer.
        """
        match padding_type:
            case PaddingType.REFLECT:
                return nn.ReflectionPad2d(self.padding)
            case PaddingType.REPLICATE:
                return nn.ReplicationPad2d(self.padding)
            case PaddingType.ZERO:
                return nn.ZeroPad2d(self.padding)
            case _:
                raise NotImplementedError(
                    f"Padding type {padding_type} is not implemented."
                )

    def _normalization_selector(
        self, normalization_type: NormalizationType
    ) -> nn.Module:
        """Selects the normalization type.

        Args:
            normalization_type (NormalizationType): Normalization type.
            out_channels (int): Number of output channels.

        Raises:
            NotImplementedError: If the normalization type is not implemented.

        Returns:
            nn.Module: Normalization layer.
        """
        match normalization_type:
            case NormalizationType.BATCH:
                return nn.BatchNorm2d(self.norm_dim)
            case NormalizationType.INSTANCE:
                return nn.InstanceNorm2d(self.norm_dim)
            case NormalizationType.LAYER:
                return nn.LayerNorm(self.norm_dim)
            case NormalizationType.NONE:
                return None
            case _:
                raise NotImplementedError(
                    f"Normalization type {normalization_type} is not implemented."
                )

    def _activation_layer_selector(self, activation_type: ActivationType) -> nn.Module:
        """Selects the activation type.

        Args:
            activation_type (ActivationType): Activation type.

        Raises:
            NotImplementedError: If the activation type is not implemented.

        Returns:
            nn.Module: Activation layer.
        """
        match activation_type:
            case ActivationType.RELU:
                return nn.ReLU()
            case ActivationType.LEAKY_RELU:
                return nn.LeakyReLU(0.2)
            case ActivationType.TANH:
                return nn.Tanh()
            case ActivationType.SIGMOID:
                return nn.Sigmoid()
            case _:
                raise NotImplementedError(
                    f"Activation type {activation_type} is not implemented."
                )


class ConvBlocks(nn.Module):
    """Convolutional blocks.

    Args:
        layer_multiplier (int): Number of channels multiplier.
        max_layer_multiplier (int): Maximum channels number.
        kernel_size (int): Kernel size.
        stride (int): Stride.
        padding (int, optional): Padding. Defaults to 0.
        padding_type (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
        activation_type (ActivationType, optional): Act type. Defaults to ActivationType.RELU.
    """

    def __init__(
        self,
        layer_multiplier: int = 64,
        max_layer_multiplier: int = 1024,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 0,
        padding_type: PaddingType = PaddingType.ZERO,
        normalization_type: NormalizationType = NormalizationType.NONE,
        activation_type: ActivationType = ActivationType.RELU,
    ) -> None:
        super().__init__()

        self.layers = []
        self.layer_multiplier = layer_multiplier

        while self.layer_multiplier != max_layer_multiplier:
            self.layers.append(
                ConvBlock(
                    self.layer_multiplier,
                    self.layer_multiplier * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    padding_type=padding_type,
                    normalization_type=normalization_type,
                    activation_type=activation_type,
                )
            )
            self.layer_multiplier *= 2

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolutional blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


class ResBlock(nn.Module):
    """Residual block.

    Args:
        channels (int): Number of channels.
        kernel_size (int): Kernel size.
        stride (int): Stride.
        padding (int, optional): Padding. Defaults to 0.
        padding_type (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
        activation_type (ActivationType, optional): Act type. Defaults to ActivationType.RELU.
    """

    def __init__(
        self,
        channels: int = 256,
        kernel_size=3,
        stride=1,
        padding=1,
        normalization_type=NormalizationType.NONE,
        padding_type=PaddingType.ZERO,
        activation_type=ActivationType.RELU,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                normalization_type=normalization_type,
                padding_type=padding_type,
                activation_type=activation_type,
            ),
            ConvBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                normalization_type=normalization_type,
                padding_type=padding_type,
                activation_type=activation_type,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x) + x


class ResBlocks(nn.Module):
    """Residual blocks.

    Args:
        channels (int): Number of channels.
        repeat_num (int): Number of times to repeat the ResBlocks.
    """

    def __init__(
        self,
        channels: int = 256,
        repeat_num: int = 4,
    ) -> None:
        super().__init__()

        self.layers = []

        for _ in range(repeat_num):
            self.layers.append(
                ResBlock(
                    channels=channels,
                )
            )

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


if __name__ == "__main__":
    print("\n\n")
    print("ConvBlock ============================", end="\n\n")
    print(ConvBlock(3, 64, 4, 2), (3, 256, 256))
    print("\n\n")
    print("ConvBlocks ===========================", end="\n\n")
    print(ConvBlocks(), (3, 256, 256))
    print("\n\n")
    print("ResBlock =============================", end="\n\n")
    print(ResBlock(), (256, 256, 256))
    print("\n\n")
    print("ResBlocks ============================", end="\n\n")
    print(ResBlocks(), (256, 256, 256))
