import torch
from torch import nn

# class DisConvBlock(nn.Module):


# class DiscResBlocks(nn.Module):


# class DiscResBlock(nn.Module):
#     def __init__(
#         self, in_channels, out_channels, kernel_size, stride, padding, pool=True
#     ):
#         super().__init__()

#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
#             nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
#         )


x = torch.randn(size=(32, 128, 64, 64))

conv = nn.Conv2d(128, 256, 1, 1, 0, bias=False)



x_new = conv(x)

print(x_new.shape)
