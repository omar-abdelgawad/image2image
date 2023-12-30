# pylint: skip-file
import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        