"""Tests for the shapes of the pix2pix models."""

import pytest
import torch

from img2img.models.pix2pix.discriminator import Discriminator
from img2img.models.pix2pix.generator import Generator

shapes = [(5, 3, 256, 256), (1, 3, 512, 512)]


@pytest.mark.parametrize(
    "shape",
    shapes,
)
def test_generator_shapes(shape):
    """Test generator out shapes."""
    gen = Generator()
    # input and output shapes should match
    assert gen(torch.rand(*shape)).shape == shape


@pytest.mark.parametrize(
    "shape",
    shapes,
)
def test_discriminator_shapes(shape):
    """Test discriminator out shapes."""
    out_dict = {256: 30, 512: 62}
    o = out_dict[shape[2]]
    disc = Discriminator()
    x = y = torch.rand(*shape)
    assert disc(x, y).shape == (shape[0], 1, o, o)
