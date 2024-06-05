"""Configuration module Interface."""

from pathlib import Path

from torch import cuda

from img2img.cli import get_main_parser

from .transform import get_transforms

from .enums import ActivationType, DatasetType, NormalizationType, PaddingType

# TODO: Add logger instead of all the print statements.


args = get_main_parser()
DEVICE = "cuda" if cuda.is_available() else "cpu"
LEARNING_RATE = args.rate
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
IMAGE_SIZE = args.image_size
NORM_MEAN = 0.5
NORM_STD = 0.5
CHECKPOINT_PERIOD = 5
NUM_EPOCHS = args.num_epochs
LOAD_MODEL = args.load_model
SAVE_MODEL = args.save_model
both_transform, transform_only_input, transform_only_mask, transforms, prediction_transform = get_transforms(IMAGE_SIZE,
                                                                                                             NORM_MEAN,
                                                                                                             NORM_STD)


class GEN_HYPERPARAMS:
    """Hyperparameters for the generator."""

    DIM = 64
    NORM = NormalizationType.INSTANCE
    ACTIV = ActivationType.RELU
    N_DOWNSAMPLE = 2
    N_RES = 4
    PAD_TYPE = PaddingType.REFLECT


class DIS_HYPERPARAMS:
    """Hyperparameters for the discriminator."""

    DIM = 64
    NORM = NormalizationType.NONE
    ACTIV = ActivationType.LEAKY_RELU
    N_LAYER = 4
    GAN_TYPE = "lsgan"
    NUM_SCALES = 3
    PAD_TYPE = PaddingType.REFLECT
