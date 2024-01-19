"""Configurations for the model's training, loading, saving, evaluation, and Data transforms."""
from enum import Enum
from pathlib import Path

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from unit.cli import custom_arg_parser

# TODO: Add logger instead of all the print statements.
# TODO: Change tensorboard summarywriter to be a global entity instead of passing it to functions
# TODO: currently the cfg.py acts as a gloabl variable instead of configuration. We need to modularize by
# making sure that these cfg vars are always passed in as arguments not used as a global variable.
# TODO: move all default values to a config.yaml and read from it.


class DatasetType(Enum):
    """Enum for the dataset type."""

    ANIME_DATASET = Path(
        "/media/omarabdelgawad/New Volume/Datasets/image_coloring/anime_dataset/"
    )
    NATURAL_VIEW_DATASET = Path(
        "/media/omarabdelgawad/New Volume/Datasets/image_coloring/natural_view/"
    )
    Edges2Shoes = Path(
        "/media/omarabdelgawad/New Volume/Datasets/image_coloring/edges2shoes/"
    )


class PaddingType(Enum):
    """Enum for the padding type."""

    REFLECT = "reflect"
    REPLICATE = "replicate"
    ZERO = "zero"


class NormalizationType(Enum):
    """Enum for the normalization type."""

    BATCH = "batch"
    INSTANCE = "instance"
    LAYER = "layer"
    NONE = None


class ActivationType(Enum):
    """Enum for the activation type."""

    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    NONE = None


args = custom_arg_parser()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS_IMG = 3
BETA_OPTIM = (0.5, 0.999)
WEIGHT_DECAY = 0.0001
LR_POLICY = "step"
STEP_SIZE = 100000
GAMMA = 0.5
INIT = "kaiming"

GAN_WEIGHT = 1
RECONSTRUCTION_X_WEIGHT = 10
RECONSTRUCTION_H_WEIGHT = 0
RECONSTRUCTION_KL_WEIGHT = 0.01
RECONSTRUCTION_X_CYC_WEIGHT = 10
RECONSTRUCTION_KL_CYC_WEIGHT = 0.01

LEARNING_RATE = args.rate
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
IMAGE_SIZE = args.image_size
NUM_EPOCHS = args.num_epochs
LOAD_MODEL = args.load_model
SAVE_MODEL = args.save_model
NUM_IMAGES_DATASET = args.num_images_dataset
VAL_BATCH_SIZE = args.val_batch_size
CHOSEN_DATASET = DatasetType.Edges2Shoes


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


CHECKPOINT_DIR = Path("./out/last_trained_weights")
TRAIN_DATASET_PATH = CHOSEN_DATASET.value / "train"
VAL_DATASET_PATH = CHOSEN_DATASET.value / "val"
EVALUATION_PATH = Path("./out/evaluation")

# TODO: understand the augmentations below and improve them (maybe add more augmentations).
both_transform = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
    ],
    additional_targets={"image0": "image"},
)

# this is equivalent to first domain transform
transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.1),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ]
)

# this is equivalent to second domain transform
transform_only_mask = A.Compose(
    [
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)
