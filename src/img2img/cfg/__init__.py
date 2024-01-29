""" Configuration module Interface. """
from pathlib import Path

from torch import cuda

from img2img.cli import custom_arg_parser
from .enums import DatasetType, PaddingType, NormalizationType, ActivationType

# TODO: Add logger instead of all the print statements.

args = custom_arg_parser()
DEVICE = "cuda" if cuda.is_available() else "cpu"
LEARNING_RATE = args.rate
BETA_OPTIM = (0.5, 0.999)
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
IMAGE_SIZE = args.image_size
CHANNELS_IMG = 3
L_1_LAMBDA = 100
NORM_MEAN = 0.5
NORM_STD = 0.5
CHECKPOINT_PERIOD = 5
NUM_EPOCHS = args.num_epochs
LOAD_MODEL = args.load_model
SAVE_MODEL = args.save_model
CHOSEN_DATASET = DatasetType.ANIME_DATASET
TRAIN_DATASET_PATH = CHOSEN_DATASET.value / "train"
VAL_DATASET_PATH = CHOSEN_DATASET.value / "val"
OUT_PATH = Path("./out")
NUM_IMAGES_DATASET = args.num_images_dataset
VAL_BATCH_SIZE = args.val_batch_size

# tunit config
CHANNELS_MULTIPLIER = 64
K = args.cluster_number

# unit config
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


import albumentations as A
from albumentations.pytorch import ToTensorV2

both_transform = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
    ],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.1),
        # TODO: calculate mean and std for the dataset instead of using these values.
        A.Normalize(
            mean=[NORM_MEAN, NORM_MEAN, NORM_MEAN],
            std=[NORM_STD, NORM_STD, NORM_STD],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(
            mean=[NORM_MEAN, NORM_MEAN, NORM_MEAN],
            std=[NORM_STD, NORM_STD, NORM_STD],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


"""unit transforms

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
"""

# TODO: make transforms differ from model to another
