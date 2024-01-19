"""Configurations for the model's training, loading, saving, evaluation, and Data transforms."""
from enum import Enum
from pathlib import Path

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Callable

from pix2pix.cli import custom_arg_parser

# TODO: Add logger instead of all the print statements.
# TODO: Change tensorboard summarywriter to be a global entity instead of passing it to functions


class DatasetType(Enum):
    """Enum for the dataset type."""

    ANIME_DATASET = Path(
        "/media/omarabdelgawad/New Volume/Datasets/image_coloring/anime_dataset/"
    )
    NATURAL_VIEW_DATASET = Path(
        "/media/omarabdelgawad/New Volume/Datasets/image_coloring/natural_view/"
    )


args = custom_arg_parser()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = args.rate
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
IMAGE_SIZE = args.image_size
CHANNELS_IMG = 3
L_1_LAMBDA = 100
NUM_EPOCHS = args.num_epochs
LOAD_MODEL = args.load_model
SAVE_MODEL = args.save_model
CHECKPOINT_DISC = Path("last_trained_weights/disc.pth.tar")
CHECKPOINT_GEN = Path("last_trained_weights/gen.pth.tar")
CHOSEN_DATASET = DatasetType.NATURAL_VIEW_DATASET
TRAIN_DATASET_PATH = CHOSEN_DATASET.value / "train"
VAL_DATASET_PATH = CHOSEN_DATASET.value / "val"
EVALUATION_PATH = Path("./evaluation")
NUM_IMAGES_DATASET = args.num_images_dataset
VAL_BATCH_SIZE = args.val_batch_size

both_transform = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE), A.HorizontalFlip(p=0.5)],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.1),
        # TODO: calculate mean and std for the dataset instead of using these values.
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

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
