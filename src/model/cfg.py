"""Configurations for the model's training, loading, saving, evaluation, and Data transforms."""
from enum import Enum
from pathlib import Path
import argparse
from typing import Optional
from typing import Sequence

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# TODO: Make these variables such as load_model and save_model into command line arguments.
# TODO: Add logger instead of all the print statements.
# TODO: Change tensorboard summarywriter to be a global entity instead of passing it to functions


class DatasetType(Enum):
    ANIME_DATASET = Path(
        "/media/omarabdelgawad/New Volume/Datasets/image_coloring/anime_dataset/"
    )
    NATURAL_VIEW_DATASET = Path(
        "/media/omarabdelgawad/New Volume/Datasets/image_coloring/natural_view/"
    )

# custom parser
def positive_float(value: str) -> float:
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive float value")
    return ivalue

def positive_int(value) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue

def custom_arg_parser(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="An unsupervised image-to-image translation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-lr", "--learning_rate",
        type=positive_float,
        default=2e-4,
        help="Learning rate",
        metavar="LEARNING_RATE",
        dest="lr",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=positive_int,
        default=16,
        help="Batch size",
        metavar="BATCH_SIZE",
        dest="b",
    )
    parser.add_argument(
        "-w", "--num_workers",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Number of workers",
        metavar="NUM_WORKERS",
        dest="w",
    )
    parser.add_argument(
        "-i", "--image_size",
        type=int,
        default=256,
        choices=[256, 512, 1024],
        help="Image size",
        metavar="IMAGE_SIZE",
        dest="i",
    )
    parser.add_argument(
        "-e", "--num_epochs",
        type=positive_int,
        default=100,
        help="Number of epochs",
        metavar="NUM_EPOCHS",
        dest="e",
    )
    parser.add_argument(
        "-l", "--load_model",
        action="store_false",
        default=True,
        help="Load model",
        dest="l",
    )
    parser.add_argument(
        "-s", "--save_model",
        action="store_true",
        default=False,
        help="Save model",
        dest="s",
    )
    parser.add_argument(
        "-d", "--num_images_dataset",
        type=positive_int,
        default=1000,
        help="Number of images in dataset",
        metavar="NUM_IMAGES_DATASET",
        dest="d",
    )
    parser.add_argument(
        "-v", "--val_batch_size",
        type=positive_int,
        default=8,
        help="Validation batch size",
        metavar="VAL_BATCH_SIZE",
        dest="v",
    )

    args = parser.parse_args(argv)
    return args

args = custom_arg_parser()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = args.lr
BATCH_SIZE = args.b
NUM_WORKERS = args.w
IMAGE_SIZE = args.i
CHANNELS_IMG = 3
L_1_LAMBDA = 100
NUM_EPOCHS = args.e
LOAD_MODEL = args.l
SAVE_MODEL = args.s
CHECKPOINT_DISC = Path("last_trained_weights/disc.pth.tar")
CHECKPOINT_GEN = Path("last_trained_weights/gen.pth.tar")
CHOSEN_DATASET = DatasetType.NATURAL_VIEW_DATASET
TRAIN_DATASET_PATH = CHOSEN_DATASET.value / "train"
VAL_DATASET_PATH = CHOSEN_DATASET.value / "val"
EVALUATION_PATH = Path("./evaluation")
NUM_IMAGES_DATASET = args.d
VAL_BATCH_SIZE = args.v

both_transform = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE), A.HorizontalFlip(p=0.5)],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.1),
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
