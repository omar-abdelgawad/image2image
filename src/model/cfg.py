"""Configurations for the model's training, loading, saving, evaluation, and Data transforms."""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# TODO: Make these variables such as load_model and save_model into command line arguments.
# TODO: Add logger instead of all the print statements.
# TODO: Change paths to be os agnostic using pathlib.
# TODO: Change tensorboard summarywriter to be a global entity instead of passing it to functions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L_1_LAMBDA = 100
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
TRAIN_DATASET_PATH = "/media/omarabdelgawad/New Volume/Datasets/archive/data/train"
VAL_DATASET_PATH = "/media/omarabdelgawad/New Volume/Datasets/archive/data/val"
EVALUATION_PATH = "./evaluation"
NUM_IMAGES_DATASET = 1000
VAL_BATCH_SIZE = 8

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