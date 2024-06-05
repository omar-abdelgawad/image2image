from . import *

TRAIN_DIR = "/media/omarabdelgawad/New Volume/Datasets/vangogh2photo/train"
VAL_DIR = "/media/omarabdelgawad/New Volume/Datasets/vangogh2photo/val"
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
BATCH_SIZE = 2
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"
CHOSEN_DATASET = DatasetType.VANGOGH2PHOTO
TRAIN_DATASET_PATH = CHOSEN_DATASET.value / "train"
VAL_DATASET_PATH = CHOSEN_DATASET.value / "val"
