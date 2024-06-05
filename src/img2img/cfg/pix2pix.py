from . import *

BETA_OPTIM = (0.5, 0.999)
NUM_WORKERS = args.num_workers
CHANNELS_IMG = 3
L_1_LAMBDA = 100
NORM_MEAN = 0.5
NORM_STD = 0.5
CHOSEN_DATASET = DatasetType.ANIME_DATASET
TRAIN_DATASET_PATH = CHOSEN_DATASET.value / "train"
VAL_DATASET_PATH = CHOSEN_DATASET.value / "val"
OUT_PATH = Path("./out")
NUM_IMAGES_DATASET = args.num_images_dataset
VAL_BATCH_SIZE = args.val_batch_size


# Hyperparameters for the generator and discriminator
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
