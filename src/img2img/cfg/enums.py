"""Global Enums/Types for the project."""

from enum import Enum
from pathlib import Path


class DatasetType(Enum):
    """Enum for the dataset type."""

    ANIME_DATASET = Path("/media/omarabdelgawad/New Volume/Datasets/Anime_Dataset")
    NATURAL_VIEW_DATASET = Path("/media/omarabdelgawad/New Volume/Datasets/Natural_View")
    EDGES2SHOES = Path("/media/omarabdelgawad/New Volume/Datasets/Edges2Shoes")
    AFHQ_CATS_DATASET = Path("/media/omarabdelgawad/New Volume/Datasets/AFHQ_Cats")
    VANGOGH2PHOTO = Path("/media/omarabdelgawad/New Volume/Datasets/vangogh2photo")
    yukiyoe = Path("/media/omarabdelgawad/New Volume/Datasets/yukiyoe")
    monet = Path("/media/omarabdelgawad/New Volume/Datasets/monet")


class PaddingType(Enum):
    """Enum for the padding type."""

    REFLECT = "reflect"
    REPLICATE = "replicate"
    ZERO = "zeros"


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
