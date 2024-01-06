import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from tunit import cfg


class AFHQCatDataset(Dataset):
    """Dataset class for AFHQ Cat Colorization dataset.

    Args:
        root_dir (root_dir): Path for dataset dir.
    """

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)[: cfg.NUM_IMAGES_DATASET]
        print(f"The length of the dataset is: {len(self.list_files)}")

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file_name = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file_name)
        image = np.array(Image.open(img_path))
        target_image = image[:, : image.shape[1] // 2, :]
        input_image = image[:, image.shape[1] // 2 :, :]

        augmentations = cfg.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]

        input_image = cfg.transform_only_input(image=input_image)["image"]
        target_image = cfg.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


def create_dataset(root_dir: Path | str, dataset_type: cfg.DatasetType) -> Dataset:
    """Create a Dataset from given root_dir and dataset_type.

    Args:
        root_dir (Path | str): Path for Dataset dir.
        dataset_type (cfg.DatasetType): Type for dataset to create.

    Raises:
        ValueError: If dataset_type is not supported.

    Returns:
        Dataset: Pytorch Dataset object.
    """
    if dataset_type == cfg.DatasetType.AFHQ_CATS_DATASET:
        return AFHQCatDataset(root_dir)
    else:
        raise ValueError("Dataset type not supported")
