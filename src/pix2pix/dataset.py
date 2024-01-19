"""Dataset Module. Contains Dataset classes that can be constructed using the path.
 Currently Contains AnimeDataset class."""
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset

from pix2pix import cfg


class AnimeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset class for Anime Colorization dataset.

    Args:
        root_dir (root_dir): Path for dataset dir.
    """

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)[: cfg.NUM_IMAGES_DATASET]
        print(f"The length of the dataset is: {len(self.list_files)}")

    def __len__(self) -> int:
        return len(self.list_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_file_name = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file_name)
        image = np.array(Image.open(img_path))
        target_image = image[:, : image.shape[1] // 2, :]
        input_image = image[:, image.shape[1] // 2 :, :]

        augmentations = cfg.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]

        out_input_image = cfg.transform_only_input(image=input_image)["image"]
        out_target_image = cfg.transform_only_mask(image=target_image)["image"]

        return out_input_image, out_target_image


class NaturaViewDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset class for Natural View Colorization dataset.

    Args:
        root_dir (root_dir): Path for dataset dir.
    """

    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)[: cfg.NUM_IMAGES_DATASET]
        print(f"The length of the dataset is: {len(self.list_files)}")

    def __len__(self) -> int:
        return len(self.list_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_file_name = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # stitch images together with rgb on the left
        image = np.concatenate((image, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)), axis=1)
        target_image = image[:, : image.shape[1] // 2, :]
        input_image = image[:, image.shape[1] // 2 :, :]

        augmentations = cfg.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]

        input_image = cfg.transform_only_input(image=input_image)["image"]
        target_image = cfg.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


# TODO: make a factory class instead.
def create_dataset(root_dir: Path | str, dataset_type: cfg.DatasetType) -> Dataset[Any]:
    """Create a Dataset from given root_dir and dataset_type.

    Args:
        root_dir (Path | str): Path for Dataset dir.
        dataset_type (cfg.DatasetType): Type for dataset to create.

    Raises:
        ValueError: If dataset_type is not supported.

    Returns:
        Dataset: Pytorch Dataset object.
    """
    if dataset_type == cfg.DatasetType.ANIME_DATASET:
        return AnimeDataset(root_dir)
    elif dataset_type == cfg.DatasetType.NATURAL_VIEW_DATASET:
        return NaturaViewDataset(root_dir)
    else:
        raise ValueError("Dataset type not supported")
