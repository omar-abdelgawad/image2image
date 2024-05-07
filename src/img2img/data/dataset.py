"""Dataset Module. Contains Dataset classes that can be constructed using the path."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from img2img import cfg


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

        out_input_image: torch.Tensor = cfg.transform_only_input(image=input_image)[
            "image"
        ]
        out_target_image: torch.Tensor = cfg.transform_only_mask(image=target_image)[
            "image"
        ]

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

        target_image = cv2.imread(img_path)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)

        augmentations = cfg.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]

        out_input_image: torch.Tensor = cfg.transform_only_input(image=input_image)[
            "image"
        ]
        out_target_image: torch.Tensor = cfg.transform_only_mask(image=target_image)[
            "image"
        ]

        return out_input_image, out_target_image


class Edges2Shoes(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset class for edges2shoes dataset.

    Args:
        root_dir (root_dir): Path for dataset dir.
    """

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)[: cfg.NUM_IMAGES_DATASET]
        print(f"The length of the dataset is: {len(self.list_files)}")

    def __len__(self) -> int:
        return len(self.list_files)

    def __getitem__(self, index: int):
        img_file_name = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        input_image = image[:, : image.shape[1] // 2, :]
        target_image = image[:, image.shape[1] // 2 :, :]

        augmentations = cfg.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]

        input_image = cfg.transform_only_input(image=input_image)["image"]
        target_image = cfg.transform_only_mask(image=target_image)["image"]

        assert input_image.shape == target_image.shape
        assert input_image.shape[0] == 3
        return input_image, target_image


class AFHQCatDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
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
    elif dataset_type == cfg.DatasetType.EDGES2SHOES:
        return Edges2Shoes(root_dir)
    elif dataset_type == cfg.DatasetType.AFHQ_CATS_DATASET:
        return AFHQCatDataset(root_dir)
    else:
        raise ValueError("Dataset type not supported")


def get_loader(
    root_dir: Path | str,
    dataset_type: cfg.DatasetType,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[Any]:
    """Create a Dataloader from given root_dir and dataset_type.

    Args:
        root_dir (Path | str): Path for Dataset dir.
        dataset_type (cfg.DatasetType): Type for dataset to create.

    Raises:
        ValueError: If dataset_type is not supported.

    Returns:
        Dataset: Pytorch Dataset object.
    """
    dataset = create_dataset(root_dir, dataset_type)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def test():
    # TODO: add these tests to pytest and remove this function.
    """Test function for dataset module."""
    # show image using torchvision package
    # import makegrid
    from torchvision.utils import make_grid, save_image

    dataset = create_dataset(cfg.TRAIN_DATASET_PATH, cfg.DatasetType.EDGES2SHOES)
    input_image, target_image = dataset[0]
    input_image = (input_image) * 0.5 + 0.5
    target_image = (target_image) * 0.5 + 0.5
    print(input_image.requires_grad, target_image.requires_grad)
    print(input_image.shape)
    print(target_image.shape)
    print(type(input_image))
    print(type(target_image))
    print(input_image.max())
    print(input_image.min())
    print(target_image.max())
    print(target_image.min())
    image_to_save = make_grid([input_image, target_image])
    save_image(image_to_save, "test.png")


if __name__ == "__main__":
    test()
