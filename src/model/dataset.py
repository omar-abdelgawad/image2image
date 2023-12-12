"""Dataset Module. Contains Dataset classes that can be constructed using the path.
 Currently Contains AnimeDataset class."""
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

from model import cfg


class AnimeDataset(Dataset):
    def __init__(self, root_dir) -> None:
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


class NaturaViewDataset(Dataset):
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)[: cfg.NUM_IMAGES_DATASET]
        print(f"The length of the dataset is: {len(self.list_files)}")

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
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
def create_dataset(root_dir, dataset_type):
    match dataset_type:
        case cfg.DatasetType.ANIME_DATASET:
            return AnimeDataset(root_dir)
        case cfg.DatasetType.NATURAL_VIEW_DATASET:
            return NaturaViewDataset(root_dir)
        case _:
            raise ValueError("Dataset type not supported")
