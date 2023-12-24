"""Dataset Module. Contains Dataset classes that can be constructed using the path.
 Currently Contains edges2shoes class."""
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from unit import cfg


class Edges2Shoes(Dataset):
    """Dataset class for edges2shoes dataset.

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
