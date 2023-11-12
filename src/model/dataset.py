"""Dataset Module. Contains Dataset classes that can be constructed using the path. Currently Contains AnimeDataset class."""
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import cfg


class AnimeDataset(Dataset):
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)[: cfg.NUM_IMAGES_DATASET]
        # print(self.list_files) # maybe print the number of files
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
