from pathlib import Path

import numpy as np
import torch
from PIL import Image

from img2img import cfg
from img2img.models.pix2pix.generator import Generator
from img2img.models.pix2pix.utils import remove_normalization


class Pix2PixPredictor:
    def __init__(self, model_path: str | Path):
        self.device = cfg.DEVICE
        self.model = Generator(in_channels=3).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)["state_dict"]
        )
        self.model.eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        augmentations = cfg.both_transform(image=x)
        input_image = augmentations["image"]
        out_input_image: torch.Tensor = cfg.transform_only_input(image=input_image)[
            "image"
        ]
        out_input_image = out_input_image.to(self.device)
        with torch.inference_mode():
            y = self.model(out_input_image.unsqueeze(0))  # must have a batch dimension
            y = remove_normalization(y)
            y = y.cpu().detach().numpy()
            y = y.squeeze(0) * 255
            y = y.astype(np.uint8)
            assert y.shape == (3, 256, 256)
            y = np.moveaxis(y, 0, -1)
        return y


def test():
    model_path = "./out/saved_models/anime_training/gen.pth.tar"
    predictor = Pix2PixPredictor(model_path)
    image_path = "out/evaluation/pix2pix_predictor_test_image.png"
    # take x as an input image in numpy array format where x.shape = (anything, anything, 3)
    x = np.array(Image.open(image_path))  # returns (429, 488, 4)
    x = x[:, :, :3]  # remove alpha channel
    print(x.shape)
    y = predictor(x)
    image_y = Image.fromarray(y)
    return image_y


if __name__ == "__main__":
    image = test()
