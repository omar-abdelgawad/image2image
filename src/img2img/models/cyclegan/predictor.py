from pathlib import Path

from img2img.utils.cyclegan import load_checkpoint
from img2img.models.cyclegan.generator import Generator
from img2img.cfg import cyclegan as cfg
import numpy as np
import torch
from PIL import Image


class CycleGanPredictor:
    def __init__(self, model_path: str | Path):
        self.device = cfg.DEVICE
        self.model = Generator(img_channels=3, num_residuals=9).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)["state_dict"]
        )
        self.model.eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        augmentations = cfg.prediction_transform(image=x)
        input_image = augmentations["image"].to(self.device)
        print(input_image.shape)
        with torch.inference_mode():
            y = self.model(input_image.unsqueeze(0))  # must have a batch dimension
            y = y * 0.5 + 0.5
            y = y.cpu().detach().numpy()
            y = y.squeeze(0) * 255
            y = y.astype(np.uint8)
            assert y.shape == (3, 256, 256)
            y = np.moveaxis(y, 0, -1)
        return y


def test():
    # test yukiyoe2photo model
    model_path = "./out/cyclegan_yukiyoe2photo/last_trained_weights/genh.pth.tar"
    predictor = CycleGanPredictor(model_path)
    image_path = "out/evaluation/cyclegan_predictor_test_image.jpg"
    # take x as an input image in numpy array format where x.shape = (anything, anything, 3)
    x = np.array(Image.open(image_path))
    print(x.shape)
    y = predictor(x)
    image_y = Image.fromarray(y)
    image_y.show()


if __name__ == "__main__":
    test()
