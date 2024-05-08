import numpy as np
from PIL import Image

from img2img.models.pix2pix.predictor import Pix2PixPredictor

predictor = Pix2PixPredictor(model_path="./out/saved_models/anime_training/gen.pth.tar")


def process_image(image_file):
    # Open the image
    image = Image.open(image_file)

    # Example processing: convert to RGB array
    processed_image = np.array(image.convert("RGB"))

    # Ensure the array shape is correct
    assert processed_image.shape[2] == 3

    # Process the image using the Pix2Pix model
    processed_image = predictor(processed_image)

    # Convert the processed image array back to PIL Image
    processed_image = Image.fromarray(processed_image)

    # Return the processed image as a PIL Image object
    return processed_image
