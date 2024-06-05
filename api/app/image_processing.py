import numpy as np
from PIL import Image

from img2img.models.pix2pix.predictor import Pix2PixPredictor

from img2img.models.cyclegan.predictor import CycleGanPredictor

# Initialize predictors
print("Initializing predictors...")
anime_predictor = Pix2PixPredictor(
    model_path="./out/saved_models/anime_training/gen.pth.tar"
)
monet_predictor = CycleGanPredictor(
    model_path="./out/saved_models/monet_training/genh.pth.tar"
)
yukiyoe_predictor = CycleGanPredictor(
    model_path="./out/saved_models/yukiyoe_training/genh.pth.tar"
)
vangogh_predictor = CycleGanPredictor(
    model_path="./out/saved_models/vangogh_training/genh.pth.tar"
)


predictors = {
    "anime": anime_predictor,
    "monet": monet_predictor,
    "yukiyoe": yukiyoe_predictor,
    "vangogh": vangogh_predictor,
}


def process_image(image_file, style):
    # Open the image
    image = Image.open(image_file)
    # Example processing: convert to RGB array
    processed_image = np.array(image.convert("RGB"))

    # Ensure the array shape is correct
    assert processed_image.shape[2] == 3

    # Process the image using the appropriate model
    processed_image = predictors[style](processed_image)

    # Convert the processed image array back to PIL Image
    processed_image = Image.fromarray(processed_image)

    processed_image.show()
    # Return the processed image as a PIL Image object
    return processed_image
