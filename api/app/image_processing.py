from PIL import Image
from io import BytesIO
from img2img.models.pix2pix.predictor import Pix2PixPredictor
import numpy as np

predictor = Pix2PixPredictor(
    model_path="/home/omarabdelgawad/my_workspace/projects/github_repos/image2image/out/saved_models/anime_training/gen.pth.tar"
)


def process_image(image_file):
    # Open the image
    image = Image.open(image_file)

    # Example processing: convert to grayscale
    processed_image = image.convert("RGB")
    processed_image = np.array(processed_image)
    assert processed_image.shape[2] == 3
    processed_image = predictor(processed_image)
    processed_image = Image.fromarray(processed_image)
    processed_image.show()
    # Save processed image to a BytesIO object
    output_buffer = BytesIO()
    processed_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    # Return the processed image as bytes
    return output_buffer.getvalue()
