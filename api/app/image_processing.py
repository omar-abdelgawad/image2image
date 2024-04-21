from PIL import Image
from io import BytesIO


def process_image(image_file):
    # Open the image
    image = Image.open(image_file)

    # Example processing: convert to grayscale
    processed_image = image.convert("RGB")
    processed_image.show()
    # Save processed image to a BytesIO object
    output_buffer = BytesIO()
    processed_image.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    # Return the processed image as bytes
    return output_buffer.getvalue()
