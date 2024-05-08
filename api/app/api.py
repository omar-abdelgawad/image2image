import base64
import io

from flask import Blueprint, jsonify, request

from .image_processing import process_image

api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/coloring", methods=["POST"])
def process_image_route():
    # if "image" not in request.files:
    #     return jsonify({"error": "No image provided"}), 400
    img_byte_array = io.BytesIO()

    base64_image_data = request.json["image"]
    image_data = base64.b64decode(base64_image_data)

    processed_image = process_image(io.BytesIO(image_data))

    # Convert the processed image to bytes

    processed_image.save(img_byte_array, format="JPEG")
    img_byte_array.seek(0)

    # Convert bytes to base64 encoded string
    base64_image = base64.b64encode(img_byte_array.getvalue()).decode("utf-8")

    return jsonify({"processed_image": base64_image})
