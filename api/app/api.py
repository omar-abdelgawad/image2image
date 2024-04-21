from flask import Blueprint, request, jsonify

from .image_processing import process_image

api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/process_image", methods=["POST"])
def process_image_route():
    print("process_image_route")
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files["image"]
    # image_file.save("save_images/saved.png")
    processed_image = process_image(image_file)
    return processed_image
