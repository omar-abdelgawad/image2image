from flask import Blueprint, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
from PIL import Image
import os
from .image_processing import process_image

api_blueprint = Blueprint("api", __name__)


CORS(api_blueprint)


@api_blueprint.route("/coloring", methods=["POST"])
def process_image_route():
    print("process_image_route")
    print(request.get_json(silent=True))
    # if "image" not in request.files:
    #     return jsonify({"error": "No image provided"}), 400
    image_file = request.get_json(silent=True)["path"]
    processed_image = process_image(image_file)
    os.makedirs("save_images", exist_ok=True)
    processed_image.save(
        "/home/eyad/watashi-ubuntu/academics/seminar/pbl/pbl-front/public/img.png"
    )
    # get absolute path of the saved image
    abs_path = (
        "/home/eyad/watashi-ubuntu/academics/seminar/pbl/pbl-front/public/img.png"
    )
    # return processed_image
    return jsonify({"path": abs_path})
