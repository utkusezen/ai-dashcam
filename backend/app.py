from flask import Flask, request, jsonify
from PIL import Image
import io
import models_pipeline

app = Flask(__name__)


@app.route("/", methods=["GET"])
def healthcheck():
    return "API is running", 200


@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    try:
        result = models_pipeline.run(image)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)