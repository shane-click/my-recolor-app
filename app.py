import os
import io
import requests

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ───────────────────────────────
# REQUIRED: set your sk-proj-… key as an env var named OPENAI_API_KEY
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is missing")
# ───────────────────────────────

app = Flask(__name__)
CORS(app)

# Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return "Backend is running. POST to /recolor with 'image' and 'color'."


@app.route("/recolor", methods=["POST"])
def recolor():
    """
    Expects multipart/form-data containing:
      • image : the uploaded file
      • color : the desired roof colour (e.g. 'Dulux Acratex Charcoal')
    Returns:
      • the recoloured image as PNG
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(image_file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    color = request.form.get("color", "")
    if not color:
        return jsonify({"error": "No color provided"}), 400

    # Build prompt
    prompt = (
        f"A detailed, realistic photograph of a house. Only change the roof colour to {color}, "
        "keeping the rest of the house, sky, and surroundings the same. Ultra-realistic photography style."
    )

    try:
        # Direct call to OpenAI Images endpoint (bypasses the 0.28 SDK)
        url = "https://api.openai.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        json_data = {
            "prompt": prompt,
            "n": 1,
            "size": "512x512"
        }

        resp = requests.post(url, headers=headers, json=json_data, timeout=60)
        resp.raise_for_status()
        image_url = resp.json()["data"][0]["url"]

        # Download generated image
        img_data = requests.get(image_url, timeout=60).content

        return send_file(
            io.BytesIO(img_data),
            mimetype="image/png",
            as_attachment=False,
            download_name="recolored.png"
        )

    except requests.HTTPError as err:
        return jsonify({"error": f"OpenAI API error: {err.response.text}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Local testing →  python app.py
    app.run(debug=True, host="0.0.0.0", port=5000)
