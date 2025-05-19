import os
import io
import requests

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── REQUIRED ────────────────────────────────────────────────
API_KEY     = os.getenv("OPENAI_API_KEY")      # sk-proj-xxxxxxxx…
PROJECT_ID  = "proj_b6A7WmLHkhfYIzLr9bLCbH9z"  # ← your project ID

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is missing")
# ────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return "Backend is running. POST to /recolor with 'image' and 'color'."


@app.route("/recolor", methods=["POST"])
def recolor():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "File type not allowed"}), 400

    color = request.form.get("color", "")
    if not color:
        return jsonify({"error": "No color provided"}), 400

    prompt = (
        f"A detailed, realistic photograph of a house. Only change the roof colour to {color}, "
        "keeping the rest of the house, sky, and surroundings the same. Ultra-realistic photography style."
    )

    try:
        # ── Raw HTTPS call to OpenAI Images endpoint ───────────
        url = "https://api.openai.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "OpenAI-Project": PROJECT_ID,        # ← project header
        }
        payload = {"prompt": prompt, "n": 1, "size": "512x512"}

        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        image_url = r.json()["data"][0]["url"]
        # ───────────────────────────────────────────────────────

        img_data = requests.get(image_url, timeout=60).content
        return send_file(
            io.BytesIO(img_data),
            mimetype="image/png",
            as_attachment=False,
            download_name="recolored.png",
        )

    except requests.HTTPError as err:
        return jsonify({"error": err.response.text}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
