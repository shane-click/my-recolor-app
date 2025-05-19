import os
import io
import requests
import openai


from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

api_key = os.getenv("OPENAI_API_KEY")          #  sk-proj-xxxxxxxx…

# 2 ▸ if it’s a project key, monkey-patch the validator
if api_key and api_key.startswith("sk-proj-"):
    import openai.api_requestor as _ar
    import openai.error as _err

    def _accept_proj_keys(key: str):
        """Allow sk-, sk-proj-, sess-  (reject None / wrong type)."""
        if key is None or not isinstance(key, str):
            raise _err.AuthenticationError(
                "No API key provided. Create one at https://platform.openai.com/account/api-keys"
            )
        # any string that *starts with* the approved prefixes is ok
        if not key.startswith(("sk-", "sk-proj-", "sess-")):
            raise _err.AuthenticationError("Malformed API key")

    _ar._validate_api_key = _accept_proj_keys      # ← HOT PATCH
# ──────────────────────────────────────────────────────────────

openai.api_key = api_key            
    
app = Flask(__name__)
CORS(app)

# Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return "Backend is running. POST to /recolor with 'image' and 'color'."

@app.route("/recolor", methods=["POST"])
def recolor():
    """
    Expects:
      - `image`: the uploaded file
      - `color`: the color string (e.g. 'Dulux Acratex Charcoal')
    Returns:
      - A new image (PNG) with the roof color changed
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

    # Read the uploaded image in bytes (not strictly used in 'create' method, but let's read it anyway)
    uploaded_image_bytes = image_file.read()

    # Build a prompt that tries to preserve the original image except for the roof.
    # We describe the scene so that only the roof is recolored, everything else remains the same.
    prompt = (
        f"A detailed, realistic photograph of a house. Only change the roof color to {color}, "
        "keeping the rest of the house, sky, and surroundings the same. Ultra-realistic photography style."
    )

    try:
        # Call the OpenAI Image Create endpoint with the prompt
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"  # or "1024x1024", etc.
        )
        # Extract the returned image URL
        image_url = response["data"][0]["url"]

        # Download the generated image
        img_data = requests.get(image_url).content

        # Return it to the client as a PNG
        return send_file(
            io.BytesIO(img_data),
            mimetype="image/png",
            as_attachment=False,
            download_name="recolored.png"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local testing, run:  python app.py
    app.run(debug=True, host="0.0.0.0", port=5000)
