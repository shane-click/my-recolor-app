import os
import io
import sys
import traceback
import requests
import replicate
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── Replicate token (set in Render → Environment) ─────────────
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")      # r8_********
if not REPLICATE_API_TOKEN:
    raise RuntimeError("Set REPLICATE_API_TOKEN in Render → Environment")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
# ──────────────────────────────────────────────────────────────

# Public T4-tier models (no HPC permission required)
SEG_MODEL     = "pablodawson/segment-anything-automatic:latest"  # SAM auto-mask
INPAINT_MODEL = "sepal/sdxl-inpainting:latest"                   # SDXL in-paint

app = Flask(__name__)
CORS(app)

ALLOWED = {"png", "jpg", "jpeg"}
def allowed(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/")
def index():
    return "POST /recolor (multipart) with fields: image, color"


@app.route("/recolor", methods=["POST"])
def recolor():
    # 1 ▸ validate upload & colour
    if "image" not in request.files:
        return jsonify(error="No image file"), 400
    f = request.files["image"]
    if f.filename == "" or not allowed(f.filename):
        return jsonify(error="Unsupported file type"), 400

    colour = request.form.get("color", "").strip()
    if not colour:
        return jsonify(error="Missing 'color' field"), 400

    # 2 ▸ read once, wrap in BytesIO for both API calls
    data = f.read()
    img  = io.BytesIO(data)
    img.name = "upload.png"

    # 3 ▸ automatic segmentation (SAM)
    try:
        masks = replicate.run(
            SEG_MODEL,
            input={
                "image": img,
                "resize_width": 1024,
                "points_per_side": 32
            }
        )
    except Exception as e:
        detail = getattr(e, "detail", str(e))          # <-- show real reason
        traceback.print_exc(file=sys.stderr)
        return jsonify(error=f"Segmentation API error: {detail}"), 500

    # choose largest mask or fall back to full image
    if masks:
        mask_url = max(
            masks,
            key=lambda url: int(
                requests.head(url, timeout=10).headers.get("Content-Length", 0)
            )
        )
    else:
        mask_url = None  # recolour whole image if no mask

    # 4 ▸ SDXL in-paint
    img.seek(0)
    prompt = (
        f"Change only the roof to {colour}. Keep everything else identical. "
        "Ultra-realistic photograph."
    )

    payload = {
        "prompt": prompt,
        "image":  img,
        "num_inference_steps": 30,
        "guidance_scale": 7,
        "strength": 0.4
    }
    if mask_url:
        payload["mask"] = mask_url

    try:
        result = replicate.run(INPAINT_MODEL, input=payload)
        png = requests.get(result[0], timeout=60).content
        return send_file(
            io.BytesIO(png),
            mimetype="image/png",
            download_name="recolored.png"
        )
    except Exception as e:
        detail = getattr(e, "detail", str(e))
        traceback.print_exc(file=sys.stderr)
        return jsonify(error=f"In-paint API error: {detail}"), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
