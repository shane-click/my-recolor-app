import os
import io
import sys
import traceback
import requests
import replicate
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── Replicate token ────────────────────────────────────────────
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")        # r8_********
if not REPLICATE_API_TOKEN:
    raise RuntimeError("Set REPLICATE_API_TOKEN in Render → Environment")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
# ───────────────────────────────────────────────────────────────

# PUBLIC T4-tier models (no HPC permissions required)
SEG_MODEL     = "pablodawson/segment-anything-automatic:latest"   # auto–mask (SAM)
INPAINT_MODEL = "sepal/sdxl-inpainting:latest"                    # SDXL in-paint

app = Flask(__name__)
CORS(app)

ALLOWED = {"png", "jpg", "jpeg"}
def allowed(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/")
def home():
    return "POST /recolor (multipart) with fields: image, color"


@app.route("/recolor", methods=["POST"])
def recolor():
    # ── Validate upload & colour ───────────────────────────
    if "image" not in request.files:
        return jsonify(error="No image file"), 400
    f = request.files["image"]
    if f.filename == "" or not allowed(f.filename):
        return jsonify(error="Unsupported file type"), 400

    colour = request.form.get("color", "").strip()
    if not colour:
        return jsonify(error="Missing 'color' field"), 400

    # ── Read once & wrap in BytesIO for both API calls ────
    img_bytes = f.read()
    img_stream = io.BytesIO(img_bytes)
    img_stream.name = "upload.png"

    # ── 1) Automatic segmentation (SAM) ───────────────────
    try:
        masks = replicate.run(
            SEG_MODEL,
            input={
                "image": img_stream,
                # optional tweaks
                "resize_width": 1024,
                "points_per_side": 32
            }
        )
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return jsonify(error=f"Segmentation API error: {str(e)}"), 500

    if masks:
        # choose the largest mask (coarsest object → likely the roof)
        mask_url = max(
            masks,
            key=lambda url: int(
                requests.head(url, timeout=10).headers.get("Content-Length", 0)
            ),
        )
    else:
        mask_url = None  # fall back: recolour whole image

    # ── 2) SDXL in-paint only the masked area ──────────────
    img_stream.seek(0)  # rewind for second upload
    prompt = (
        f"Change only the roof to {colour}. Keep everything else identical. "
        "Ultra-realistic photograph."
    )

    inpaint_input = {
        "prompt": prompt,
        "image": img_stream,
        "num_inference_steps": 30,
        "guidance_scale": 7,
        "strength": 0.4,
    }
    if mask_url:
        inpaint_input["mask"] = mask_url

    try:
        result = replicate.run(INPAINT_MODEL, input=inpaint_input)
        out_png = requests.get(result[0], timeout=60).content
        return send_file(
            io.BytesIO(out_png),
            mimetype="image/png",
            download_name="recolored.png",
        )
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return jsonify(error=f"In-paint API error: {str(e)}"), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
