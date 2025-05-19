import os
import io
import sys
import traceback
import requests
import replicate
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ──────────────────────────────────────────────────────────────
#  Replicate API token (add in Render → Environment as r8_…)
# ──────────────────────────────────────────────────────────────
token = os.getenv("REPLICATE_API_TOKEN")
if not token:
    raise RuntimeError("Set REPLICATE_API_TOKEN in Render → Environment")
os.environ["REPLICATE_API_TOKEN"] = token

# Public T4-tier models (no HPC permission needed)
SEG_MODEL     = "pablodawson/segment-anything-automatic:latest"
INPAINT_MODEL = "sepal/sdxl-inpainting:latest"

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
    # ── validate upload + colour ─────────────────────────────
    if "image" not in request.files:
        return jsonify(error="No image file"), 400
    f = request.files["image"]
    if f.filename == "" or not allowed(f.filename):
        return jsonify(error="Unsupported file type"), 400

    colour = request.form.get("color", "").strip()
    if not colour:
        return jsonify(error="Missing 'color' field"), 400

    # ── read once, wrap in BytesIO ───────────────────────────
    img_bytes = f.read()
    img_stream = io.BytesIO(img_bytes)
    img_stream.name = "upload.png"

    # ── 1️⃣  Automatic mask with SAM ─────────────────────────
    try:
        masks = replicate.run(
            SEG_MODEL,
            input={
                "image": img_stream,
                "resize_width": 1024,
                "points_per_side": 32
            }
        )
    except Exception as e:
        # --- DEBUG: print everything to Render logs ----------
        traceback.print_exc(file=sys.stderr)
        print("── Replicate error attrs ──", file=sys.stderr)
        for attr in dir(e):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(e, attr)
                print(f"{attr}: {val!r}", file=sys.stderr)
            except Exception:
                pass
        print("── end attrs ──", file=sys.stderr)
        # ------------------------------------------------------
        msg = getattr(e, "detail", None) or getattr(e, "message", None) \
              or (e.args[0] if e.args else str(e))
        return jsonify(error=f"Segmentation API error: {msg}"), 500

    # choose largest mask (roof usually biggest) or fallback
    if masks:
        mask_url = max(
            masks,
            key=lambda url: int(
                requests.head(url, timeout=10).headers.get("Content-Length", 0)
            )
        )
    else:
        mask_url = None  # recolour whole frame

    # ── 2️⃣  SDXL in-paint ───────────────────────────────────
    img_stream.seek(0)
    prompt = (
        f"Change only the roof to {colour}. Keep everything else identical. "
        "Ultra-realistic photograph."
    )

    payload = {
        "prompt": prompt,
        "image":  img_stream,
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
        traceback.print_exc(file=sys.stderr)
        err = getattr(e, "detail", None) or getattr(e, "message", None) \
              or (e.args[0] if e.args else str(e))
        return jsonify(error=f"In-paint API error: {err}"), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
