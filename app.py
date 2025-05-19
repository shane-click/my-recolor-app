import os, io, requests, replicate
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ──────────── CONFIG ────────────
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")   # r8_…
if not REPLICATE_API_TOKEN:
    raise RuntimeError("Add REPLICATE_API_TOKEN in Render → Environment")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

SEG_MODEL     = "okaris/grounded-sam"   # auto-mask
INPAINT_MODEL = "stability-ai/sdxl"     # SDXL in-paint
# ─────────────────────────────────

app = Flask(__name__)
CORS(app)

ALLOWED = {"png", "jpg", "jpeg"}
def allowed(fn): return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/")
def home():
    return "POST /recolor (multipart) with fields: image, color"


@app.route("/recolor", methods=["POST"])
def recolor():
    # 1 ▸ validate inputs
    if "image" not in request.files:
        return jsonify(error="No image file"), 400

    upload = request.files["image"]
    if upload.filename == "" or not allowed(upload.filename):
        return jsonify(error="Unsupported file type"), 400

    colour = request.form.get("color", "").strip()
    if not colour:
        return jsonify(error="Missing 'color' field"), 400

    # 2 ▸ read file once into bytes (works for both API calls)
    image_bytes = upload.read()

    # 3 ▸ auto-mask the roof with Grounded-SAM
    try:
        mask_urls = replicate.run(
            f"{SEG_MODEL}:latest",
            input={
                "image": image_bytes,         # bytes are JSON-serialisable
                "mask_prompt": "roof",
                "negative_mask_prompt": "sky",
                "adjustment_factor": -10      # shrink mask slightly
            }
        )
        if not mask_urls:
            return jsonify(error="Roof mask not found"), 500
        mask_url = mask_urls[0]
    except Exception as e:
        return jsonify(error=f"Segmentation error: {e}"), 500

    # 4 ▸ in-paint just the roof area with SDXL
    prompt = (
        f"Replace only the roof with {colour}. "
        "Keep lighting, perspective and everything else identical. "
        "Ultra-realistic photo."
    )
    neg_prompt = "blurry, oversaturated, distorted"

    try:
        result = replicate.run(
            f"{INPAINT_MODEL}:latest",
            input={
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "image": image_bytes,
                "mask":  mask_url,
                "prompt_strength": 0.35,
                "num_inference_steps": 35,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 1024
            }
        )
        out_url = result[0]
        png = requests.get(out_url, timeout=60).content
        return send_file(
            io.BytesIO(png),
            mimetype="image/png",
            download_name="recolored.png"
        )
    except Exception as e:
        return jsonify(error=f"In-paint error: {e}"), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
