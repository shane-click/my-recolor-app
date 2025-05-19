import os, io, sys, traceback, requests, replicate
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ─── Replicate token ───────────────────────────────────
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
# ───────────────────────────────────────────────────────

# T4-class models (no HPC permission needed)
SEG_MODEL     = "pablodawson/segment-anything-automatic"   # SAM auto-mask
INPAINT_MODEL = "sepal/sdxl-inpainting"                    # SDXL in-paint

app = Flask(__name__)
CORS(app)

ALLOWED = {"png", "jpg", "jpeg"}
def ok(fn): return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/", methods=["GET"])
def index():
    return "POST /recolor multipart: image, color (string)"


@app.route("/recolor", methods=["POST"])
def recolor():
    # ── validation ────────────────────────────────
    if "image" not in request.files:
        return jsonify(error="No image file"), 400
    up = request.files["image"]
    if up.filename == "" or not ok(up.filename):
        return jsonify(error="Unsupported file"), 400
    colour = request.form.get("color", "").strip()
    if not colour:
        return jsonify(error="Missing 'color'"), 400

    # ── read once, wrap in BytesIO ────────────────
    data = up.read()
    img  = io.BytesIO(data); img.name = "upload.png"

    # ── 1) SAM automatic masks (T4, always allowed) ──
    try:
        masks = replicate.run(
            f"{SEG_MODEL}:latest",
            input={
                "image": img,
                "resize_width": 1024,
                "points_per_side": 32
            }
        )
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return jsonify(error=f"Segmentation API error: {e}"), 500

    if not masks:
        # fall back: recolour whole image
        mask_url = None
    else:
        # use the *largest* mask (coarse roof usually dominates)
        mask_url = max(masks, key=lambda u: int(requests.head(u).headers.get("Content-Length", 0)))

    # ── 2) SDXL in-painting ────────────────────────
    prompt = f"Change only the roof to {colour}. Keep everything else identical, photo-realistic."
    img.seek(0)                           # rewind for second upload

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
        out = replicate.run(f"{INPAINT_MODEL}:latest", input=payload)
        png = requests.get(out[0], timeout=60).content
        return send_file(io.BytesIO(png),
                         mimetype="image/png",
                         download_name="recolored.png")
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return jsonify(error=f"In-paint API error: {e}"), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
