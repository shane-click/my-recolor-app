import os, io, requests, replicate
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── Replicate token ─────────────────────────
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
# ────────────────────────────────────────────

SEG_MODEL     = "brigade/grounded-sam"          # openly accessible
INPAINT_MODEL = "fofr/controlnet-recolor-sdxl"  # public SDXL recolor

app = Flask(__name__)
CORS(app)

ALLOWED = {"png", "jpg", "jpeg"}
def allowed(fn): return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/recolor", methods=["POST"])
def recolor():
    if "image" not in request.files:
        return jsonify(error="No image file"), 400

    upload = request.files["image"]
    if upload.filename == "" or not allowed(upload.filename):
        return jsonify(error="Unsupported file"), 400

    colour = request.form.get("color", "").strip()
    if not colour:
        return jsonify(error="Missing 'color'"), 400

    # read once, wrap in stream
    img_bytes = upload.read()
    img_stream = io.BytesIO(img_bytes)
    img_stream.name = "upload.png"

    # Grounded-SAM (public)
    try:
        mask_urls = replicate.run(
            f"{SEG_MODEL}:latest",
            input={
                "image": img_stream,
                "text_prompt": "roof",
                "box_threshold": 0.25,
                "text_threshold": 0.25
            }
        )
        if not mask_urls:
            return jsonify(error="Roof mask not found"), 500
        mask_url = mask_urls[0]
    except Exception as e:
        return jsonify(error=f"Segmentation error: {e}"), 500

    # reset stream for reuse
    img_stream.seek(0)

    # SDXL ControlNet Recolor (public)
    prompt = f"Change only the roof to {colour}. Keep everything else identical."
    try:
        result = replicate.run(
            f"{INPAINT_MODEL}:latest",
            input={
                "image": img_stream,
                "mask":  mask_url,
                "prompt": prompt,
                "num_steps": 30,
                "guidance_scale": 7,
                "strength": 0.4,
                "seed": 0
            }
        )
        out_url = result[0]
        out_png = requests.get(out_url, timeout=60).content
        return send_file(io.BytesIO(out_png),
                         mimetype="image/png",
                         download_name="recolored.png")
    except Exception as e:
        return jsonify(error=f"In-paint error: {e}"), 500


@app.route("/")
def home():
    return "POST /recolor with image + color"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
