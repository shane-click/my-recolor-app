import os, io, sys, traceback, requests, replicate
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ─────────── CONFIG ───────────
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")      # r8_…
if not REPLICATE_API_TOKEN:
    raise RuntimeError("Set REPLICATE_API_TOKEN in Render → Environment")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

SEG_MODEL     = "schananas/grounded_sam"    # public Grounded-SAM
INPAINT_MODEL = "sepal/sdxl-inpainting"     # public SDXL in-paint
# ───────────────────────────────

app = Flask(__name__)
CORS(app)

ALLOWED = {"png", "jpg", "jpeg"}
def allowed(fn): return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/")
def index():
    return "POST /recolor (multipart) with fields: image, color"


@app.route("/recolor", methods=["POST"])
def recolor():
    # 1 ▸ validate upload + colour
    if "image" not in request.files:
        return jsonify(error="No image file"), 400
    f = request.files["image"]
    if f.filename == "" or not allowed(f.filename):
        return jsonify(error="Unsupported file type"), 400

    colour = request.form.get("color", "").strip()
    if not colour:
        return jsonify(error="Missing 'color' field"), 400

    # 2 ▸ read once, wrap in stream
    img_bytes  = f.read()
    img_stream = io.BytesIO(img_bytes); img_stream.name = "upload.png"

    # 3 ▸ try to auto-mask the roof
    try:
        mask_urls = replicate.run(
            f"{SEG_MODEL}:latest",
            input={
                "image": img_stream,
                "text_prompt": "house roof",
                "box_threshold": 0.2,
                "text_threshold": 0.2,
                "dilate_kernel_size": 30
            }
        )
    except Exception as e:
        # print full traceback to Render logs
        traceback.print_exc(file=sys.stderr)
        return jsonify(error=f"Segmentation error: {str(e)}"), 500

    if mask_urls:
        mask_url = mask_urls[0]
    else:
        # fallback: recolour entire image
        mask_url = None

    # 4 ▸ SDXL in-paint (mask optional)
    prompt = f"Change only the roof to {colour}. Keep everything else identical."
    img_stream.seek(0)                      # rewind for second upload

    inpaint_input = {
        "prompt": prompt,
        "image": img_stream,
        "num_inference_steps": 30,
        "guidance_scale": 7,
        "strength": 0.4
    }
    if mask_url:
        inpaint_input["mask"] = mask_url

    try:
        result  = replicate.run(f"{INPAINT_MODEL}:latest", input=inpaint_input)
        out_url = result[0]
        png     = requests.get(out_url, timeout=60).content
        return send_file(io.BytesIO(png),
                         mimetype="image/png",
                         download_name="recolored.png")
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return jsonify(error=f"In-paint error: {str(e)}"), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
