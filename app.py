# … imports & setup remain the same …

@app.route("/recolor", methods=["POST"])
def recolor():
    if "image" not in request.files:
        return jsonify(error="No image file"), 400
    upload = request.files["image"]
    if upload.filename == "" or not allowed(upload.filename):
        return jsonify(error="Bad filename"), 400

    colour = request.form.get("color", "").strip()
    if not colour:
        return jsonify(error="Missing 'color'"), 400

    # ⭐ read the entire file once into memory
    image_bytes = upload.read()

    # 1️⃣  Grounded-SAM auto-mask
    try:
        mask_urls = replicate.run(
            f"{SEG_MODEL}:latest",
            input={
                "image": image_bytes,        # ⭐ bytes instead of FileStorage
                "mask_prompt": "roof",
                "negative_mask_prompt": "sky",
                "adjustment_factor": -10
            }
        )
        if not mask_urls:
            return jsonify(error="Roof mask not found"), 500
        mask_url = mask_urls[0]
    except Exception as e:
        return jsonify(error=f"Segmentation error: {e}"), 500

    # 2️⃣  SDXL in-paint
    prompt = (f"Replace only the roof with {colour}. "
              "Keep lighting, perspective, and everything else identical. Ultra-realistic photo.")
    neg = "blurry, oversaturated, distorted, extra objects"

    try:
        result = replicate.run(
            f"{INPAINT_MODEL}:latest",
            input={
                "prompt": prompt,
                "negative_prompt": neg,
                "image": image_bytes,      # ⭐ same bytes again
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
