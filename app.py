import base64
import uuid
from io import BytesIO

import numpy as np
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")


MAX_PIXELS = 1200 * 1200
RESULT_CACHE = {}
MAX_CACHED_RESULTS = 10


def load_grayscale_image(file_storage):
    image = Image.open(BytesIO(file_storage)).convert("L")
    image.thumbnail((1200, 1200))
    return np.array(image)


def compress_image_svd(img_matrix, k):
    u, s, vt = np.linalg.svd(img_matrix, full_matrices=False)
    u_k = u[:, :k]
    s_k = np.diag(s[:k])
    vt_k = vt[:k, :]
    return np.dot(u_k, np.dot(s_k, vt_k))


def image_to_base64_url(image_bytes, fmt):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def image_to_png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def human_readable_size(byte_count):
    if byte_count < 1024:
        return f"{byte_count} B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f} KB"
    return f"{byte_count / (1024 * 1024):.2f} MB"


def compression_stats(img_shape, k):
    m, n = img_shape
    uncompressed_size = m * n
    compressed_size = k * (1 + m + n)
    ratio = compressed_size / uncompressed_size
    return {
        "ratio": ratio,
        "percentage": ratio * 100,
    }


def store_result_once(result_payload):
    token = uuid.uuid4().hex
    RESULT_CACHE[token] = result_payload

    if len(RESULT_CACHE) > MAX_CACHED_RESULTS:
        oldest_token = next(iter(RESULT_CACHE))
        RESULT_CACHE.pop(oldest_token, None)

    return token


@app.route("/", methods=["GET", "POST"])
def index():
    result_token = request.args.get("r")
    context = {
        "result": RESULT_CACHE.pop(result_token, None) if result_token else None,
        "error": None,
        "k": 50,
    }

    if context["result"]:
        context["k"] = context["result"].get("k", 50)

    if request.method == "POST":
        image_file = request.files.get("image")
        k = request.form.get("k", "50")

        if not image_file or image_file.filename == "":
            context["error"] = "Please choose an image first."
            return render_template("index.html", **context)

        try:
            k = int(k)
        except ValueError:
            context["error"] = "k must be a number."
            return render_template("index.html", **context)

        context["k"] = k

        try:
            file_bytes = image_file.read()
            if not file_bytes:
                context["error"] = "Uploaded file is empty."
                return render_template("index.html", **context)

            original_file_size = len(file_bytes)
            original_matrix = load_grayscale_image(file_bytes)
            m, n = original_matrix.shape

            if m * n > MAX_PIXELS:
                context["error"] = "Image is too large. Try a smaller image for this demo UI."
                return render_template("index.html", **context)

            max_rank = min(m, n)
            if k < 1 or k > max_rank:
                context["error"] = f"k must be between 1 and {max_rank} for this image."
                return render_template("index.html", **context)

            compressed_matrix = compress_image_svd(original_matrix, k)
            compressed_matrix = np.clip(compressed_matrix, 0, 255)

            original_img = Image.fromarray(original_matrix.astype("uint8"))
            compressed_img = Image.fromarray(compressed_matrix.astype("uint8"))
            original_png = image_to_png_bytes(original_img)
            compressed_png = image_to_png_bytes(compressed_img)

            stats = compression_stats(original_matrix.shape, k)
            compressed_file_size = len(compressed_png)

            result_payload = {
                "original": image_to_base64_url(original_png, "PNG"),
                "compressed": image_to_base64_url(compressed_png, "PNG"),
                "width": n,
                "height": m,
                "max_rank": max_rank,
                "ratio": f"{stats['ratio']:.3f}",
                "percentage": f"{stats['percentage']:.1f}",
                "original_size": human_readable_size(original_file_size),
                "compressed_size": human_readable_size(compressed_file_size),
                "download_name": f"compressed_k{k}.png",
                "k": k,
            }

            token = store_result_once(result_payload)
            return redirect(url_for("index", r=token))
        except Exception:
            context["error"] = "Could not process this file. Please upload a valid image."

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
