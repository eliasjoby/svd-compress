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
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}


def load_grayscale_image(file_storage):
    image = Image.open(BytesIO(file_storage)).convert("L")
    image.thumbnail((1200, 1200))
    return np.array(image)


def compress_image_svd(img_matrix, k):
    u, s, vt = np.linalg.svd(img_matrix, full_matrices=False)
    u_k = u[:, :k]
    s_k = s[:k]
    vt_k = vt[:k, :]
    compressed = (u_k * s_k) @ vt_k
    return compressed, u_k, s_k, vt_k


def reconstruct_from_factors(u_k, s_k, vt_k):
    return (u_k * s_k) @ vt_k


def image_to_base64_url(image_bytes, fmt):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def bytes_to_download_url(file_bytes, mime_type):
    encoded = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_to_encoded_bytes(image, output_format, quality=None):
    fmt = output_format.upper()
    buffer = BytesIO()

    if fmt == "PNG":
        image.save(buffer, format="PNG", optimize=True, compress_level=9)
    elif fmt == "JPEG":
        image.save(buffer, format="JPEG", optimize=True, quality=quality)
    elif fmt == "WEBP":
        image.save(buffer, format="WEBP", quality=quality, method=6)
    else:
        raise ValueError("Unsupported output format.")

    return buffer.getvalue()


def factors_to_npz_bytes(u_k, s_k, vt_k, original_shape, k):
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        u=u_k.astype(np.float32),
        s=s_k.astype(np.float32),
        vt=vt_k.astype(np.float32),
        original_shape=np.array(original_shape, dtype=np.int32),
        k=np.array([k], dtype=np.int32),
        version=np.array([1], dtype=np.int32),
    )
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


def percent_savings(original_bytes, candidate_bytes):
    if original_bytes == 0:
        return 0.0
    return ((original_bytes - candidate_bytes) / original_bytes) * 100.0


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
        "output_format": "JPEG",
        "quality": 75,
    }

    if context["result"]:
        context["k"] = context["result"].get("k", 50)
        context["output_format"] = context["result"].get("output_format", "JPEG")
        context["quality"] = context["result"].get("quality", 75)

    if request.method == "POST":
        image_file = request.files.get("image")
        k = request.form.get("k", "50")
        output_format = request.form.get("output_format", "JPEG").upper()
        quality = request.form.get("quality", "75")

        if not image_file or image_file.filename == "":
            context["error"] = "Please choose an image first."
            return render_template("index.html", **context)

        try:
            k = int(k)
        except ValueError:
            context["error"] = "k must be a number."
            return render_template("index.html", **context)

        try:
            quality = int(quality)
        except ValueError:
            context["error"] = "Quality must be a number."
            return render_template("index.html", **context)

        context["k"] = k
        context["output_format"] = output_format
        context["quality"] = quality

        try:
            file_bytes = image_file.read()
            if not file_bytes:
                context["error"] = "Uploaded file is empty."
                return render_template("index.html", **context)

            original_file_size = len(file_bytes)
            original_matrix = load_grayscale_image(file_bytes)
            m, n = original_matrix.shape

            if output_format not in SUPPORTED_FORMATS:
                context["error"] = "Unsupported output format selected."
                return render_template("index.html", **context)

            if output_format in {"JPEG", "WEBP"} and not (1 <= quality <= 95):
                context["error"] = "Quality must be between 1 and 95 for JPEG/WebP."
                return render_template("index.html", **context)

            if m * n > MAX_PIXELS:
                context["error"] = "Image is too large. Try a smaller image for this demo UI."
                return render_template("index.html", **context)

            max_rank = min(m, n)
            if k < 1 or k > max_rank:
                context["error"] = f"k must be between 1 and {max_rank} for this image."
                return render_template("index.html", **context)

            _, u_k, s_k, vt_k = compress_image_svd(original_matrix, k)

            round_trip_matrix = reconstruct_from_factors(u_k, s_k, vt_k)
            round_trip_matrix = np.clip(round_trip_matrix, 0, 255)

            original_img = Image.fromarray(original_matrix.astype("uint8"))
            compressed_img = Image.fromarray(round_trip_matrix.astype("uint8"))
            original_png = image_to_encoded_bytes(original_img, "PNG")
            compressed_bytes = image_to_encoded_bytes(
                compressed_img,
                output_format,
                quality=quality if output_format in {"JPEG", "WEBP"} else None,
            )
            npz_bytes = factors_to_npz_bytes(u_k, s_k, vt_k, original_matrix.shape, k)

            stats = compression_stats(original_matrix.shape, k)
            compressed_file_size = len(compressed_bytes)
            npz_file_size = len(npz_bytes)

            image_savings_pct = percent_savings(original_file_size, compressed_file_size)
            npz_savings_pct = percent_savings(original_file_size, npz_file_size)

            extension = "jpg" if output_format == "JPEG" else output_format.lower()
            image_mime = "image/jpeg" if output_format == "JPEG" else f"image/{output_format.lower()}"

            result_payload = {
                "original": image_to_base64_url(original_png, "PNG"),
                "compressed": image_to_base64_url(compressed_bytes, output_format),
                "width": n,
                "height": m,
                "max_rank": max_rank,
                "ratio": f"{stats['ratio']:.3f}",
                "percentage": f"{stats['percentage']:.1f}",
                "original_size": human_readable_size(original_file_size),
                "compressed_size": human_readable_size(compressed_file_size),
                "npz_size": human_readable_size(npz_file_size),
                "compressed_savings": f"{image_savings_pct:+.1f}%",
                "npz_savings": f"{npz_savings_pct:+.1f}%",
                "output_format": output_format,
                "quality": quality,
                "download_name": f"compressed_k{k}.{extension}",
                "download_npz_name": f"svd_factors_k{k}.npz",
                "compressed_download": bytes_to_download_url(compressed_bytes, image_mime),
                "npz_download": bytes_to_download_url(npz_bytes, "application/octet-stream"),
                "k": k,
            }

            token = store_result_once(result_payload)
            return redirect(url_for("index", r=token))
        except Exception:
            context["error"] = "Could not process this file. Please upload a valid image."

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
