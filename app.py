import base64
import uuid
from io import BytesIO

import numpy as np
from flask import Flask, Response, redirect, render_template, request, url_for
from PIL import Image

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")


MAX_PIXELS = 1200 * 1200
RESULT_CACHE = {}
MAX_CACHED_RESULTS = 10


def load_rgb_image(file_storage):
    image = Image.open(BytesIO(file_storage)).convert("RGB")
    image.thumbnail((1200, 1200))
    return np.array(image)


def compress_channel_svd(img_matrix, k):
    u, s, vt = np.linalg.svd(img_matrix, full_matrices=False)
    u_k = u[:, :k]
    s_k = s[:k]
    vt_k = vt[:k, :]
    return (u_k * s_k) @ vt_k, u_k, s_k, vt_k


def compress_image_svd(img_matrix, k):
    channel_factors = []
    reconstructed = np.zeros_like(img_matrix, dtype=np.float32)

    for channel_idx in range(img_matrix.shape[2]):
        channel = img_matrix[:, :, channel_idx]
        compressed_channel, u_k, s_k, vt_k = compress_channel_svd(channel, k)
        reconstructed[:, :, channel_idx] = compressed_channel
        channel_factors.append((u_k, s_k, vt_k))

    return reconstructed, channel_factors


def image_to_base64_url(image_bytes, fmt):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def encode_image_bytes(image, fmt, quality=85):
    fmt = fmt.upper()
    buffer = BytesIO()

    if fmt == "PNG":
        image.save(buffer, format="PNG", optimize=True, compress_level=9)
    elif fmt == "JPEG":
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    elif fmt == "WEBP":
        image.save(buffer, format="WEBP", quality=quality, method=6)
    else:
        raise ValueError("Unsupported format.")

    return buffer.getvalue()


def build_svd_npz_bytes(channel_factors, img_shape, k):
    (u_r, s_r, vt_r), (u_g, s_g, vt_g), (u_b, s_b, vt_b) = channel_factors
    buffer = BytesIO()
    metadata = np.array(
        [
            f"shape={img_shape[0]}x{img_shape[1]}",
            f"k={k}",
            "mode=RGB",
            "channels=3",
            "version=1",
        ],
        dtype="U32",
    )
    np.savez_compressed(
        buffer,
        u_r=u_r,
        s_r=s_r,
        vt_r=vt_r,
        u_g=u_g,
        s_g=s_g,
        vt_g=vt_g,
        u_b=u_b,
        s_b=s_b,
        vt_b=vt_b,
        metadata=metadata,
    )
    return buffer.getvalue()


def human_readable_size(byte_count):
    if byte_count < 1024:
        return f"{byte_count} B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f} KB"
    return f"{byte_count / (1024 * 1024):.2f} MB"


def compression_stats(img_shape, k):
    m, n, channels = img_shape
    uncompressed_size = m * n * channels
    compressed_size = channels * k * (1 + m + n)
    ratio = compressed_size / uncompressed_size
    return {
        "ratio": ratio,
        "percentage": ratio * 100,
    }


def file_size_stats(original_size, generated_size):
    reduction_bytes = original_size - generated_size
    reduction_pct = (reduction_bytes / original_size) * 100 if original_size else 0.0
    return {
        "reduction_bytes": reduction_bytes,
        "reduction_pct": reduction_pct,
    }


def detect_mime(fmt):
    return {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
        "NPZ": "application/octet-stream",
    }.get(fmt, "application/octet-stream")


def store_result_once(result_payload):
    token = uuid.uuid4().hex
    cache_set(token, result_payload)

    return token


def cache_set(key, value):
    RESULT_CACHE[key] = value
    while len(RESULT_CACHE) > MAX_CACHED_RESULTS:
        oldest_token = next(iter(RESULT_CACHE))
        RESULT_CACHE.pop(oldest_token, None)


@app.route("/", methods=["GET", "POST"])
def index():
    result_token = request.args.get("r")
    context = {
        "result": RESULT_CACHE.pop(result_token, None) if result_token else None,
        "error": None,
        "k": 50,
        "format": "JPEG",
        "quality": 75,
    }

    if context["result"]:
        context["k"] = context["result"].get("k", 50)
        context["format"] = context["result"].get("format", "JPEG")
        context["quality"] = context["result"].get("quality", 75)

    if request.method == "POST":
        image_file = request.files.get("image")
        k = request.form.get("k", "50")
        output_format = request.form.get("format", "JPEG").upper()
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

        if output_format not in {"PNG", "JPEG", "WEBP"}:
            context["error"] = "Output format must be PNG, JPEG, or WEBP."
            return render_template("index.html", **context)

        quality = max(30, min(95, quality))

        context["k"] = k
        context["format"] = output_format
        context["quality"] = quality

        try:
            file_bytes = image_file.read()
            if not file_bytes:
                context["error"] = "Uploaded file is empty."
                return render_template("index.html", **context)

            original_file_size = len(file_bytes)
            original_matrix = load_rgb_image(file_bytes)
            m, n = original_matrix.shape[:2]

            if m * n > MAX_PIXELS:
                context["error"] = "Image is too large. Try a smaller image for this demo UI."
                return render_template("index.html", **context)

            max_rank = min(m, n)
            if k < 1 or k > max_rank:
                context["error"] = f"k must be between 1 and {max_rank} for this image."
                return render_template("index.html", **context)

            compressed_matrix, channel_factors = compress_image_svd(original_matrix, k)
            compressed_matrix = np.clip(compressed_matrix, 0, 255)

            original_img = Image.fromarray(original_matrix.astype("uint8"), mode="RGB")
            compressed_img = Image.fromarray(compressed_matrix.astype("uint8"), mode="RGB")
            original_png = encode_image_bytes(original_img, "PNG")
            compressed_bytes = encode_image_bytes(compressed_img, output_format, quality)
            svd_npz_bytes = build_svd_npz_bytes(channel_factors, original_matrix.shape, k)

            stats = compression_stats(original_matrix.shape, k)
            compressed_file_size = len(compressed_bytes)
            npz_file_size = len(svd_npz_bytes)
            image_file_stats = file_size_stats(original_file_size, compressed_file_size)
            npz_file_stats = file_size_stats(original_file_size, npz_file_size)

            image_token = uuid.uuid4().hex
            npz_token = uuid.uuid4().hex
            cache_set(f"image:{image_token}", {
                "bytes": compressed_bytes,
                "mime": detect_mime(output_format),
                "name": f"compressed_k{k}.{output_format.lower()}",
            })
            cache_set(f"npz:{npz_token}", {
                "bytes": svd_npz_bytes,
                "mime": detect_mime("NPZ"),
                "name": f"compressed_factors_k{k}.npz",
            })

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
                "compressed_delta": f"{image_file_stats['reduction_pct']:.1f}%",
                "npz_size": human_readable_size(npz_file_size),
                "npz_delta": f"{npz_file_stats['reduction_pct']:.1f}%",
                "download_name": f"compressed_k{k}.{output_format.lower()}",
                "npz_name": f"compressed_factors_k{k}.npz",
                "download_image_url": url_for("download_result", kind="image", token=image_token),
                "download_npz_url": url_for("download_result", kind="npz", token=npz_token),
                "format": output_format,
                "quality": quality,
                "k": k,
            }

            token = store_result_once(result_payload)
            return redirect(url_for("index", r=token))
        except Exception:
            context["error"] = "Could not process this file. Please upload a valid image."

    return render_template("index.html", **context)


@app.route("/download/<kind>/<token>")
def download_result(kind, token):
    cache_key = f"{kind}:{token}"
    payload = RESULT_CACHE.get(cache_key)
    if not payload:
        return Response("Download has expired. Please recompress the image.", status=404)

    return Response(
        payload["bytes"],
        mimetype=payload["mime"],
        headers={"Content-Disposition": f"attachment; filename={payload['name']}"},
    )


if __name__ == "__main__":
    app.run(debug=True)
