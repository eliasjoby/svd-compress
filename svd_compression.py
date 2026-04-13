import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from io import BytesIO


def load_rgb_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def compress_channel_svd(img_matrix, k):
    # compute svd
    u, s, vt = np.linalg.svd(img_matrix, full_matrices=False)
    u_k = u[:, :k]
    s_k = s[:k]
    vt_k = vt[:k, :]

    compressed = (u_k * s_k) @ vt_k
    return compressed, u_k, s_k, vt_k


def compress_image_svd(img_matrix, k):
    channel_factors = []
    reconstructed = np.zeros_like(img_matrix, dtype=np.float32)

    for channel_idx in range(img_matrix.shape[2]):
        channel_matrix = img_matrix[:, :, channel_idx]
        compressed_channel, u_k, s_k, vt_k = compress_channel_svd(channel_matrix, k)
        reconstructed[:, :, channel_idx] = compressed_channel
        channel_factors.append((u_k, s_k, vt_k))

    return reconstructed, channel_factors


def encode_image_bytes(image, fmt="JPEG", quality=75):
    fmt = fmt.upper()
    buffer = BytesIO()

    if fmt == "PNG":
        image.save(buffer, format="PNG", optimize=True, compress_level=9)
    elif fmt == "WEBP":
        image.save(buffer, format="WEBP", quality=quality, method=6)
    else:
        image.save(buffer, format="JPEG", quality=quality, optimize=True)

    return buffer.getvalue()


def save_svd_npz(path, channel_factors, img_shape, k):
    (u_r, s_r, vt_r), (u_g, s_g, vt_g), (u_b, s_b, vt_b) = channel_factors
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
        path,
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

def plot_results(original, compressed, k):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(compressed.astype("uint8"))
    axes[1].set_title(f"Compressed (k={k})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def compression_stats(img_shape, k):
    m, n, channels = img_shape
    uncompressed_size = m * n * channels
    compressed_size = channels * k * (1 + m + n)  # per-channel factor storage
    ratio = compressed_size / uncompressed_size
    print(f"Compression ratio: {ratio:.2f} (compressed is {ratio*100:.1f}% of original)")


def print_file_size_report(original_path, output_paths):
    original_size = os.path.getsize(original_path)
    print("\nDisk size report")
    print(f"Original file: {original_path} -> {original_size / 1024:.1f} KB")

    for path in output_paths:
        size = os.path.getsize(path)
        ratio = size / original_size
        print(f"{path}: {size / 1024:.1f} KB ({ratio * 100:.1f}% of original)")


def parse_args():
    parser = argparse.ArgumentParser(description="SVD-based image compression demo")
    parser.add_argument("--input", default="input.jpg", help="Path to input image")
    parser.add_argument("--k", type=int, default=50, help="Rank for SVD reconstruction")
    parser.add_argument(
        "--format",
        choices=["JPEG", "PNG", "WEBP"],
        default="JPEG",
        help="Output image format",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=75,
        help="Quality for JPEG/WEBP (30-95)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable side-by-side matplotlib preview",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    image_path = args.input
    img_matrix = load_rgb_image(image_path)
    print("Image loaded with shape:", img_matrix.shape)

    k = args.k
    max_rank = min(img_matrix.shape[0], img_matrix.shape[1])
    if k < 1 or k > max_rank:
        raise ValueError(f"k must be between 1 and {max_rank}")

    quality = max(30, min(95, args.quality))

    compressed_img, channel_factors = compress_image_svd(img_matrix, k)
    compressed_img = np.clip(compressed_img, 0, 255)

    if not args.no_plot:
        plot_results(img_matrix, compressed_img, k)

    compression_stats(img_matrix.shape, k)

    compressed_img = Image.fromarray(compressed_img.astype("uint8"), mode="RGB")
    output_dir = os.path.join("examples", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    extension = "jpg" if args.format == "JPEG" else args.format.lower()
    output_image = os.path.join(output_dir, f"compressed_k{k}.{extension}")
    output_npz = os.path.join(output_dir, f"compressed_factors_k{k}.npz")

    image_bytes = encode_image_bytes(compressed_img, fmt=args.format, quality=quality)
    with open(output_image, "wb") as image_file:
        image_file.write(image_bytes)

    save_svd_npz(output_npz, channel_factors, img_matrix.shape, k)

    print(f"Compressed image saved as {output_image}")
    print(f"SVD factors saved as {output_npz}")
    print_file_size_report(image_path, [output_image, output_npz])

if __name__ == "__main__":
    main()