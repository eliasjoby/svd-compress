import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_rgb_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def compress_image_svd(img_matrix, k):
    # compute svd
    U, S, VT = np.linalg.svd(img_matrix, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    VT_k = VT[:k, :]

    compressed = (U_k * S_k) @ VT_k
    return compressed, U_k, S_k, VT_k


def compress_color_image_svd(img_array, k):
    channel_factors = []
    reconstructed = np.zeros_like(img_array, dtype=np.float32)

    for channel_idx in range(img_array.shape[2]):
        channel_matrix = img_array[:, :, channel_idx]
        channel_compressed, U_k, S_k, VT_k = compress_image_svd(channel_matrix, k)
        reconstructed[:, :, channel_idx] = channel_compressed
        channel_factors.append((U_k, S_k, VT_k))

    return reconstructed, channel_factors


def save_svd_npz(path, channel_factors, original_shape, k):
    (u_r, s_r, vt_r), (u_g, s_g, vt_g), (u_b, s_b, vt_b) = channel_factors

    np.savez_compressed(
        path,
        u_r=u_r.astype(np.float32),
        s_r=s_r.astype(np.float32),
        vt_r=vt_r.astype(np.float32),
        u_g=u_g.astype(np.float32),
        s_g=s_g.astype(np.float32),
        vt_g=vt_g.astype(np.float32),
        u_b=u_b.astype(np.float32),
        s_b=s_b.astype(np.float32),
        vt_b=vt_b.astype(np.float32),
        original_shape=np.array(original_shape, dtype=np.int32),
        k=np.array([k], dtype=np.int32),
        channels=np.array([len(channel_factors)], dtype=np.int32),
        version=np.array([1], dtype=np.int32),
    )

def plot_results(original, compressed, k):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(compressed)
    axes[1].set_title(f"Compressed (k={k})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def compression_stats(img_shape, k):
    m, n = img_shape[:2]
    channels = img_shape[2] if len(img_shape) == 3 else 1
    uncompressed_size = m * n * channels
    compressed_size = channels * k * (1 + m + n)  # per-channel SVD factor storage
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

def main():
    image_path = "input.jpg"
    img_array = load_rgb_image(image_path)
    print("Image loaded with shape:", img_array.shape)

    k = 50
    compressed_img, channel_factors = compress_color_image_svd(img_array, k)
    compressed_img = np.clip(compressed_img, 0, 255)

    plot_results(img_array, compressed_img.astype("uint8"), k)
    compression_stats(img_array.shape, k)

    # Save in multiple formats for fair file-size comparisons.
    compressed_img = Image.fromarray(compressed_img.astype("uint8"), mode="RGB")
    output_dir = os.path.join("examples", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_png = os.path.join(output_dir, f"compressed_k{k}.png")
    output_jpg = os.path.join(output_dir, f"compressed_k{k}.jpg")
    output_webp = os.path.join(output_dir, f"compressed_k{k}.webp")
    output_npz = os.path.join(output_dir, f"svd_factors_k{k}.npz")

    compressed_img.save(output_png, optimize=True, compress_level=9)
    compressed_img.save(output_jpg, format="JPEG", quality=85, optimize=True)
    compressed_img.save(output_webp, format="WEBP", quality=80, method=6)
    save_svd_npz(output_npz, channel_factors, img_array.shape, k)

    print(f"Compressed image saved as {output_png}")
    print(f"Compressed image saved as {output_jpg}")
    print(f"Compressed image saved as {output_webp}")
    print(f"SVD factors saved as {output_npz}")
    print_file_size_report(image_path, [output_png, output_jpg, output_webp, output_npz])

if __name__ == "__main__":
    main()