import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_grayscale_image(path):
    img = Image.open(path).convert('L')  
    return np.array(img)

def compress_image_svd(img_matrix, k):
    # compute svd
    U, S, VT = np.linalg.svd(img_matrix, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    VT_k = VT[:k, :]

    compressed = (U_k * S_k) @ VT_k
    return compressed, U_k, S_k, VT_k


def save_svd_npz(path, u_k, s_k, vt_k, original_shape, k):
    np.savez_compressed(
        path,
        u=u_k.astype(np.float32),
        s=s_k.astype(np.float32),
        vt=vt_k.astype(np.float32),
        original_shape=np.array(original_shape, dtype=np.int32),
        k=np.array([k], dtype=np.int32),
        version=np.array([1], dtype=np.int32),
    )

def plot_results(original, compressed, k):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(compressed, cmap='gray')
    axes[1].set_title(f"Compressed (k={k})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def compression_stats(img_shape, k):
    m, n = img_shape
    uncompressed_size = m * n
    compressed_size = k * (1 + m + n)  # k singular values + U_k + VT_k
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
    image_path = "input.jpg"  # replace with grayscale img
    img_matrix = load_grayscale_image(image_path)
    print("Image loaded with shape:", img_matrix.shape)

    k = 50
    compressed_img, U_k, S_k, VT_k = compress_image_svd(img_matrix, k)
    compressed_img = np.clip(compressed_img, 0, 255)

    plot_results(img_matrix, compressed_img, k)
    compression_stats(img_matrix.shape, k)

    # Save in multiple formats for fair file-size comparisons.
    compressed_img = Image.fromarray(compressed_img.astype("uint8"))
    output_dir = os.path.join("examples", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_png = os.path.join(output_dir, f"compressed_k{k}.png")
    output_jpg = os.path.join(output_dir, f"compressed_k{k}.jpg")
    output_webp = os.path.join(output_dir, f"compressed_k{k}.webp")
    output_npz = os.path.join(output_dir, f"svd_factors_k{k}.npz")

    compressed_img.save(output_png, optimize=True, compress_level=9)
    compressed_img.save(output_jpg, format="JPEG", quality=85, optimize=True)
    compressed_img.save(output_webp, format="WEBP", quality=80, method=6)
    save_svd_npz(output_npz, U_k, S_k, VT_k, img_matrix.shape, k)

    print(f"Compressed image saved as {output_png}")
    print(f"Compressed image saved as {output_jpg}")
    print(f"Compressed image saved as {output_webp}")
    print(f"SVD factors saved as {output_npz}")
    print_file_size_report(image_path, [output_png, output_jpg, output_webp, output_npz])

if __name__ == "__main__":
    main()