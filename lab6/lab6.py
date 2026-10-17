
import csv
import sys
import zlib

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage.io import imread, imsave
from skimage.metrics import structural_similarity


Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)


def dct_matrix(N=8):
    D = np.zeros((N, N))
    for k in range(N):
        a = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        for n in range(N):
            D[k, n] = a * np.cos(
                np.pi * (2 * n + 1) * k / (2 * N)
            )
    return D


D = dct_matrix()


def dct2(block):
    return D @ block @ D.T


def idct2(coeffs):
    return D.T @ coeffs @ D


def qtable(q):
    q = max(1, min(99, int(q)))
    S = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((S * Q50 + 50) / 100), 1, 255)


def load_gray(path=None):
    if path:
        img = imread(path, as_gray=True)
        if img.max() <= 1:
            img = img * 255
    else:
        img = camera()

    img = np.asarray(img, dtype=np.float64)
    h = (img.shape[0] // 8) * 8
    w = (img.shape[1] // 8) * 8
    return np.clip(img[:h, :w], 0, 255).round().astype(np.uint8)


def blockify(img):
    h, w = img.shape
    return (
        img.reshape(h // 8, 8, w // 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 8, 8)
    )


def unblockify(blocks, h, w):
    return (
        blocks.reshape(h // 8, w // 8, 8, 8)
        .transpose(0, 2, 1, 3)
        .reshape(h, w)
    )


def encode(img, q):
    Q = qtable(q)
    blocks = blockify(img.astype(np.float64) - 128)
    coeffs = np.array([
        np.round(dct2(block) / Q) for block in blocks
    ])
    return coeffs.astype(np.int16), Q


def decode(coeffs, Q, h, w):
    blocks = np.array([
        idct2(block.astype(np.float64) * Q)
        for block in coeffs
    ])
    img = unblockify(blocks, h, w) + 128
    return np.clip(img, 0, 255).round().astype(np.uint8)


def psnr(original, reconstructed):
    a = original.astype(np.float64)
    b = reconstructed.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255 ** 2 / mse))


def ssim(original, reconstructed):
    return float(structural_similarity(
        original, reconstructed, data_range=255
    ))


def compressed_size(coeffs):
    return len(zlib.compress(coeffs.tobytes(), level=9))


def run_tests():
    print("ЛАБОРАТОРНА РОБОТА №6 — JPEG-lite")

    assert np.allclose(D @ D.T, np.eye(8), atol=1e-12)
    print("[OK] Базис DCT ортогональний")

    block = np.arange(64, dtype=float).reshape(8, 8)
    error = np.max(np.abs(block - idct2(dct2(block))))
    assert error < 1e-10
    print(f"[OK] Зворотне DCT, похибка: {error:.3e}")

    assert np.array_equal(qtable(50), Q50)
    print("[OK] Таблиця qtable(50) збігається з Q50")

    img = load_gray()
    coeffs, Q = encode(img, 50)
    restored = decode(coeffs, Q, *img.shape)
    assert restored.shape == img.shape
    print("[OK] Кодер і декодер працюють")
    print(f"PSNR: {psnr(img, restored):.2f} dB")
    print(f"SSIM: {ssim(img, restored):.4f}")


def main():
    run_tests()

    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    original = load_gray(input_path)
    h, w = original.shape
    raw_size = h * w
    qualities = [95, 90, 75, 50, 30, 15, 5]
    results = []

    print("\nРезультати стиснення")
    print("q | байти | PSNR | SSIM | нульові коефіцієнти")

    for q in qualities:
        coeffs, Q = encode(original, q)
        restored = decode(coeffs, Q, h, w)

        size = compressed_size(coeffs)
        p = psnr(original, restored)
        s = ssim(original, restored)
        zero_percent = 100 * (
            1 - np.count_nonzero(coeffs) / coeffs.size
        )

        imsave(f"q{q:02d}.png", restored, check_contrast=False)

        results.append({
            "quality": q,
            "bytes": size,
            "compression_ratio": round(raw_size / max(size, 1), 3),
            "zero_coefficients_percent": round(zero_percent, 2),
            "PSNR_dB": round(p, 3),
            "SSIM": round(s, 5)
        })

        print(
            f"{q:2d} | {size:6d} | {p:6.2f} | "
            f"{s:.4f} | {zero_percent:.1f}%"
        )

    with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(
        [r["quality"] for r in results],
        [r["bytes"] for r in results], marker="o"
    )
    axes[0].set_title("Розмір файлу")
    axes[0].set_ylabel("Байт")

    axes[1].plot(
        [r["quality"] for r in results],
        [r["PSNR_dB"] for r in results], marker="o"
    )
    axes[1].set_title("PSNR")
    axes[1].set_ylabel("дБ")

    axes[2].plot(
        [r["quality"] for r in results],
        [r["SSIM"] for r in results], marker="o"
    )
    axes[2].set_title("SSIM")
    axes[2].set_ylabel("Значення")

    for ax in axes:
        ax.set_xlabel("Якість q")
        ax.grid(True)

    fig.suptitle("JPEG-lite: розмір і якість")
    fig.tight_layout()
    fig.savefig("curve.png", dpi=150)
    plt.close(fig)

    print("\nСтворено curve.png, results.csv та зображення q95.png–q05.png.")
    print("Для власного зображення: python lab6.py photo.png")


if __name__ == "__main__":
    main()
