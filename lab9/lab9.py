import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# Лабораторна робота №9. BC1 encoder.
# Запуск: python lab9.py
# Результати зберігаються в папці results.

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


def to565(color):
    """Квантує RGB-колір до формату RGB565 і повертає його назад у RGB."""
    color = np.clip(np.asarray(color, dtype=np.float32), 0, 255)
    r = int(round(float(color[0]) * 31 / 255))
    g = int(round(float(color[1]) * 63 / 255))
    b = int(round(float(color[2]) * 31 / 255))
    return np.array([
        round(r * 255 / 31),
        round(g * 255 / 63),
        round(b * 255 / 31)
    ], dtype=np.float32)


def make_palette(c0, c1):
    """Створює палітру з двох кінцевих кольорів та двох проміжних."""
    c0 = np.asarray(c0, dtype=np.float32)
    c1 = np.asarray(c1, dtype=np.float32)
    return np.stack((
        c0,
        c1,
        (2 * c0 + c1) / 3,
        (c0 + 2 * c1) / 3
    )).astype(np.float32)


def pick_endpoints(blk, mode="bbox"):
    """Вибирає два крайні кольори блока."""
    if mode == "bbox":
        return blk.max(axis=0), blk.min(axis=0)

    if mode == "pixel":
        luminance = blk @ np.array([0.299, 0.587, 0.114])
        return blk[int(luminance.argmax())], blk[int(luminance.argmin())]

    raise ValueError("Невідомий режим. Використовуйте 'bbox' або 'pixel'.")


def encode_block(blk, mode="bbox"):
    """Кодує блок 4x4: формує палітру та індекс найближчого кольору."""
    c0, c1 = pick_endpoints(blk, mode)
    c0 = to565(c0)
    c1 = to565(c1)

    # Для чотириколірного режиму BC1 перший endpoint має бути не меншим.
    if np.dot(c0, c0) < np.dot(c1, c1):
        c0, c1 = c1, c0

    palette = make_palette(c0, c1)
    distances = ((blk[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    indices = distances.argmin(axis=1)
    return palette, indices


def encode_image(image, mode="bbox"):
    """Кодує зображення блоками 4x4 та повертає відновлене зображення."""
    image = np.asarray(image, dtype=np.uint8)
    height, width, _ = image.shape

    # BC1 працює з блоками 4x4. Доповнюємо зображення до кратного 4 розміру.
    padded_h = ((height + 3) // 4) * 4
    padded_w = ((width + 3) // 4) * 4
    padded = np.pad(
        image,
        ((0, padded_h - height), (0, padded_w - width), (0, 0)),
        mode="edge"
    )
    decoded = np.zeros_like(padded, dtype=np.uint8)

    for y in range(0, padded_h, 4):
        for x in range(0, padded_w, 4):
            block = padded[y:y + 4, x:x + 4].reshape(16, 3).astype(np.float32)
            palette, indices = encode_block(block, mode)
            reconstructed = palette[indices].reshape(4, 4, 3)
            decoded[y:y + 4, x:x + 4] = np.clip(
                np.rint(reconstructed), 0, 255
            ).astype(np.uint8)

    return decoded[:height, :width]


def make_gradient(size=128):
    y, x = np.mgrid[0:size, 0:size]
    return np.stack((
        x * 255 / (size - 1),
        y * 255 / (size - 1),
        (x + y) * 255 / (2 * (size - 1))
    ), axis=2).astype(np.uint8)


def make_normal_map(size=128):
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    nx = 0.35 * np.sin(5 * np.pi * x)
    ny = 0.35 * np.cos(5 * np.pi * y)
    nz = np.sqrt(np.maximum(0.01, 1 - nx * nx - ny * ny))
    normals = np.stack((nx, ny, nz), axis=2)
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    return np.clip((normals + 1) * 127.5, 0, 255).astype(np.uint8)


def make_sharp_edges(size=128):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    colors = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
        [255, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0]
    ], dtype=np.uint8)
    stripe_width = size // len(colors)
    for i, color in enumerate(colors):
        image[:, i * stripe_width:(i + 1) * stripe_width] = color
    return image


def make_photo():
    """Завантажує тестове фото astronaut; якщо scikit-image відсутній,
    створює кольорове тестове зображення."""
    try:
        from skimage import data
        return np.asarray(Image.fromarray(data.astronaut()).convert("RGB"), dtype=np.uint8)
    except Exception:
        size = 256
        y, x = np.mgrid[0:size, 0:size]
        r = 100 + 90 * np.sin(x / 19) + 35 * np.cos(y / 13)
        g = 110 + 80 * np.sin((x + y) / 27)
        b = 120 + 90 * np.cos(x / 23)
        return np.clip(np.stack((r, g, b), axis=2), 0, 255).astype(np.uint8)


def rmse(original, reconstructed):
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    return float(np.sqrt(np.mean(diff ** 2)))


def save_image(array, path):
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def difference_map(original, reconstructed):
    diff = np.abs(original.astype(np.int16) - reconstructed.astype(np.int16))
    return np.clip(diff * 4, 0, 255).astype(np.uint8)


def normal_map_error(original, reconstructed):
    """Рахує кутову похибку між нормалями, закодованими в RGB."""
    a = original.astype(np.float32) / 127.5 - 1.0
    b = reconstructed.astype(np.float32) / 127.5 - 1.0
    a /= np.maximum(np.linalg.norm(a, axis=2, keepdims=True), 1e-8)
    b /= np.maximum(np.linalg.norm(b, axis=2, keepdims=True), 1e-8)
    cosine = np.clip(np.sum(a * b, axis=2), -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    return float(angles.mean()), float(angles.max()), float(np.mean(angles > 2) * 100)


def main():
    tests = {
        "photo": make_photo(),
        "gradient": make_gradient(),
        "normal_map": make_normal_map(),
        "sharp_edges": make_sharp_edges()
    }

    print("Лабораторна робота №9 — BC1 encoder")
    print("=" * 68)
    print(f"{'Зображення':<16} {'BBox RMSE':>12} {'Pixel RMSE':>12}")
    print("-" * 68)

    results = {}
    for name, original in tests.items():
        results[name] = {}
        for mode in ("bbox", "pixel"):
            reconstructed = encode_image(original, mode)
            results[name][mode] = reconstructed
            save_image(reconstructed, OUT_DIR / f"bc1_{name}_{mode}.png")
            save_image(
                difference_map(original, reconstructed),
                OUT_DIR / f"difference_{name}_{mode}.png"
            )

        bbox_error = rmse(original, results[name]["bbox"])
        pixel_error = rmse(original, results[name]["pixel"])
        print(f"{name:<16} {bbox_error:>12.2f} {pixel_error:>12.2f}")

    normal_original = tests["normal_map"]
    mean_angle, max_angle, over_two = normal_map_error(
        normal_original, results["normal_map"]["pixel"]
    )
    print("-" * 68)
    print("Кутова похибка normal map (режим pixel):")
    print(f"Середня: {mean_angle:.2f}°")
    print(f"Максимальна: {max_angle:.2f}°")
    print(f"Пікселів з похибкою понад 2°: {over_two:.1f}%")

    # Додатковий власний тест: різкі кольорові переходи.
    own = make_sharp_edges(128)
    own_decoded = encode_image(own, "pixel")
    save_image(own, OUT_DIR / "own_original.png")
    save_image(own_decoded, OUT_DIR / "own_bc1.png")
    save_image(difference_map(own, own_decoded), OUT_DIR / "own_difference.png")

    print("=" * 68)
    print(f"Готово! Усі PNG-файли збережено в папці: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
