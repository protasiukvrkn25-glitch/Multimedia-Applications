
import time
import numpy as np
import cv2
from PIL import Image
from skimage import data


def conv2(img, K):
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=float)

    for i in range(kh):
        for j in range(kw):
            out += K[i, j] * pad[
                i:i + img.shape[0],
                j:j + img.shape[1]
            ]

    return out


def gauss1d(sigma, n):
    x = np.arange(n) - n // 2
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def conv_sep(img, k):
    n = len(k) // 2

    pad = np.pad(img, ((0, 0), (n, n)), mode="edge")
    tmp = np.zeros_like(img, dtype=float)

    for j, v in enumerate(k):
        tmp += v * pad[:, j:j + img.shape[1]]

    pad = np.pad(tmp, ((n, n), (0, 0)), mode="edge")
    out = np.zeros_like(img, dtype=float)

    for i, v in enumerate(k):
        out += v * pad[i:i + img.shape[0], :]

    return out


def best(f, r=5):
    times = []

    for _ in range(r):
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)

    return min(times) * 1000


def save_image(filename, img):
    result = np.clip(np.round(img), 0, 255).astype(np.uint8)
    Image.fromarray(result).save(filename)


def main():
    gray = np.array(
        Image.fromarray(data.astronaut()).convert("L")
    ).astype(float)

    save_image("original.png", gray)

    mean_kernel = np.ones((3, 3)) / 9
    mean_result = conv2(gray, mean_kernel)

    print("Середня яскравість оригіналу:", gray.mean())
    print("Середня яскравість після усереднення:", mean_result.mean())

    k = gauss1d(1.6, 7)
    BLUR = np.outer(k, k)
    blur = conv2(gray, BLUR)
    save_image("blur.png", blur)

    SX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)

    SY = SX.T
    gx = conv2(gray, SX)
    gy = conv2(gray, SY)
    edges = np.hypot(gx, gy)

    if edges.max() > 0:
        edges = edges / edges.max() * 255

    save_image("sobel.png", edges)

    detail = gray - blur
    sharp = np.clip(gray + 0.7 * detail, 0, 255)
    save_image("sharp.png", sharp)

    print("\nЗавдання 3. Порівняння часу")
    print(f"{'Ядро':<10}{'Звичайна, мс':<18}"
          f"{'Роздільна, мс':<18}{'Виграш':<12}")

    for n in (3, 5, 9, 15, 31):
        k = gauss1d(max(n / 5, 0.6), n)
        K = np.outer(k, k)

        ordinary_time = best(lambda: conv2(gray, K))
        separate_time = best(lambda: conv_sep(gray, k))

        print(
            f"{n}x{n:<8}{ordinary_time:<18.3f}"
            f"{separate_time:<18.3f}"
            f"{ordinary_time / separate_time:.2f}x"
        )

        difference = np.abs(
            conv2(gray, K) - conv_sep(gray, k)
        ).max()

        print(f"Максимальна розбіжність: {difference:.3e}")

    print("\nЗавдання 4. Порівняння з OpenCV")
    n, sigma = 15, 3.0
    k = gauss1d(sigma, n)

    mine = conv_sep(gray, k)
    lib = cv2.GaussianBlur(
        gray, (n, n), sigma,
        borderType=cv2.BORDER_REPLICATE
    )

    print("Максимальна розбіжність:", np.abs(mine - lib).max())
    print(f"Мій час: {best(lambda: conv_sep(gray, k)):.3f} мс")
    print(
        f"OpenCV: {best(lambda: cv2.GaussianBlur(gray, (n, n), sigma, borderType=cv2.BORDER_REPLICATE)):.3f} мс"
    )

    print("\nУсі зображення збережено.")
    

if __name__ == "__main__":
    main()
