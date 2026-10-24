import numpy as np
from PIL import Image
from skimage import data
from pathlib import Path

OUT = Path(__file__).resolve().parent


def to_light(c):
    c = np.asarray(c, dtype=float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_code(L):
    L = np.asarray(L, dtype=float)
    c = np.where(L <= 0.0031308, L * 12.92, 1.055 * np.power(np.maximum(L, 0), 1 / 2.4) - 0.055)
    return np.clip(c * 255, 0, 255)


def checker(N=512, cell=4, ang=18):
    y, x = np.indices((N, N)).astype(float) - N / 2
    a = np.deg2rad(ang)
    xr = x * np.cos(a) - y * np.sin(a)
    yr = x * np.sin(a) + y * np.cos(a)
    return (((np.floor(xr / cell) + np.floor(yr / cell)) % 2) * 255).astype(np.uint8)


def half_naive(img):
    img = np.asarray(img, dtype=float)
    h, w = img.shape
    img = img[:h // 2 * 2, :w // 2 * 2]
    h, w = img.shape
    return img.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def half_linear(img):
    L = to_light(img)
    h, w = L.shape
    L = L[:h // 2 * 2, :w // 2 * 2]
    h, w = L.shape
    return to_code(L.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)))


def save_gray(name, arr):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(OUT / name)


def main():
    print("Лабораторна робота №8 — Ресемплінг і міпмапи")
    gray = np.array(Image.fromarray(data.astronaut()).convert("L"))
    # Підготовка зображення 512x512 для контрольних значень
    gray = np.array(Image.fromarray(gray).resize((512, 512), Image.Resampling.LANCZOS))

    small = gray[340:356, 372:388]
    im = Image.fromarray(small)
    im.resize((320, 320), Image.Resampling.NEAREST).save(OUT / "up_nearest.png")
    im.resize((320, 320), Image.Resampling.BILINEAR).save(OUT / "up_bilinear.png")
    im.resize((320, 320), Image.Resampling.LANCZOS).save(OUT / "up_lanczos.png")
    print("Завдання 1: збережено up_nearest.png, up_bilinear.png, up_lanczos.png")

    ch = checker()
    save_gray("checker.png", ch)
    k = 4
    nearest = ch[::k, ::k]
    area = ch.reshape(512 // k, k, 512 // k, k).mean(axis=(1, 3))
    save_gray("ds_nearest.png", nearest)
    save_gray("ds_area.png", area)
    print("Завдання 2:")
    print("  кожен 4-й піксель: середнє %.1f, розкид %.1f" % (nearest.mean(), nearest.std()))
    print("  усереднення 4x4: середнє %.1f, розкид %.1f" % (area.mean(), area.std()))

    g = gray.astype(float)
    level = 0
    while g.shape[0] >= 2 and g.shape[1] >= 2:
        save_gray(f"mip_{level}.png", g)
        print("Міпмап %d: %dx%d, середня яскравість %.1f" %
              (level, g.shape[1], g.shape[0], g.mean()))
        g = half_naive(g)
        level += 1

    cb = (np.indices((256, 256)).sum(axis=0) % 2 * 255).astype(float)
    print("Завдання 4, шахівниця 1 піксель:")
    print("  наївно: %.1f" % half_naive(cb).mean())
    print("  правильно через лінійне світло: %.1f" % half_linear(cb).mean())

    gn = gray.astype(float)
    gl = gray.astype(float)
    print("Рівень | Розмір | Наївно | Правильно | Різниця")
    print("0      | 512    | %.1f   | %.1f      | %.1f" % (gn.mean(), gl.mean(), gl.mean() - gn.mean()))
    for lv in range(1, 6):
        gn, gl = half_naive(gn), half_linear(gl)
        print("%d      | %-6d | %.1f   | %.1f      | %+.1f" %
              (lv, gn.shape[0], gn.mean(), gl.mean(), gl.mean() - gn.mean()))

    print("Готово. Усі зображення збережено в:", OUT)


if __name__ == "__main__":
    main()
