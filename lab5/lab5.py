
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rgb2ycbcr(img):
    a = img.astype(np.float64)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return Y, Cb, Cr


def ycbcr2rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255).round().astype(np.uint8)


def down422(C):
    h, w = C.shape
    C = C[:, :w // 2 * 2]
    return C.reshape(h, -1, 2).mean(axis=2)


def up422(C, w):
    return np.repeat(C, 2, axis=1)[:, :w]


def down420(C):
    h, w = C.shape
    C = C[:h // 2 * 2, :w // 2 * 2]
    return C.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def up420(C, h, w):
    return np.repeat(np.repeat(C, 2, axis=0), 2, axis=1)[:h, :w]


def scheme(img, mode):
    Y, Cb, Cr = rgb2ycbcr(img)
    h, w = Y.shape

    if mode == "4:4:4":
        pass
    elif mode == "4:2:2":
        Cb = up422(down422(Cb), w)
        Cr = up422(down422(Cr), w)
    elif mode == "4:2:0":
        Cb = up420(down420(Cb), h, w)
        Cr = up420(down420(Cr), h, w)
    else:
        raise ValueError("Невідома схема")

    return ycbcr2rgb(Y, Cb, Cr)


def redtext(W=320, H=96):
    im = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()

    for path in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]:
        try:
            font = ImageFont.truetype(path, 48)
            break
        except OSError:
            continue

    draw.text((12, 22), "ЧЕРВОНИЙ", font=font, fill=(255, 0, 0))
    return np.array(im)


def lines(W=320, H=160):
    im = np.zeros((H, W, 3), dtype=np.uint8)
    cols = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255)
    ]

    for i, c in enumerate(cols):
        x = 24 + i * 48
        im[12:70, x:x + 2] = c
        im[12:70, x + 22:x + 23] = c

    for i, c in enumerate(cols[:4]):
        y = 90 + i * 16
        im[y:y + 2, 24:296] = c
        im[y + 7:y + 8, 24:296] = c

    return im


def diffmap(orig, out):
    d = np.abs(out.astype(int) - orig.astype(int))
    return d.max(axis=2)


def save_diff(orig, out, filename):
    dm = diffmap(orig, out)
    plt.figure()
    plt.imshow(np.clip(dm * 2, 0, 255), cmap="inferno", vmin=0, vmax=255)
    plt.axis("off")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def measure(orig, mode):
    out = scheme(orig, mode)
    d = np.abs(out.astype(int) - orig.astype(int))
    mx = d.max(axis=2)
    return d.max(), d.mean(), (mx > 0).mean() * 100, (mx > 30).mean() * 100


def main():
    photo = data.astronaut()
    txt = redtext()
    lin = lines()

    Y, Cb, Cr = rgb2ycbcr(photo)
    back = ycbcr2rgb(Y, Cb, Cr)
    err = np.abs(back.astype(int) - photo.astype(int))
    print("RGB -> YCbCr -> RGB, максимальна похибка:", err.max())

    out422 = scheme(txt, "4:2:2")
    Image.fromarray(np.concatenate([txt, out422], axis=0)).save("text422.png")
    print("Червоний текст 4:2:2, похибка:",
          np.abs(out422.astype(int) - txt.astype(int)).max())

    out420 = scheme(txt, "4:2:0")
    Image.fromarray(np.concatenate([txt, out420], axis=0)).save("text420.png")
    print("Червоний текст 4:2:0, похибка:",
          np.abs(out420.astype(int) - txt.astype(int)).max())

    save_diff(txt, scheme(txt, "4:2:0"), "diff_text.png")
    save_diff(photo, scheme(photo, "4:2:0"), "diff_photo.png")

    trio = [scheme(lin, m) for m in ("4:4:4", "4:2:2", "4:2:0")]
    Image.fromarray(np.concatenate(trio, axis=1)).save("lines420.png")

    tests = {
        "Червоний текст": txt,
        "Тонкі лінії": lin,
        "Фотографія": photo
    }

    print("\nТАБЛИЦЯ ВИМІРЮВАНЬ")
    print(f'{"Зображення":<18} {"Схема":<8} {"Макс":>5} '
          f'{"Середня":>8} {"Змінені":>10} {"Помітні":>10}')

    for name, img in tests.items():
        for mode in ("4:4:4", "4:2:2", "4:2:0"):
            mx, avg, changed, visible = measure(img, mode)
            print(f"{name:<18} {mode:<8} {mx:5d} {avg:8.2f} "
                  f"{changed:9.1f}% {visible:9.2f}%")

    print("\nГотово! Створено п'ять PNG-файлів.")


if __name__ == "__main__":
    main()
