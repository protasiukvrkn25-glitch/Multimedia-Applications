import numpy as np
from pathlib import Path
from PIL import Image

OUT = Path("lab2_results")
OUT.mkdir(exist_ok=True)


# 1. ГАМА

def to_light(code):
    """Код пікселя 0...255 -> фізичне світло 0...1"""
    c = np.asarray(code, dtype=np.float64) / 255
    return np.where(
        c <= 0.04045,
        c / 12.92,
        ((c + 0.055) / 1.055) ** 2.4
    )


def to_code(light):
    """Фізичне світло 0...1 -> код пікселя 0...255"""
    l = np.asarray(light, dtype=np.float64)
    s = np.where(
        l <= 0.0031308,
        l * 12.92,
        1.055 * l ** (1 / 2.4) - 0.055
    )
    return s * 255


print("=== ЗАВДАННЯ 1. ГАМА ===")

mid = (to_light(0) + to_light(255)) / 2

print("Середнє чорного і білого:", round(float(to_code(mid))))
print("Світло для коду 128:", round(float(to_light(128) * 100), 1), "%")
print("Світло для коду 64: ", round(float(to_light(64) * 100), 1), "%")
print("Світло для коду 192:", round(float(to_light(192) * 100), 1), "%")

test = np.array([0, 128, 255])
print("Перевірка масиву:", to_light(test))

# 2. ШАХІВНИЦЯ

def make_board(size=512, cell=8):
    yy, xx = np.indices((size, size))
    return (((yy // cell + xx // cell) % 2) * 255).astype(np.uint8)


def box_average(img, k):
    """Зменшує зображення у k разів усередненням квадратів k x k"""
    h, w = img.shape
    return img.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


print("\n=== ЗАВДАННЯ 2. ШАХІВНИЦЯ ===")

board = make_board(512, 8)

# Наївне зменшення: середнє без переходу в лінійний простір
naive = box_average(board.astype(np.float64), 16)

# Правильне зменшення:
# 1. код -> світло
# 2. усереднення світла
# 3. світло -> код
linear = to_light(board)
correct = to_code(box_average(linear, 16))

print("Наївно:   ", round(float(naive.mean()), 1))
print("Правильно:", round(float(correct.mean()), 1))
print("Різниця:  ", round(float(correct.mean() - naive.mean()), 1))

original_light = to_light(board).mean() * 100
naive_light = to_light(naive).mean() * 100
correct_light = to_light(correct).mean() * 100

print("Середнє світло оригіналу:", round(float(original_light), 1), "%")
print("Середнє світло наївного:", round(float(naive_light), 1), "%")
print("Середнє світло правильного:", round(float(correct_light), 1), "%")

# Збереження зображень
Image.fromarray(board).save(OUT / "01_original_board.bmp")
Image.fromarray(np.clip(naive, 0, 255).astype(np.uint8)).save(
    OUT / "02_naive_board.bmp"
)
Image.fromarray(np.clip(correct, 0, 255).astype(np.uint8)).save(
    OUT / "03_correct_board.bmp"
)


# 3. YCbCr

def rgb_to_ycbcr(img):
    x = img.astype(np.float64)

    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)

    return np.stack([r, g, b], axis=-1)


def load_color_image():
    # Якщо у папці є image.jpg/png/jpeg/bmp — використовуємо його.
    possible = [
        Path("image.jpg"),
        Path("image.jpeg"),
        Path("image.png"),
        Path("image.bmp"),
        Path("photo.jpg"),
        Path("photo.png")
    ]

    for file in possible:
        if file.exists():
            img = Image.open(file).convert("RGB")
            return np.array(img)

    # Якщо фотографії немає, створюємо тестове кольорове зображення.
    print("\nФото не знайдено.")
    print("Створюю test_image.png для перевірки завдання 3-4.")
    h, w = 300, 400
    yy, xx = np.indices((h, w))

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = (xx / w * 255).astype(np.uint8)
    img[..., 1] = (yy / h * 255).astype(np.uint8)
    img[..., 2] = ((1 - xx / w) * 255).astype(np.uint8)

    Image.fromarray(img).save("test_image.png")
    return img


print("\n=== ЗАВДАННЯ 3. YCbCr ===")

img = load_color_image()

y, cb, cr = rgb_to_ycbcr(img)
back = ycbcr_to_rgb(y, cb, cr)

back_uint8 = np.clip(np.round(back), 0, 255).astype(np.uint8)

max_difference = np.abs(
    back_uint8.astype(int) - img.astype(int)
).max()

print("Максимальна різниця після RGB -> YCbCr -> RGB:",
      int(max_difference))

gray_y = np.stack([y, y, y], axis=-1)
gray_y = np.clip(gray_y, 0, 255).astype(np.uint8)

naive_gray_value = img.astype(np.float64).mean(axis=2)
naive_gray = np.stack(
    [naive_gray_value, naive_gray_value, naive_gray_value],
    axis=-1
)
naive_gray = np.clip(naive_gray, 0, 255).astype(np.uint8)

Image.fromarray(gray_y).save(OUT / "04_Y_channel.bmp")
Image.fromarray(naive_gray).save(OUT / "05_naive_gray.bmp")


# 4. HSV

def rgb_to_hsv(img):
    x = img.astype(np.float64) / 255

    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]

    mx = x.max(-1)
    mn = x.min(-1)
    d = mx - mn

    h = np.zeros_like(mx)

    m = (d != 0) & (mx == r)
    h[m] = (60 * ((g - b)[m] / d[m])) % 360

    m = (d != 0) & (mx == g)
    h[m] = 60 * ((b - r)[m] / d[m]) + 120

    m = (d != 0) & (mx == b)
    h[m] = 60 * ((r - g)[m] / d[m]) + 240

    s = np.where(
        mx == 0,
        0,
        d / np.where(mx == 0, 1, mx)
    )

    return h, s, mx


def hsv_to_rgb(h, s, v):
    h = h % 360
    c = v * s
    hp = h / 60
    x = c * (1 - np.abs((hp % 2) - 1))
    m = v - c

    r1 = np.zeros_like(h)
    g1 = np.zeros_like(h)
    b1 = np.zeros_like(h)

    masks = [
        (hp >= 0) & (hp < 1),
        (hp >= 1) & (hp < 2),
        (hp >= 2) & (hp < 3),
        (hp >= 3) & (hp < 4),
        (hp >= 4) & (hp < 5),
        (hp >= 5) & (hp < 6)
    ]

    r1[masks[0]], g1[masks[0]], b1[masks[0]] = c[masks[0]], x[masks[0]], 0
    r1[masks[1]], g1[masks[1]], b1[masks[1]] = x[masks[1]], c[masks[1]], 0
    r1[masks[2]], g1[masks[2]], b1[masks[2]] = 0, c[masks[2]], x[masks[2]]
    r1[masks[3]], g1[masks[3]], b1[masks[3]] = 0, x[masks[3]], c[masks[3]]
    r1[masks[4]], g1[masks[4]], b1[masks[4]] = x[masks[4]], 0, c[masks[4]]
    r1[masks[5]], g1[masks[5]], b1[masks[5]] = c[masks[5]], 0, x[masks[5]]

    rgb = np.stack([r1 + m, g1 + m, b1 + m], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


print("\n=== ЗАВДАННЯ 4. HSV ===")

h, s, v = rgb_to_hsv(img)

# Перевірка без змін
original_back = hsv_to_rgb(h, s, v)
hsv_difference = np.abs(
    original_back.astype(int) - img.astype(int)
).max()

print("Різниця HSV туди-назад без змін:",
      int(hsv_difference))

# 1. Насиченість * 0.3
s_low = s * 0.3
result_low = hsv_to_rgb(h, s_low, v)
Image.fromarray(result_low).save(OUT / "06_HSV_saturation_03.bmp")

# 2. Насиченість * 2, обмеження до 1
s_high = np.clip(s * 2, 0, 1)
result_high = hsv_to_rgb(h, s_high, v)
Image.fromarray(result_high).save(OUT / "07_HSV_saturation_2.bmp")

# 3. Зсув тону на 60 градусів
h_shift = (h + 60) % 360
result_hue = hsv_to_rgb(h_shift, s, v)
Image.fromarray(result_hue).save(OUT / "08_HSV_hue_plus_60.bmp")

print("1. Насиченість 0.3 — зображення стає блідим, майже сірим.")
print("2. Насиченість 2 — кольори стають більш насиченими.")
print("3. Тон +60° — кольори перефарбовуються, світлі та темні ділянки зберігаються.")


# 5. ТАБЛИЦЯ І ВИСНОВОК

print("\n=== ЗАВДАННЯ 5. ТАБЛИЦЯ ===")
print()
print("| Зображення                    | Середній код | Середнє світло |")
print("|-------------------------------|--------------|----------------|")
print(f"| Оригінальна шахівниця         | {board.mean():10.1f} | {original_light:12.1f}% |")
print(f"| Зменшена наївно в sRGB        | {naive.mean():10.1f} | {naive_light:12.1f}% |")
print(f"| Зменшена в лінійному просторі | {correct.mean():10.1f} | {correct_light:12.1f}% |")

conclusion = """
Висновок:
У цій лабораторній роботі було досліджено вплив гами на обробку зображень.
Під час наївного зменшення шахівниці усереднення виконується без переходу
в лінійний простір, тому середній код становить 127,5, але середнє фізичне
світло зменшується до приблизно 21,4%. При правильному способі спочатку
виконується перехід у лінійний простір, після чого усереднення і повернення
до коду. У результаті отримано середній код 187,5 і збережено 50% світла.
Також було виконано перетворення RGB у YCbCr і назад, а за допомогою HSV
досліджено зміну насиченості та тону зображення.
"""

print(conclusion)

(OUT / "09_table_and_conclusion.txt").write_text(
    f"""Лабораторна робота №2
Гама і колірні простори

Таблиця:

Оригінальна шахівниця: середній код {board.mean():.1f}, середнє світло {original_light:.1f}%
Зменшена наївно в sRGB: середній код {naive.mean():.1f}, середнє світло {naive_light:.1f}%
Зменшена в лінійному просторі: середній код {correct.mean():.1f}, середнє світло {correct_light:.1f}%

Максимальна різниця RGB -> YCbCr -> RGB: {int(max_difference)}
Різниця HSV туди-назад без змін: {int(hsv_difference)}

{conclusion}
""",
    encoding="utf-8"
)

print(f"\nУсі результати збережено в папці: {OUT.resolve()}")