import numpy as np
from collections import Counter

def entropy(data):
    values, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))

height = 384
width = 512

gradient = np.tile(
    np.linspace(0, 255, width, dtype=np.uint8),
    (height, 1)
)

np.random.seed(42)
noise = np.random.randint(
    0, 256, (height, width), dtype=np.uint8
)

screenshot = np.zeros((height, width), dtype=np.uint8)
screenshot[:100, :] = 240
screenshot[100:250, :] = 80
screenshot[250:, :] = 200
screenshot[120:230, 80:430] = 150
screenshot[140:200, 120:390] = 220

y = np.linspace(0, 1, height).reshape(-1, 1)
x = np.linspace(0, 1, width).reshape(1, -1)

photo = (
    100
    + 60 * np.sin(x * 10)
    + 50 * np.cos(y * 8)
    + 30 * np.sin((x + y) * 15)
)

photo = np.clip(photo, 0, 255).astype(np.uint8)

images = {
    "Градієнт": gradient,
    "Білий шум": noise,
    "Скріншот": screenshot,
    "Фотографія": photo
}

def filter_row(cur, prev, f):
    a = np.concatenate(([0], cur[:-1])).astype(np.uint8)
    b = prev

    if f == 0:
        pred = np.zeros_like(cur)
    elif f == 1:
        pred = a
    elif f == 2:
        pred = b

    return ((cur.astype(int) - pred.astype(int)) % 256).astype(np.uint8)


row = np.array(
    [40, 42, 44, 46, 48, 50, 52, 54],
    dtype=np.uint8
)

zeros = np.zeros(8, dtype=np.uint8)

print(filter_row(row, zeros, 1))
print(filter_row(row, zeros, 0))
print(filter_row(row, row, 2))


def filter_image(img, f):
    out = np.zeros_like(img)
    prev = np.zeros(img.shape[1], dtype=np.uint8)

    for y in range(img.shape[0]):
        out[y] = filter_row(img[y], prev, f)
        prev = img[y]

    return out


print()
print("Зображення        До      None      Sub       Up")

entropy_results = {}

for name, img in images.items():
    e0 = entropy(img)
    e_none = entropy(filter_image(img, 0))
    e_sub = entropy(filter_image(img, 1))
    e_up = entropy(filter_image(img, 2))

    entropy_results[name] = {
        "before": e0,
        "None": e_none,
        "Sub": e_sub,
        "Up": e_up
    }

    print(
        f"{name:<16}"
        f"{e0:>7.2f}"
        f"{e_none:>10.2f}"
        f"{e_sub:>10.2f}"
        f"{e_up:>10.2f}"
    )


def unfilter_row(enc, prev, f):
    n = len(enc)
    out = np.zeros(n, dtype=np.uint8)

    for x in range(n):
        a = out[x - 1] if x > 0 else 0
        b = prev[x]

        if f == 0:
            pred = 0
        elif f == 1:
            pred = a
        elif f == 2:
            pred = b

        out[x] = (int(enc[x]) + int(pred)) % 256

    return out


def unfilter_image(img, filters):
    out = np.zeros_like(img)
    prev = np.zeros(img.shape[1], dtype=np.uint8)

    for y in range(img.shape[0]):
        out[y] = unfilter_row(img[y], prev, filters[y])
        prev = out[y]

    return out


print()
print("Перевірка відновлення:")

for f in (0, 1, 2):
    encoded = filter_image(gradient, f)
    filters = [f] * gradient.shape[0]
    decoded = unfilter_image(encoded, filters)

    print(f"Фільтр {f}: {np.array_equal(gradient, decoded)}")


def sum_abs(row):
    r = row.astype(int)
    signed = np.where(r > 127, r - 256, r)
    return int(np.abs(signed).sum())


def filter_adaptive(img):
    out = np.zeros_like(img)
    chosen = []

    prev = np.zeros(img.shape[1], dtype=np.uint8)

    for y in range(img.shape[0]):
        best_filter = 0
        best_row = None
        best_sum = None

        for f in (0, 1, 2):
            filtered = filter_row(img[y], prev, f)
            current_sum = sum_abs(filtered)

            if best_sum is None or current_sum < best_sum:
                best_sum = current_sum
                best_filter = f
                best_row = filtered

        out[y] = best_row
        chosen.append(best_filter)

        prev = img[y]

    return out, chosen


print()
print("Адаптивний фільтр:")

adaptive_results = {}

for name, img in images.items():
    filtered, chosen = filter_adaptive(img)

    e_before = entropy(img)
    e_after = entropy(filtered)

    counts = Counter(chosen)

    adaptive_results[name] = {
        "entropy": e_after,
        "chosen": chosen,
        "counts": counts
    }

    print(
        f"{name:<16}"
        f"До: {e_before:.2f}  "
        f"Після: {e_after:.2f}  "
        f"None: {counts.get(0, 0)}  "
        f"Sub: {counts.get(1, 0)}  "
        f"Up: {counts.get(2, 0)}"
    )


print()
print("Підсумкова таблиця:")
print("Зображення        Фільтр       До / після       Виграш")

for name, img in images.items():
    before = entropy(img)

    values = {
        "None": entropy(filter_image(img, 0)),
        "Sub": entropy(filter_image(img, 1)),
        "Up": entropy(filter_image(img, 2)),
        "Adaptive": adaptive_results[name]["entropy"]
    }

    best_name = min(values, key=values.get)
    after = values[best_name]

    gain = (before - after) / before * 100

    print(
        f"{name:<16}"
        f"{best_name:<13}"
        f"{before:.2f} / {after:.2f}    "
        f"{gain:.2f}%"
    )

print()
print("Готово.")