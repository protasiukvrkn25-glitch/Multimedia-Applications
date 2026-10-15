import numpy as np
from PIL import Image
import io


# 1. Градієнт
grad = np.tile(
    np.linspace(0, 255, 512).astype(np.uint8),
    (384, 1)
)

# 2. Білий шум
rng = np.random.default_rng(1)
noise = rng.integers(
    0, 256, (384, 512)
).astype(np.uint8)

# 3. Скріншот
scr = np.array(
    Image.open("screen.png")
    .convert("L")
    .resize((512, 384))
)

# 4. Фотографія
photo = np.array(
    Image.open("photo.jpg")
    .convert("L")
    .resize((512, 384))
)

images = {
    "градієнт": grad,
    "білий шум": noise,
    "скріншот": scr,
    "фотографія": photo
}


# Ентропія
def entropy(img):
    values, counts = np.unique(
        img,
        return_counts=True
    )

    p = counts / counts.sum()

    return float(
        -(p * np.log2(p)).sum()
    )


# RLE
def rle_size(img):
    data = img.ravel()
    n = len(data)

    i = 0
    out = 0

    while i < n:
        j = i

        while (
            j + 1 < n
            and data[j + 1] == data[i]
            and j - i + 1 < 255
        ):
            j += 1

        out += 2
        i = j + 1

    return out


# Хаффман
def huffman_codes(freq):
    if len(freq) == 1:
        return {list(freq)[0]: "0"}

    nodes = [
        [n, {v: ""}]
        for v, n in freq.items()
    ]

    while len(nodes) > 1:
        nodes.sort(
            key=lambda node: node[0]
        )

        a = nodes.pop(0)
        b = nodes.pop(0)

        for v in a[1]:
            a[1][v] = "0" + a[1][v]

        for v in b[1]:
            b[1][v] = "1" + b[1][v]

        nodes.append([
            a[0] + b[0],
            {**a[1], **b[1]}
        ])

    return nodes[0][1]


# PNG
def png_size(img):
    buf = io.BytesIO()

    Image.fromarray(img).save(
        buf,
        format="PNG",
        optimize=True
    )

    return len(buf.getvalue())


# Результати
original_size = 512 * 384

print("=" * 60)
print("ЛАБОРАТОРНА РОБОТА №3")
print("ЕНТРОПІЙНЕ СТИСНЕННЯ")
print("=" * 60)

print("\nЗАВДАННЯ 1. ЕНТРОПІЯ")
print("-" * 60)

results = {}

for name, im in images.items():
    e = entropy(im)

    print(
        f"{name:<12} "
        f"{e:.2f} біта на піксель"
    )

    results[name] = {
        "entropy": e
    }


print("\nЗАВДАННЯ 2. RLE")
print("-" * 60)

for name, im in images.items():
    size = rle_size(im)
    percent = size / original_size * 100

    print(
        f"{name:<12} "
        f"{size} байт, "
        f"{percent:.2f}%"
    )

    results[name]["rle"] = percent


print("\nЗАВДАННЯ 3. ХАФФМАН")
print("-" * 60)

for name, im in images.items():

    values, counts = np.unique(
        im,
        return_counts=True
    )

    freq = {
        int(v): int(c)
        for v, c in zip(values, counts)
    }

    codes = huffman_codes(freq)

    total = sum(freq.values())

    avg = sum(
        freq[v] * len(codes[v])
        for v in freq
    ) / total

    lengths = [
        len(c)
        for c in codes.values()
    ]

    min_code = min(lengths)
    max_code = max(lengths)

    print(
        f"{name:<12} "
        f"ентропія: {entropy(im):.2f} | "
        f"Хаффман: {avg:.2f} біт | "
        f"коди: {min_code}-{max_code} біт"
    )

    results[name]["huffman"] = avg
    results[name]["min_code"] = min_code
    results[name]["max_code"] = max_code


print("\nЗАВДАННЯ 4. PNG")
print("-" * 60)

for name, im in images.items():

    size = png_size(im)
    percent = size / original_size * 100

    print(
        f"{name:<12} "
        f"{size} байт, "
        f"{percent:.2f}%"
    )

    results[name]["png"] = percent


print("\n" + "=" * 60)
print("ПІДСУМКОВА ТАБЛИЦЯ")
print("=" * 60)

print(
    f"{'Зображення':<15}"
    f"{'Ентропія':<12}"
    f"{'Хаффман':<12}"
    f"{'RLE %':<12}"
    f"{'PNG %':<12}"
)

print("-" * 60)

for name in images:

    r = results[name]

    print(
        f"{name:<15}"
        f"{r['entropy']:<12.2f}"
        f"{r['huffman']:<12.2f}"
        f"{r['rle']:<12.2f}"
        f"{r['png']:<12.2f}"
    )