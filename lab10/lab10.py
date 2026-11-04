BPP = {
    "RGBA32": 32,
    "ASTC 4x4": 8,
    "ASTC 6x6": 128 / 36,
    "ASTC 8x8": 2
}


def mem(side, fmt, mips=True):
    bits = BPP[fmt]
    size = side * side * bits / 8

    if mips:
        size *= 4 / 3

    return size / 2**20


textures = [
    ("Ground_Albedo", 2048, 2048, "ASTC 6x6"),
    ("Ground_Normal", 2048, 1024, "ASTC 4x4"),
    ("Building_Albedo", 2048, 2048, "ASTC 6x6"),
    ("Building_Normal", 2048, 1024, "ASTC 4x4"),
    ("Props_Albedo", 1024, 1024, "ASTC 6x6"),
    ("Props_Normal", 1024, 512, "ASTC 4x4"),
    ("Skybox", 2048, 1024, "ASTC 8x8"),
    ("Character_Albedo", 1024, 1024, "ASTC 6x6"),
    ("Character_Normal", 1024, 512, "ASTC 4x4"),
    ("UI_Atlas", 2048, 2048, "ASTC 4x4"),
    ("Decals", 1024, 512, "ASTC 4x4"),
    ("Noise", 512, 512, "ASTC 8x8")
]

original_sizes = [
    2048, 2048, 2048, 2048,
    1024, 1024, 2048, 1024,
    1024, 2048, 1024, 512
]

initial = sum(mem(side, "RGBA32") for side in original_sizes)
optimized = sum(
    mem(new_side, fmt)
    for name, old_side, new_side, fmt in textures
)

print("=" * 75)
print("LABORATORY WORK 10 - UNITY TEXTURE MEMORY BUDGET")
print("=" * 75)

print(f"\nInitial memory: {initial:.2f} MB\n")

print(
    f"{'Texture':22} {'Old':>8} {'New':>8} "
    f"{'Format':>12} {'Saved':>9}"
)

print("-" * 75)

for name, old_side, new_side, fmt in textures:
    old_mem = mem(old_side, "RGBA32")
    new_mem = mem(new_side, fmt)
    saved = old_mem - new_mem

    print(
        f"{name:22} "
        f"{old_mem:8.2f} "
        f"{new_mem:8.2f} "
        f"{fmt:>12} "
        f"{saved:9.2f}"
    )

print("-" * 75)

print(f"Initial memory:    {initial:.2f} MB")
print(f"Optimized memory:  {optimized:.2f} MB")
print(f"Saved memory:      {initial - optimized:.2f} MB")
print(f"Memory budget:     24.00 MB")
print(f"Budget remaining:  {24 - optimized:.2f} MB")

if optimized <= 24:
    print("\nRESULT: BUDGET OK")
else:
    print("\nRESULT: BUDGET EXCEEDED")