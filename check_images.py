from pathlib import Path
from PIL import Image

folder = Path("dataset/banana/train")
files = list(folder.rglob("*.jpg"))

ok = 0
bad = []

for file in files:
    try:
        with Image.open(file) as image:
            image.verify()
        ok += 1
    except Exception as e:
        bad.append((file, type(e).__name__, str(e)))

print("TOTAL:", len(files))
print("OK:", ok)
print("BAD:", len(bad))

if bad:
    print("\nFirst bad image:")
    print(bad[0])
else:
    print("\nALL BANANA TRAIN IMAGES ARE VALID!")