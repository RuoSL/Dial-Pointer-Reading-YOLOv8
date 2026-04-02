import os
from PIL import Image

# ====== Configuration ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(
    BASE_DIR,
    "yolo_resolution_test",
    "raw_image")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "yolo_resolution_test",
    "raw_image_crops")
os.makedirs(OUTPUT_DIR, exist_ok=True)

x0, y0 = 2867, 2600   # Top-left corner of the crop window
CROP = 1280           # Crop size (square)
exs = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ====== Batch processing ======
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(exs):
        continue

    in_path  = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, fname)

    img = Image.open(in_path).convert("RGB")
    w, h = img.size

    # Bottom-right corner of the crop window
    x1, y1 = x0 + CROP, y0 + CROP

    # Intersection with image boundaries (to avoid out-of-bounds)
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(w, x1)
    iy1 = min(h, y1)

    # Crop the intersecting region from the original image
    crop = img.crop((ix0, iy0, ix1, iy1))

    # If the crop exceeds image boundaries, pad with a black canvas to 1280×1280
    if (ix1 - ix0) != CROP or (iy1 - iy0) != CROP:
        canvas = Image.new("RGB", (CROP, CROP), (0, 0, 0))
        # Compute paste offset (keeping the top-left reference at (x0, y0))
        dx = ix0 - x0  # Negative value indicates left overflow
        dy = iy0 - y0  # Negative value indicates top overflow
        canvas.paste(crop, (max(0, -dx), max(0, -dy)))
        crop = canvas

    crop.save(out_path, quality=95)
    print("saved:", out_path)

print("Crops saved to:", OUTPUT_DIR)
