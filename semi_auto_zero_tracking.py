import os
import cv2
import json
import math

# ===============================
# Config
# ===============================
IMG_DIR = r"./data/yolo_inference_dataset"
OUT_JSON = "./Results/dial_zero_tracking.json"
IMG_EXTS = (".jpg", ".jpeg", ".png")

# Max display size (keeps aspect ratio)
MAX_W = 1600
MAX_H = 900

# ===============================
# Globals
# ===============================
clicks = []
img_show = None
display_scale = 1.0

def show_image_keep_ratio(win_name, img, max_w=MAX_W, max_h=MAX_H):
    """
    Display image with preserved aspect ratio.
    Returns the scale factor used (display = original * scale).
    """
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)  # never upscale beyond 1.0
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imshow(win_name, resized)
    return scale


def mouse_cb(event, x, y, flags, param):
    """
    Mouse callback for clicking points.
    NOTE: x,y are in displayed image coordinates -> convert back to original coords.
    """
    global clicks, img_show, display_scale

    if event == cv2.EVENT_LBUTTONDOWN:
        ox = int(round(x / display_scale))
        oy = int(round(y / display_scale))
        clicks.append((ox, oy))

        # draw on original image and re-display
        cv2.circle(img_show, (ox, oy), 6, (0, 0, 255), -1, cv2.LINE_AA)
        display_scale = show_image_keep_ratio("Review", img_show)

        print(f"Click {len(clicks)}: ({ox}, {oy})")


def manual_calibrate(img):
    """
    Manually pick CENTER and ZERO_PT on the given image.
    """
    global clicks, img_show, display_scale

    clicks = []
    img_show = img.copy()

    cv2.namedWindow("Review", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Review", mouse_cb)

    display_scale = show_image_keep_ratio("Review", img_show)

    print("  Click CENTER, then ZERO_PT. Press 'n' to confirm.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("n") and len(clicks) == 2:
            break

    cv2.destroyAllWindows()
    return clicks[0], clicks[1]


def draw_reference(img, center, zero):
    """
    Draw current reference points on a copy of img.
    """
    vis = img.copy()
    cv2.circle(vis, center, 8, (0, 255, 0), 2, cv2.LINE_AA)     # center (green ring)
    cv2.circle(vis, zero, 8, (0, 0, 255), -1, cv2.LINE_AA)      # zero (red filled)
    cv2.line(vis, center, zero, (255, 255, 0), 2, cv2.LINE_AA)  # line (cyan/yellow)
    return vis


def main():
    imgs = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(IMG_EXTS)])
    if not imgs:
        print("No images found.")
        return

    results = []

    # -------- Step 1: first image (must calibrate) --------
    first_path = os.path.join(IMG_DIR, imgs[0])
    first_img = cv2.imread(first_path)
    if first_img is None:
        print(f"Cannot read: {first_path}")
        return

    print(f"\n[1/{len(imgs)}] {imgs[0]} (initial calibration required)")
    center, zero = manual_calibrate(first_img)
    radius = math.dist(center, zero)

    results.append({
        "image": imgs[0],
        "CENTER": center,
        "ZERO_PT": zero,
        "RADIUS": round(radius, 2),
        "method": "manual_init"
    })

    # -------- Step 2: iterate remaining images --------
    for idx, name in enumerate(imgs[1:], start=2):
        img_path = os.path.join(IMG_DIR, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read: {img_path}")
            continue

        print(f"\n[{idx}/{len(imgs)}] {name}")
        print("  Press [n] accept  |  [r] recalibrate  |  [q] quit")

        # show current reference overlay (keeps aspect ratio)
        vis = draw_reference(img, center, zero)
        cv2.namedWindow("Review", cv2.WINDOW_AUTOSIZE)
        # no callback here; it's a review screen
        _ = show_image_keep_ratio("Review", vis)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord("q"):
            break

        if key == ord("r"):
            print("  Recalibrating...")
            center, zero = manual_calibrate(img)

        # save current (possibly updated) reference for this image
        radius = math.dist(center, zero)
        results.append({
            "image": name,
            "CENTER": center,
            "ZERO_PT": zero,
            "RADIUS": round(radius, 2),
            "method": "manual_update" if key == ord("r") else "propagated"
        })

    # -------- Save --------
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {OUT_JSON}")


if __name__ == "__main__":
    main()
