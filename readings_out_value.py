import os
import math
import json
import cv2
import pandas as pd
from datetime import datetime

# =====================================================
# Basic configuration
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Images to visualize / compute readings on
images_dir = os.path.join(BASE_DIR, "data", "yolo_inference_dataset")

# YOLO pose CSV (kp2 is pointer tip; kp1 is not used for computation here)
CSV_PATH = os.path.join(BASE_DIR, "Results", "yolo_inference_dataset", "pose_predictions.csv")

# Per-image CENTER / ZERO_PT / RADIUS produced by semi-auto calibration
ZERO_JSON = os.path.join(BASE_DIR, "Results", "dial_zero_tracking.json")

# Output root
output_root = os.path.join(BASE_DIR, "Results", "readings_out")

# Conversion factor: degrees per millimeter (1 full revolution = 1 mm by default)
MM_TO_DEG = 360.0 / 1.0


# =====================================================
# Utility functions
# =====================================================
def norm_angle(a: float) -> float:
    """Normalize angle to [0, 360)"""
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a


def angle_to_value_mm(angle_deg: float, zero_deg: float, mm_to_deg: float) -> float:
    """Convert angular displacement to physical displacement (mm)"""
    delta = norm_angle(angle_deg - zero_deg)
    return delta / mm_to_deg


def make_outdir(root: str) -> str:
    """Create timestamped output directory"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, f"readings_{ts}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "visuals"), exist_ok=True)
    return out


def load_pose_csv(csv_path: str):
    """
    Load YOLOv8-pose keypoint predictions from CSV.
    Uses kp2 as pointer tip. Filters out empty/NaN rows safely.
    Returns:
        dict: image_basename -> list of detections [{kp2_x, kp2_y, ...}, ...]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["image", "kp2_x", "kp2_y"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")

    # Ensure basenames for matching
    df["image_key"] = df["image"].astype(str).apply(os.path.basename)

    det_map = {}
    for img_name, g in df.groupby("image_key"):
        dets = []
        for _, row in g.iterrows():
            # Skip missing kp2 (NaN or empty)
            if pd.isna(row["kp2_x"]) or pd.isna(row["kp2_y"]):
                continue

            try:
                kp2_x = float(row["kp2_x"])
                kp2_y = float(row["kp2_y"])
            except Exception:
                continue

            # Optional sanity check: normalized coordinates should be within [0, 1]
            if not (0.0 <= kp2_x <= 1.0 and 0.0 <= kp2_y <= 1.0):
                continue

            kp1_x = None
            kp1_y = None
            if "kp1_x" in df.columns and not pd.isna(row.get("kp1_x", None)):
                try:
                    kp1_x = float(row["kp1_x"])
                except Exception:
                    kp1_x = None
            if "kp1_y" in df.columns and not pd.isna(row.get("kp1_y", None)):
                try:
                    kp1_y = float(row["kp1_y"])
                except Exception:
                    kp1_y = None

            dets.append({
                "kp2_x": kp2_x,
                "kp2_y": kp2_y,
                "kp1_x": kp1_x,
                "kp1_y": kp1_y,
            })

        det_map[img_name] = dets

    return det_map


def load_zero_json(json_path: str):
    """
    Load per-image dial calibration from dial_zero_tracking.json
    Returns:
        dict: image_basename -> {"CENTER":(x,y), "ZERO_PT":(x,y), "RADIUS":float}
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Zero JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    zmap = {}
    for item in data:
        img_name = os.path.basename(str(item.get("image", "")))
        center = item.get("CENTER", None)
        zero_pt = item.get("ZERO_PT", None)
        radius = item.get("RADIUS", None)

        if not img_name or center is None or zero_pt is None or radius is None:
            continue

        # center/zero might be lists in JSON
        try:
            cx, cy = int(center[0]), int(center[1])
            zx, zy = int(zero_pt[0]), int(zero_pt[1])
            r = float(radius)
        except Exception:
            continue

        zmap[img_name] = {
            "CENTER": (cx, cy),
            "ZERO_PT": (zx, zy),
            "RADIUS": r,
        }

    return zmap


# =====================================================
# Main processing pipeline
# =====================================================
def main():
    # Basic file existence checks
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not os.path.isfile(ZERO_JSON):
        raise FileNotFoundError(f"ZERO_JSON not found: {ZERO_JSON}")

    out_dir = make_outdir(output_root)
    vis_dir = os.path.join(out_dir, "visuals")
    rows = []

    det_map = load_pose_csv(CSV_PATH)
    zero_map = load_zero_json(ZERO_JSON)

    print(f"Loaded pose detections for {len(det_map)} images from CSV.")
    print(f"Loaded dial calibration for {len(zero_map)} images from JSON.")

    image_list = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_list:
        print("No images found in images_dir.")
        return

    skipped_no_json = 0
    skipped_no_kp2 = 0
    skipped_bad_tip = 0
    skipped_zero_vec = 0
    processed = 0

    for fname in sorted(image_list):
        # 1) Calibration for this image (CENTER / ZERO / RADIUS)
        zone = zero_map.get(fname)
        if zone is None:
            skipped_no_json += 1
            print(f" {fname}: not found in dial_zero_tracking.json. Skipped.")
            continue

        # 2) Pointer tip detection (kp2)
        dets = det_map.get(fname, [])
        if not dets:
            skipped_no_kp2 += 1
            print(f" {fname}: no valid kp2 detections in CSV. Skipped.")
            continue

        d = dets[0]  # assume single pointer per image (or take the first)

        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f" {fname}: cannot read image. Skipped.")
            continue

        h, w = img.shape[:2]

        # Dial center and zero point from JSON (pixel coords)
        cx_dial, cy_dial = zone["CENTER"]
        zx_manual, zy_manual = zone["ZERO_PT"]
        radius = zone["RADIUS"]

        # Pointer tip from kp2 (normalized -> pixel)
        tip_x = d["kp2_x"] * w
        tip_y = d["kp2_y"] * h

        # Robust check for NaN/inf tip
        if not (math.isfinite(tip_x) and math.isfinite(tip_y)):
            skipped_bad_tip += 1
            print(f" {fname}: kp2 is NaN/inf. Skipped.")
            continue

        # Compute zero direction angle (center -> ZERO_PT)
        zero_deg = norm_angle(
            math.degrees(math.atan2(zy_manual - cy_dial, zx_manual - cx_dial))
        )

        # Compute pointer angle (center -> tip)
        dx, dy = tip_x - cx_dial, tip_y - cy_dial
        if dx == 0 and dy == 0:
            skipped_zero_vec += 1
            print(f" {fname}: zero pointer vector. Skipped.")
            continue

        angle_deg = norm_angle(math.degrees(math.atan2(dy, dx)))
        value_mm = angle_to_value_mm(angle_deg, zero_deg, MM_TO_DEG)

        # ---------------- Visualization ----------------
        vis = img.copy()

        # center
        cv2.circle(vis, (int(cx_dial), int(cy_dial)), 6, (0, 0, 255), -1)

        # pointer line and tip
        cv2.line(vis, (int(cx_dial), int(cy_dial)),
                 (int(tip_x), int(tip_y)), (0, 255, 0), 2)
        cv2.circle(vis, (int(tip_x), int(tip_y)), 5, (0, 255, 255), -1)

        # zero reference direction line (uses stored radius)
        zero_rad = math.radians(zero_deg)
        zx = cx_dial + radius * math.cos(zero_rad)
        zy = cy_dial + radius * math.sin(zero_rad)
        cv2.line(vis, (int(cx_dial), int(cy_dial)),
                 (int(zx), int(zy)), (0, 165, 255), 2)
        cv2.putText(vis, "ZERO", (int(zx), int(zy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # dial circle
        if radius > 0:
            cv2.circle(vis, (int(cx_dial), int(cy_dial)), int(round(radius)),
                       (255, 0, 0), 1)

        label_text = (
            f"ang={angle_deg:.2f} deg  "
            f"zero={zero_deg:.2f} deg  "
            f"val={value_mm:.3f} mm"
        )
        cv2.putText(vis, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out_img = os.path.join(vis_dir, os.path.splitext(fname)[0] + "_vis.png")
        cv2.imwrite(out_img, vis)

        # ---------------- Save row ----------------
        rows.append({
            "filename": fname,
            "center_x_px": cx_dial,
            "center_y_px": cy_dial,
            "zero_x_px": zx_manual,
            "zero_y_px": zy_manual,
            "radius_px": radius,
            "tip_x_px": tip_x,
            "tip_y_px": tip_y,
            "zero_deg": zero_deg,
            "angle_deg": angle_deg,
            "value_mm": value_mm,
            "visual_path": out_img,
        })

        processed += 1

    # ---------------- Save outputs ----------------
    if rows:
        out_csv = os.path.join(out_dir, "dial_readings_from_kp2.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\nCompleted: {processed} results generated.")
        print(f"CSV saved to: {out_csv}")
        print(f"Visualizations saved to: {vis_dir}")
    else:
        print("No results generated.")

    # Summary of skipped reasons
    print("\nSummary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (no JSON calibration): {skipped_no_json}")
    print(f"  Skipped (no valid kp2 in CSV): {skipped_no_kp2}")
    print(f"  Skipped (kp2 NaN/inf): {skipped_bad_tip}")
    print(f"  Skipped (zero pointer vector): {skipped_zero_vec}")


if __name__ == "__main__":
    main()
