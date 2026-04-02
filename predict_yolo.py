import os
import csv
import cv2
from ultralytics import YOLO

# =====================================================
# Path configuration (relative to this script)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Trained YOLOv8-pose model
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# Input folder: predict ALL images directly inside this folder (no subfolders)
INPUT_DIR = os.path.join(BASE_DIR, "data", "yolo_inference_dataset")

# Output folder: save visualizations + CSV here
OUTPUT_DIR = os.path.join(BASE_DIR, "Results", "yolo_inference_dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_EXTS = (".jpg", ".jpeg", ".png")


def run_yolo_on_folder(model, input_dir, output_dir):
    """
    Run YOLOv8-pose inference on all images within input_dir (first level only),
    save visualizations into output_dir, and write predictions to a single CSV.
    """
    print(f"\nProcessing images in: {input_dir}")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"INPUT_DIR not found: {input_dir}")

    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(IMG_EXTS)]
    print(f"Found {len(img_files)} images.")

    if len(img_files) == 0:
        print("⚠ No images found. Nothing to do.")
        return

    csv_path = os.path.join(output_dir, "pose_predictions.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "image",
            "class",
            "x_center", "y_center", "bbox_w", "bbox_h",
            "kp1_x", "kp1_y", "kp2_x", "kp2_y"
        ])

        for img_file in img_files:
            img_path = os.path.join(input_dir, img_file)

            # ---- YOLO inference ----
            results = model.predict(
                source=img_path,
                imgsz=640,
                conf=0.25,
                iou=0.6,
                save=False,
                verbose=False
            )

            im = cv2.imread(img_path)
            if im is None:
                print(f"⚠ Cannot read image: {img_path}")
                continue

            h, w = im.shape[:2]
            has_det = False

            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                boxes = r.boxes.xywh.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)

                # pose model normally has keypoints; keep safe checks anyway
                kpts = None
                if r.keypoints is not None and r.keypoints.xy is not None:
                    kpts = r.keypoints.xy.cpu().numpy()

                for i in range(len(boxes)):
                    has_det = True
                    x_c, y_c, bw, bh = boxes[i]
                    x1, y1 = int(x_c - bw / 2), int(y_c - bh / 2)
                    x2, y2 = int(x_c + bw / 2), int(y_c + bh / 2)

                    label = r.names[cls_ids[i]]

                    # Draw bbox + label
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        im, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

                    # Default NaNs for keypoints if unavailable
                    kp1_x = kp1_y = kp2_x = kp2_y = float("nan")
                    if kpts is not None and i < len(kpts) and len(kpts[i]) >= 2:
                        kp1_x, kp1_y = kpts[i][0]
                        kp2_x, kp2_y = kpts[i][1]

                        # Keypoints + line
                        cv2.circle(im, (int(kp1_x), int(kp1_y)), 8, (0, 0, 255), -1)
                        cv2.circle(im, (int(kp2_x), int(kp2_y)), 8, (255, 0, 0), -1)
                        cv2.line(
                            im,
                            (int(kp1_x), int(kp1_y)),
                            (int(kp2_x), int(kp2_y)),
                            (255, 255, 0), 3
                        )

                    # Write normalized values to CSV
                    writer.writerow([
                        img_file, label,
                        f"{x_c / w:.6f}", f"{y_c / h:.6f}",
                        f"{bw / w:.6f}", f"{bh / h:.6f}",
                        f"{kp1_x / w:.6f}" if kp1_x == kp1_x else "",
                        f"{kp1_y / h:.6f}" if kp1_y == kp1_y else "",
                        f"{kp2_x / w:.6f}" if kp2_x == kp2_x else "",
                        f"{kp2_y / h:.6f}" if kp2_y == kp2_y else "",
                    ])

            # If no detections, still write one empty row for traceability
            if not has_det:
                writer.writerow([img_file, "", "", "", "", "", "", "", "", ""])

            # Save visualization image
            save_path = os.path.join(output_dir, img_file)
            cv2.imwrite(save_path, im)
            print(f"✓ Saved visualization: {save_path}")

    print(f"\nCSV saved at: {csv_path}")


def main():
    print("Loading YOLOv8-pose model...")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    print("Running predictions...")
    run_yolo_on_folder(model, INPUT_DIR, OUTPUT_DIR)

    print("\nAll images processed successfully!")


if __name__ == "__main__":
    main()
