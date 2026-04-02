import os
import torch
import pandas as pd
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

# =====================================================
# Base directory (project root)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# Paths (relative)
# =====================================================
DATA_YAML = os.path.join(
    BASE_DIR,
    "data",
    "yolo_pose_dataset",
    "dial_pose.yaml"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "Results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# Check dataset
# =====================================================
info = check_det_dataset(DATA_YAML)
print(info)

if __name__ == "__main__":

    # ====================== CONFIGURATION ======================
    MODEL_NAME = "yolov8s-pose.pt"   # pretrained weight (Ultralytics auto-download)
    EPOCHS = 150
    BATCH_SIZE = 8
    IMG_SIZE = 640
    RUN_NAME = "dial_pointer_train"
    DEVICE = "0" if torch.cuda.is_available() else "cpu"

    # ====================== TRAINING ======================
    print("Starting YOLOv8-pose training...")
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=RESULTS_DIR,
        name=RUN_NAME,
        save=True,
        exist_ok=True,
        workers=0,
        patience=20,
        optimizer="Adam",
        lr0=1e-3,
        cos_lr=True,
        augment=True,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mixup=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
    )

    print("\n Training complete!")
    print(f" Results saved in: {results.save_dir}")

    # ====================== TESTING ======================
    print("\n Evaluating on test set...")

    best_model_path = os.path.join(
        results.save_dir,
        "weights",
        "best.pt"
    )

    model = YOLO(best_model_path)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMG_SIZE,
        device=DEVICE,
        save_json=True
    )

    print("\n TEST METRICS SUMMARY:")
    print(f"Box Precision:    {metrics.box.mp:.4f}")
    print(f"Box Recall:       {metrics.box.mr:.4f}")
    print(f"Box mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"Box mAP@0.5:0.95: {metrics.box.map:.4f}")

    summary_path = os.path.join(
        results.save_dir,
        "test_metrics.csv"
    )

    pd.DataFrame([{
        "Box_Precision": metrics.box.mp,
        "Box_Recall": metrics.box.mr,
        "Box_mAP@0.5": metrics.box.map50,
        "Box_mAP@0.5:0.95": metrics.box.map
    }]).to_csv(summary_path, index=False)

    print(f"Test metrics saved to: {summary_path}")

    # ====================== VISUALIZATION ======================
    print("\n Generating test visualizations...")

    model.predict(
        data=DATA_YAML,
        split="test",
        imgsz=IMG_SIZE,
        conf=0.25,
        save=True,
        save_txt=True,
        project=results.save_dir,
        name="test_predictions",
        exist_ok=True,
        device=DEVICE
    )

    print(f" Detection results saved to: {results.save_dir}/test_predictions")

    # ====================== EXPORT ======================
    print("\n Exporting model to ONNX...")
    model.export(format="onnx", dynamic=True, simplify=True)

    export_path = best_model_path.replace(".pt", ".onnx")
    print(f"Best weights: {best_model_path}")
    print(f"Exported ONNX: {export_path}")
