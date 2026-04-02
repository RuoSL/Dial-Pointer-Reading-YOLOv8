import cv2
import numpy as np
import os

def add_strong_fog(img, beta=0.2, A=255, contrast_factor=0.7):
    """
    Strong fog simulation based on a simplified atmospheric scattering model.
    """
    img = img.astype(np.float32)

    # Transmission (simplified as a constant)
    t = np.exp(-beta)

    # Atmospheric scattering model
    foggy = img * t + A * (1 - t)

    # Contrast compression to mimic visibility loss
    foggy = (foggy - 128) * contrast_factor + 128
    foggy = np.clip(foggy, 0, 255).astype(np.uint8)
    return foggy

# =====================================================
#   Relative paths (based on current script location)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input folder (clear images)
input_folder = os.path.join(
    BASE_DIR,
    "yolo_resolution_test",
    "images"
)

# Output root directory
output_base = os.path.join(
    BASE_DIR,
    "yolo_resolution_test",
    "Images_generated_fog"
)
os.makedirs(output_base, exist_ok=True)


# Fog levels (including the original image)
betas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# =====================================================
#   Batch processing
# =====================================================

for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_folder, file_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠ Unable to read image: {img_path}")
        continue

    img_name = os.path.splitext(file_name)[0]
    out_dir = os.path.join(output_base, img_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f" Processing: {file_name} → {out_dir}")

    for b in betas:
        if b == 0:
            fog_img = img
        else:
            fog_img = add_strong_fog(
                img,
                beta=b,
                A=255,
                contrast_factor=0.55
            )

        save_name = f"fog_beta_{b:.2f}.jpg"
        cv2.imwrite(os.path.join(out_dir, save_name), fog_img)

    print(f" Finished: {file_name}")

print("\n Batch fog generation completed!")
