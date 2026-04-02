import cv2
import numpy as np
import os
import random


# ============================
#   Strong blur augmentation functions
# ============================

def add_motion_blur(img, degree=30, angle=None):
    """
    Apply strong motion blur to an image.

    Parameters:
        img (ndarray): Input image.
        degree (int): Length of the motion blur kernel.
        angle (int or None): Motion direction in degrees.
                             If None, a random angle in [0, 180] is used.

    Returns:
        ndarray: Motion-blurred image.
    """
    if angle is None:
        angle = random.randint(0, 180)  # Random direction to mimic real motion blur

    img = np.array(img)
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)

    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree

    blurred = cv2.filter2D(img, -1, motion_blur_kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def add_defocus_blur(img, ksize=31):
    """
    Apply strong defocus (Gaussian) blur to an image.

    Parameters:
        img (ndarray): Input image.
        ksize (int): Kernel size for Gaussian blur.

    Returns:
        ndarray: Defocus-blurred image.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


# ============================
#   Input and output paths
# ============================

# Directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relative paths
input_folder = os.path.join(BASE_DIR, "yolo_resolution_test", "Images")
output_base  = os.path.join(BASE_DIR, "yolo_resolution_test", "Images_generated_blur")

os.makedirs(output_base, exist_ok=True)

# ============================
#   Blur intensity levels
# ============================

motion_levels = [20, 35, 50, 70]      # Increasing motion blur strength
defocus_levels = [25, 35, 45, 55]     # Increasing defocus blur strength


# ============================
#   Batch processing
# ============================

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

    print(f" Processing: {file_name}")

    # Apply motion blur
    for lvl in motion_levels:
        blur_img = add_motion_blur(img, degree=lvl)
        cv2.imwrite(os.path.join(out_dir, f"motion_blur_{lvl}.jpg"), blur_img)

    # Apply defocus blur
    for k in defocus_levels:
        defocus_img = add_defocus_blur(img, ksize=k)
        cv2.imwrite(os.path.join(out_dir, f"defocus_blur_{k}.jpg"), defocus_img)

    print(f" Finished: {file_name}")

print("\n blur augmentation completed!")
