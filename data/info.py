import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
import os

# =====================================================
# Path configuration (relative paths)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input image (single file)
img_path = os.path.join(
    BASE_DIR,
    "yolo_resolution_test",
    "raw_image",
    "DSCF0002.JPG"
)

# =====================================================
# Read image
# =====================================================

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")

# =====================================================
# Storage for selected points
# =====================================================

points = {}
print("Left-click = dial center, Right-click = zero-reference direction")

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return

    x, y = int(event.xdata), int(event.ydata)

    # Left mouse button: select dial center
    if event.button == 1:
        points["center"] = (x, y)
        print(f"Center selected: {points['center']}")

    # Right mouse button: select zero-reference direction
    elif event.button == 3 and "center" in points:
        cx, cy = points["center"]
        dx, dy = x - cx, cy - y
        r = math.hypot(x - cx, y - cy)
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360

        points["zero_point"] = (x, y)
        points["radius"] = r
        points["zero_angle"] = angle

        # Visualization
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.plot([cx, x], [cy, y], 'r-', lw=2, label='Zero-reference direction')
        ax.scatter(*points["center"], color='yellow', s=80, label='Center')
        ax.scatter(*points["zero_point"], color='red', s=50, label='Zero point')
        ax.legend()
        ax.set_title(f"Radius = {r:.2f} px, Zero angle = {angle:.2f}°")
        plt.show()

        # Print results
        print("\nFinal results:")
        print(f"Image: {os.path.basename(img_path)}")
        print(f"Center: {points['center']}")
        print(f"Zero point: {points['zero_point']}")
        print(f"Radius: {r:.2f} px")
        print(f"Zero angle (clockwise): {angle:.2f}°")

        # Save results to Excel
        df_new = pd.DataFrame([{
            "filename": os.path.basename(img_path),
            "center_x": points["center"][0],
            "center_y": points["center"][1],
            "zero_x": points["zero_point"][0],
            "zero_y": points["zero_point"][1],
            "radius_px": round(points["radius"], 2),
            "zero_angle_deg": round(points["zero_angle"], 2)
        }])

        if os.path.exists(save_excel):
            old = pd.read_excel(save_excel)
            df_new = pd.concat([old, df_new], ignore_index=True)

        df_new.to_excel(save_excel, index=False)
        print(f"Saved to: {save_excel}")

        plt.close('all')

# =====================================================
# Start interactive selection
# =====================================================
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax.set_title("Left-click = Center, Right-click = Zero-reference direction")
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
