import os
import re
import cv2
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import pytesseract

BASE_DIR = Path(__file__).resolve().parent
pytesseract.pytesseract.tesseract_cmd = str(BASE_DIR / "Tesseract-OCR" / "tesseract.exe")

input_folder = BASE_DIR / "data" / "raw_image"
base_output_folder = BASE_DIR / "Results"

run_folder_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
output_folder = base_output_folder / run_folder_name
output_folder.mkdir(parents=True, exist_ok=False)

roi_folder = output_folder / "roi"
annotated_folder = output_folder / "annotated"
roi_folder.mkdir(exist_ok=True)
annotated_folder.mkdir(exist_ok=True)

csv_path = output_folder / "temperature_datetime_all.csv"
excel_path = output_folder / "temperature_datetime_all_fixed.xlsx"

INTERVAL = timedelta(minutes=30)
START_TIME = datetime(2022, 1, 12, 11, 48)
START_ID = 2
END_ID = 6

total_images = 0
missing_temp = 0
missing_date = 0


def extract_num_from_filename(name: str):
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else None


def extract_info(img, x_ratio, save_debug=False, stem=""):
    h, w = img.shape[:2]

    # OCR crop region:
    # x starts from x_ratio * image width
    # y starts from 85% of image height
    # cropped area = bottom-right part of the image
    x1 = int(w * x_ratio)
    y1 = int(h * 0.85)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))

    roi = img[y1:h, x1:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=15)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if save_debug and stem:
        cv2.imwrite(str(roi_folder / f"{stem}_x{int(x_ratio*100)}_roi.png"), roi)
        cv2.imwrite(str(roi_folder / f"{stem}_x{int(x_ratio*100)}_roi_bw.png"), bw)

    data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT)
    words = [data["text"][i].strip() for i in range(len(data["text"])) if data["text"][i].strip()]
    full_text = " ".join(words)

    return full_text, (x1, y1, w, h), bw


def annotate_preview(img, crop_coords, text, out_path, x_ratio):
    x1, y1, w, h = crop_coords
    vis = img.copy()

    # Draw OCR region
    cv2.rectangle(vis, (x1, y1), (w - 1, h - 1), (0, 255, 0), 2)

    # Mark ROI start point
    cv2.circle(vis, (x1, y1), 8, (0, 0, 255), -1)
    cv2.putText(
        vis,
        f"ROI start ({x_ratio:.2f}W, 0.85H)",
        (x1 + 10, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # Label the OCR region
    cv2.putText(
        vis,
        "OCR region",
        (x1 + 10, min(h - 10, y1 + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Show OCR text on top-left
    disp = text[:70] + ("..." if len(text) > 70 else "")
    cv2.rectangle(vis, (10, 10), (10 + 9 * len(disp), 45), (0, 0, 0), -1)
    cv2.putText(
        vis,
        disp,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(out_path), vis)


x_ratios = [0.30, 0.35, 0.40]
results = []

files = sorted(os.listdir(input_folder))
for file_name in tqdm(files, desc="Processing images"):
    if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    total_images += 1
    img_path = input_folder / file_name
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot read image: {img_path}")
        continue

    stem = Path(file_name).stem
    num = extract_num_from_filename(file_name)

    temp_c = None
    datetime_str = None
    used_ratio = None
    full_text_used = ""
    crop_used = None

    for x_ratio in x_ratios:
        full_text, crop_coords, bw = extract_info(img, x_ratio, save_debug=True, stem=stem)

        temp_match = re.search(r"([-+]?\d+)\s*°?\s*[Cc]\b", full_text)
        temp_c_try = int(temp_match.group(1)) if temp_match else None

        datetime_match = re.search(
            r"\b\d{4}[/-]\d{2}[/-]\d{2}\s*\d{2}[:：]\d{2}(?:[:：]\d{2})?\b",
            full_text,
        )
        datetime_str_try = datetime_match.group(0) if datetime_match else None

        if temp_c_try is not None and datetime_str_try is not None:
            temp_c = temp_c_try
            datetime_str = datetime_str_try
            used_ratio = x_ratio
            full_text_used = full_text
            crop_used = crop_coords

            annotate_preview(
                img,
                crop_used,
                full_text_used,
                annotated_folder / f"{stem}_ok.png",
                x_ratio,
            )
            break

        if temp_c is None and temp_c_try is not None:
            temp_c = temp_c_try
            used_ratio = x_ratio
            full_text_used = full_text
            crop_used = crop_coords

    if temp_c is None:
        missing_temp += 1
    if datetime_str is None:
        missing_date += 1

    results.append({
        "filename": file_name,
        "num": num,
        "x_ratio_used": used_ratio,
        "temperature_C": temp_c,
        "datetime_ocr": datetime_str,
        "ocr_text_sample": full_text_used[:120],
    })

df = pd.DataFrame(results)

all_ids = list(range(START_ID, END_ID + 1))
full = pd.DataFrame({
    "num": all_ids,
    "filename": [f"DSCF{n:04d}.JPG" for n in all_ids],
})

full["datetime_full"] = [START_TIME + (i * INTERVAL) for i in range(len(full))]

merged = pd.merge(full, df.drop(columns=["filename"], errors="ignore"), on="num", how="left")
merged["datetime_ocr_dt"] = pd.to_datetime(merged["datetime_ocr"], errors="coerce")
merged["datetime_filled"] = merged["datetime_ocr_dt"].fillna(merged["datetime_full"])

existing_nums = set(df["num"].dropna().astype(int))
missing_nums = sorted(set(all_ids) - existing_nums)

merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
merged.to_excel(excel_path, index=False)

print("\n=== OCR Summary Report ===")
print(f"Total images processed: {total_images}")
if total_images > 0:
    print(f"Missing temperature: {missing_temp} ({missing_temp/total_images*100:.1f}%)")
    print(f"Missing datetime (OCR): {missing_date} ({missing_date/total_images*100:.1f}%)")
print("===============================")
print(f"Fixed CSV saved to: {csv_path}")
print(f"Fixed Excel saved to: {excel_path}")
if missing_nums:
    print(f"Missing image nums in folder: {len(missing_nums)} (e.g. first 20) {missing_nums[:20]}")