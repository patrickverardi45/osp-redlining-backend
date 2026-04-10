import csv
import os
import re

import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import ImageDraw, Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DEBUG_DIR = "debug_images"
CROP_DIR = "field_crops"
BORE_ROWS_CSV = "bore_rows.csv"

os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

Image.MAX_IMAGE_PIXELS = None


def draw_debug(img, boxes, filename):
    debug = img.copy()
    draw = ImageDraw.Draw(debug)

    for name, box in boxes.items():
        draw.rectangle(box, outline="red", width=4)
        draw.text((box[0] + 6, box[1] + 6), name, fill="red")

    path = os.path.join(DEBUG_DIR, filename)
    debug.save(path)
    return path


def save_crop(img, box, filename):
    crop = img.crop(box)
    path = os.path.join(CROP_DIR, filename)
    crop.save(path)
    return path


def normalize_station(text: str) -> str:
    text = text.strip().replace(" ", "")
    text = text.replace("O", "0").replace("o", "0")

    if "+" not in text:
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) == 4:
            return f"{digits[:2]}+{digits[2:]}"
        if len(digits) == 5:
            return f"{digits[:2]}+{digits[2:]}"
        return text

    match = re.match(r"^(\d{2})\+(\d{2,3})$", text)
    if match:
        return f"{match.group(1)}+{match.group(2)}"

    return text


def normalize_depth(text: str) -> str:
    text = text.strip().replace(",", ".")
    text = text.replace("O", "0").replace("o", "0")
    match = re.search(r"\d+(?:\.\d+)?", text)
    return match.group(0) if match else text


def normalize_boc(text: str) -> str:
    text = text.strip().replace("O", "0").replace("o", "0")
    match = re.search(r"\d+", text)
    return match.group(0) if match else text


def looks_like_station(text: str) -> bool:
    return bool(re.match(r"^\d{2}\+\d{2,3}$", text))


def looks_like_depth(text: str) -> bool:
    return bool(re.match(r"^\d+(?:\.\d+)?$", text))


def looks_like_boc(text: str) -> bool:
    return bool(re.match(r"^\d+$", text))


def save_bore_rows(rows):
    with open(BORE_ROWS_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["station", "depth", "boc"])

        for row in rows:
            writer.writerow([row["station"], row["depth"], row["boc"]])

    return BORE_ROWS_CSV


def extract_table_data(image_path):
    img = cv2.imread(image_path)

    # 1. GRAYSCALE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. STRONG CONTRAST BOOST
    gray = cv2.convertScaleAbs(gray, alpha=2.5, beta=0)

    # 3. THRESHOLD FOR HANDWRITING + TABLE
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # 4. REMOVE HORIZONTAL TABLE LINES
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

    # 5. REMOVE VERTICAL TABLE LINES
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

    lines_removed = cv2.add(horizontal_lines, vertical_lines)
    cleaned = cv2.subtract(thresh, lines_removed)

    # 6. LIGHT MORPH CLEANUP
    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cleanup_kernel)

    # 7. INVERT BACK FOR OCR
    final = cv2.bitwise_not(cleaned)

    processed_path = os.path.join(CROP_DIR, "left_table_processed.png")
    cv2.imwrite(processed_path, final)

    raw_text = pytesseract.image_to_string(
        final,
        config="--psm 6"
    )

    lines = raw_text.split("\n")
    parsed_rows = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        upper = line.upper()
        if "STATION" in upper or "DEPTH" in upper or "BOC" in upper:
            continue

        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue

        station = normalize_station(parts[0])

        # pull all numbers from the line
        nums = re.findall(r"\d+(?:\.\d+)?", line)

        if len(nums) >= 2:
            depth = normalize_depth(nums[0])
            boc = normalize_boc(nums[1])

            if looks_like_station(station) and looks_like_depth(depth) and looks_like_boc(boc):
                parsed_rows.append({
                    "station": station,
                    "depth": depth,
                    "boc": boc
                })

    deduped = []
    seen = set()
    for row in parsed_rows:
        key = (row["station"], row["depth"], row["boc"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped, raw_text, processed_path


def process_pdf(file_path: str):
    try:
        images = convert_from_path(file_path, dpi=250)
        img = images[0]
        w, h = img.size

        boxes = {
            "DATE": (
                int(w * 0.14), int(h * 0.19),
                int(w * 0.43), int(h * 0.28)
            ),
            "CREW": (
                int(w * 0.14), int(h * 0.24),
                int(w * 0.43), int(h * 0.33)
            ),
            "JOB": (
                int(w * 0.66), int(h * 0.19),
                int(w * 0.98), int(h * 0.28)
            ),
            "PRINT": (
                int(w * 0.66), int(h * 0.24),
                int(w * 0.94), int(h * 0.33)
            ),
            "LEFT_TABLE": (
                int(w * 0.06), int(h * 0.30),
                int(w * 0.38), int(h * 0.93)
            ),
        }

        debug_path = draw_debug(img, boxes, "review_boxes_with_table.png")

        date_path = save_crop(img, boxes["DATE"], "date_review.png")
        crew_path = save_crop(img, boxes["CREW"], "crew_review.png")
        job_path = save_crop(img, boxes["JOB"], "job_review.png")
        print_path = save_crop(img, boxes["PRINT"], "print_review.png")
        table_path = save_crop(img, boxes["LEFT_TABLE"], "left_table_review.png")

        parsed_rows, raw_ocr_text, processed_table_path = extract_table_data(table_path)
        csv_saved_to = save_bore_rows(parsed_rows)

        return {
            "status": "TABLE OCR COMPLETE",
            "debug_image": debug_path,
            "field_review_images": {
                "date": {"raw_image": date_path},
                "crew": {"raw_image": crew_path},
                "job": {"raw_image": job_path},
                "print": {"raw_image": print_path},
                "left_table": {
                    "raw_image": table_path,
                    "processed_image": processed_table_path
                },
            },
            "raw_ocr_text": raw_ocr_text,
            "parsed_rows": parsed_rows,
            "bore_rows_saved_to": csv_saved_to,
            "parsed_row_count": len(parsed_rows),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }