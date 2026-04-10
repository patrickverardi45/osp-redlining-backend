from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

STATION_MAP = str.maketrans({
    "I": "1", "l": "1", "|": "1", "!": "1", "/": "1", "\\": "1", "t": "1", "T": "1", "i": "1",
    "O": "0", "o": "0", "D": "0", "Q": "0", "U": "0", "u": "0", "C": "0",
    "S": "5", "s": "5", "$": "5",
    "g": "9", "q": "9", "G": "6", "b": "6",
    "z": "2", "Z": "2", "a": "2", "A": "4",
    "x": "+", "X": "+", "h": "+", "H": "+", "k": "+", "K": "+", "*": "+", "=": "+", "#": "+",
    ",": "", ".": "", " ": "",
})
NUMERIC_MAP = str.maketrans({
    "I": "1", "l": "1", "|": "1", "!": "1", "/": "1", "\\": "1", "t": "1", "T": "1", "i": "1",
    "O": "0", "o": "0", "D": "0", "Q": "0", "U": "0", "u": "0", "C": "0",
    "S": "5", "s": "5", "$": "5",
    "g": "9", "q": "9", "G": "6", "b": "6",
    "z": "2", "Z": "2", "a": "2", "A": "4",
    ",": ".", ";": ".", ":": ".", " ": "",
})


def ensure_tesseract() -> Optional[str]:
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        getattr(pytesseract.pytesseract, "tesseract_cmd", None),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
    return None


def load_pages(file_bytes: bytes, filename: str) -> List[np.ndarray]:
    suffix = Path(filename or "upload").suffix.lower()
    if suffix == ".pdf":
        return [cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY) for page in convert_from_bytes(file_bytes, dpi=220)]
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    return [np.array(image)]


def group_centers(projection: np.ndarray, ratio: float, gap: int = 3) -> List[int]:
    if projection.size == 0 or float(projection.max()) <= 0:
        return []
    idx = np.where(projection > float(projection.max()) * ratio)[0]
    if len(idx) == 0:
        return []
    groups: List[Tuple[int, int]] = []
    start = prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value <= prev + gap:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return [(a + b) // 2 for a, b in groups]


def cluster_positions(values: Sequence[int], max_gap: int) -> List[int]:
    clustered: List[List[int]] = []
    for value in values:
        value = int(value)
        if not clustered or value - clustered[-1][-1] > max_gap:
            clustered.append([value])
        else:
            clustered[-1].append(value)
    return [int(round(sum(bucket) / len(bucket))) for bucket in clustered]


def detect_tables(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    h, w = gray.shape[:2]
    hor = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(80, w // 20), 1)))
    ver = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(80, h // 20))))
    grid = cv2.add(hor, ver)
    count, _, stats, _ = cv2.connectedComponentsWithStats(grid, 8)
    boxes: List[Tuple[int, int, int, int]] = []
    for idx in range(1, count):
        x, y, ww, hh, _area = stats[idx]
        if 250 < ww < 800 and 800 < hh < 2200:
            boxes.append((int(x), int(y), int(ww), int(hh)))
    boxes.sort(key=lambda item: (item[0], item[1]))
    return boxes[:3]


def detect_grid(table_gray: np.ndarray) -> Tuple[List[int], List[int]]:
    inv = cv2.threshold(table_gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    h, w = table_gray.shape[:2]
    hor = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 5), 1)))
    ver = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 10))))
    ys = group_centers(hor.sum(axis=1), ratio=0.35)
    xs = cluster_positions(group_centers(ver.sum(axis=0), ratio=0.45), max_gap=45)
    return xs[:4], ys


def trim_to_ink(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return gray
    mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
    points = cv2.findNonZero(mask)
    if points is None:
        return gray
    x, y, w, h = cv2.boundingRect(points)
    return gray[max(0, y - 1): y + h + 1, max(0, x - 1): x + w + 1]


def remove_strong_lines(gray: np.ndarray) -> np.ndarray:
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ver = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, gray.shape[0] // 2))))
    hor = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, gray.shape[1] // 3), 1)))
    mask = cv2.bitwise_or(ver, hor)
    cleaned = gray.copy()
    cleaned[mask > 0] = 255
    return cleaned


def build_variants(gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    base = gray.copy()
    if base.size == 0:
        return []
    base[:, :2] = 255
    base[:, -2:] = 255
    base[:2, :] = 255
    base[-2:, :] = 255
    cleaned = remove_strong_lines(base)
    variants: List[Tuple[str, np.ndarray]] = []
    for name, img in [
        ("gray", cleaned),
        ("contrast", cv2.convertScaleAbs(cleaned, alpha=1.85, beta=0)),
        ("otsu", cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("adaptive", cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)),
    ]:
        cropped = trim_to_ink(img)
        if cropped.size == 0:
            continue
        for scale in (3, 4):
            up = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            up = cv2.copyMakeBorder(up, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
            variants.append((f"{name}_x{scale}", up))
    return variants


def normalize_station(raw: str) -> Tuple[str, bool, str]:
    text = (raw or "").translate(STATION_MAP)
    text = re.sub(r"[^0-9+]", "", text)
    if not text:
        return "", False, "no station-like chars"
    if "+" not in text:
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 3:
            text = f"{digits[:-2]}+{digits[-2:]}"
    match = re.fullmatch(r"(\d{1,3})\+(\d{1,2})", text)
    if not match:
        return text, False, "station regex failed"
    left, right = match.groups()
    return f"{int(left)}+{int(right):02d}", True, "ok"


def normalize_depth(raw: str) -> Tuple[str, bool, str]:
    text = (raw or "").translate(NUMERIC_MAP)
    text = re.sub(r"[^0-9.]", "", text)
    if not text:
        return "", False, "no numeric chars"
    match = re.search(r"\d+(?:\.\d)?", text)
    if not match:
        return text, False, "depth parse failed"
    value_text = match.group(0)
    if "." not in value_text and len(value_text) >= 2:
        value_text = value_text[0] + "." + value_text[1]
    try:
        value = float(value_text)
    except Exception:
        return value_text, False, "depth float failed"
    if not (0.5 <= value <= 15.0):
        return f"{value:.1f}", False, "depth out of range"
    return f"{value:.1f}", True, "ok"


def normalize_boc(raw: str) -> Tuple[str, bool, str]:
    digits = re.sub(r"\D", "", (raw or "").translate(NUMERIC_MAP))
    if not digits:
        return "", False, "no boc digits"
    value = int(digits[0])
    if not (0 <= value <= 20):
        return str(value), False, "boc out of range"
    return str(value), True, "ok"


def ocr_cell(gray: np.ndarray, field: str, timeout_s: int = 2) -> Dict[str, Any]:
    if gray.size == 0:
        return {"raw": "", "cleaned": "", "valid": False, "reason": "empty crop", "variant": "none", "attempts": []}
    ink = int((255 - gray).sum())
    if ink < 2500:
        return {"raw": "", "cleaned": "", "valid": False, "reason": "blank cell", "variant": "none", "attempts": []}

    normalizer = {
        "station": normalize_station,
        "depth": normalize_depth,
        "boc": normalize_boc,
    }[field]
    configs = {
        "station": ["--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789+", "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789+"],
        "depth": ["--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.", "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789."],
        "boc": ["--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789", "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789"],
    }[field]

    attempts: List[Dict[str, Any]] = []
    best = {"raw": "", "cleaned": "", "valid": False, "reason": "no attempts", "variant": "none", "attempts": attempts}
    for variant_name, variant_img in build_variants(gray)[:4]:
        for cfg in configs:
            try:
                raw = pytesseract.image_to_string(variant_img, config=cfg, timeout=timeout_s).strip()
            except Exception as exc:
                raw = ""
                cleaned, valid, reason = "", False, f"ocr timeout/error: {type(exc).__name__}"
            else:
                cleaned, valid, reason = normalizer(raw)
            attempt = {
                "variant": variant_name,
                "config": cfg,
                "raw": raw,
                "cleaned": cleaned,
                "valid": valid,
                "reason": reason,
            }
            attempts.append(attempt)
            if valid and not best["valid"]:
                best = {**attempt, "attempts": attempts}
                return best
            if len(cleaned) > len(best.get("cleaned", "")):
                best = {**attempt, "attempts": attempts}
    return best


def station_to_int(station: str) -> Optional[int]:
    match = re.fullmatch(r"(\d{1,3})\+(\d{2})", station or "")
    if not match:
        return None
    return int(match.group(1)) * 100 + int(match.group(2))


def infer_station_sequence(rows: List[Dict[str, Any]]) -> None:
    stations = [station_to_int(row.get("station_cleaned", "")) for row in rows]
    valid = [value for value in stations if value is not None]
    if len(valid) < 2:
        return
    diffs = [abs(valid[i] - valid[i - 1]) for i in range(1, len(valid)) if abs(valid[i] - valid[i - 1]) > 0]
    if not diffs:
        return
    step = min(diffs, key=lambda value: abs(value - 50))
    for idx, value in enumerate(stations):
        if value is not None:
            continue
        prev_idx = next((j for j in range(idx - 1, -1, -1) if stations[j] is not None), None)
        next_idx = next((j for j in range(idx + 1, len(stations)) if stations[j] is not None), None)
        guess: Optional[int] = None
        if prev_idx is not None and next_idx is not None:
            gap = next_idx - prev_idx
            if gap > 0:
                delta = stations[next_idx] - stations[prev_idx]
                guess = stations[prev_idx] + int(round(delta / gap)) * (idx - prev_idx)
        elif prev_idx is not None:
            guess = stations[prev_idx] - step
        elif next_idx is not None:
            guess = stations[next_idx] + step
        if guess is None:
            continue
        rows[idx]["station_cleaned"] = f"{guess // 100}+{guess % 100:02d}"
        rows[idx]["station_valid"] = True
        rows[idx]["station_reason"] = "inferred from neighboring rows"
        stations[idx] = guess


def build_segments(rows: List[Dict[str, Any]], source_file: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    infer_station_sequence(rows)
    segments: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    working = [row for row in rows if row.get("station_cleaned") and row.get("depth_cleaned") and row.get("boc_cleaned")]
    for idx in range(len(working) - 1):
        current = working[idx]
        nxt = working[idx + 1]
        current_int = station_to_int(current.get("station_cleaned", ""))
        next_int = station_to_int(nxt.get("station_cleaned", ""))
        if current_int is None or next_int is None:
            invalid.append({**current, "rejection_reason": "missing normalized station pair"})
            continue
        start_int = min(current_int, next_int)
        end_int = max(current_int, next_int)
        if start_int == end_int:
            invalid.append({**current, "rejection_reason": "start_station >= end_station"})
            continue
        segments.append(
            {
                "start_station": f"{start_int // 100}+{start_int % 100:02d}",
                "end_station": f"{end_int // 100}+{end_int % 100:02d}",
                "reason": current.get("notes_cleaned", "") or "",
                "source_file": source_file,
                "confidence": 1.0 if current.get("station_valid") and current.get("depth_valid") and current.get("boc_valid") else 0.65,
                "station": current.get("station_cleaned", ""),
                "depth_ft": float(current.get("depth_cleaned")) if current.get("depth_cleaned") else None,
                "boc_ft": float(current.get("boc_cleaned")) if current.get("boc_cleaned") else None,
                "notes": current.get("notes_cleaned", "") or "",
            }
        )
    return segments, invalid


def process_uploaded_bore_logs(files: Sequence[Tuple[str, bytes]], output_root: str) -> Dict[str, Any]:
    tesseract_cmd = ensure_tesseract()
    if not tesseract_cmd:
        raise RuntimeError("tesseract is not installed or not in PATH")

    debug_root = os.path.join(output_root, "bore_ocr_debug")
    os.makedirs(debug_root, exist_ok=True)

    extracted_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    file_summary: List[Dict[str, Any]] = []

    for filename, file_bytes in files:
        pages = load_pages(file_bytes, filename)
        file_segments = 0
        file_invalid = 0
        for page_index, gray in enumerate(pages, start=1):
            tables = detect_tables(gray)
            if not tables:
                invalid_rows.append({"source_file": filename, "source_page": page_index, "rejection_reason": "no bore-log table detected"})
                file_invalid += 1
                continue
            for table_index, (x, y, w, h) in enumerate(tables, start=1):
                table_gray = gray[y:y + h, x:x + w]
                if int((255 - table_gray).sum()) < 2500000:
                    invalid_rows.append({"source_file": filename, "source_page": page_index, "table_index": table_index, "rejection_reason": "table looked blank"})
                    file_invalid += 1
                    continue
                xs, ys = detect_grid(table_gray)
                if len(xs) < 4 or len(ys) < 5:
                    invalid_rows.append({"source_file": filename, "source_page": page_index, "table_index": table_index, "rejection_reason": "table grid detection failed"})
                    file_invalid += 1
                    continue
                table_debug = os.path.join(debug_root, Path(filename).stem, f"page_{page_index}", f"table_{table_index}")
                os.makedirs(table_debug, exist_ok=True)
                cv2.imwrite(os.path.join(table_debug, "table.png"), table_gray)
                table_rows: List[Dict[str, Any]] = []
                blank_streak = 0
                for row_index in range(1, min(len(ys) - 1, 24)):
                    y1 = ys[row_index] + 3
                    y2 = ys[row_index + 1] - 3
                    if y2 <= y1 or (y2 - y1) < 10:
                        continue
                    row_band = table_gray[y1:y2, xs[0]:xs[-1]]
                    if int((255 - row_band).sum()) < 70000:
                        blank_streak += 1
                        if blank_streak >= 4 and table_rows:
                            break
                        continue
                    blank_streak = 0
                    row_record: Dict[str, Any] = {
                        "source_file": filename,
                        "source_page": page_index,
                        "table_index": table_index,
                        "row_index": row_index,
                        "notes_raw": "",
                        "notes_cleaned": "",
                    }
                    row_dir = os.path.join(table_debug, f"row_{row_index:02d}")
                    os.makedirs(row_dir, exist_ok=True)
                    for col_index, field in enumerate(["station", "depth", "boc"]):
                        x1 = xs[col_index] + 5
                        x2 = xs[col_index + 1] - 5
                        cell = table_gray[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(row_dir, f"{field}_raw.png"), cell)
                        result = ocr_cell(cell, field)
                        row_record[f"{field}_raw"] = result["raw"]
                        row_record[f"{field}_cleaned"] = result["cleaned"]
                        row_record[f"{field}_valid"] = result["valid"]
                        row_record[f"{field}_reason"] = result["reason"]
                        row_record[f"best_{field}_variant"] = result["variant"]
                        row_record[f"{field}_attempts"] = json.dumps(result["attempts"], ensure_ascii=False)
                    if row_record.get("station_raw") or row_record.get("depth_raw") or row_record.get("boc_raw"):
                        table_rows.append(row_record)
                    else:
                        invalid_rows.append({**row_record, "rejection_reason": "blank OCR row"})
                        file_invalid += 1
                segments, segment_invalid = build_segments(table_rows, filename)
                extracted_rows.extend(segments)
                invalid_rows.extend(segment_invalid)
                file_segments += len(segments)
                file_invalid += len(segment_invalid)
        file_summary.append({"source_file": filename, "valid_segments": file_segments, "invalid_rows": file_invalid})

    return {
        "rows": extracted_rows,
        "invalid_rows": invalid_rows,
        "summary": file_summary,
        "debug_root": debug_root,
    }
