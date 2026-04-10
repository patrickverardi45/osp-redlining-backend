import csv
import io
import re
from pathlib import Path

from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["bore-rows"])

BORE_ROWS_CSV = Path("bore_rows.csv")


class BulkBoreRowsRequest(BaseModel):
    rows_text: str
    replace_existing: bool = True


def normalize_cell(value: str) -> str:
    return (value or "").strip()


def normalize_station(value: str) -> str:
    value = normalize_cell(value)
    value = value.replace(" ", "")
    value = value.replace("O", "0").replace("o", "0")

    if "+" in value:
        match = re.match(r"^(\d{2})\+(\d{2,3})$", value)
        if match:
            return f"{match.group(1)}+{match.group(2)}"
        return value

    digits = re.sub(r"[^0-9]", "", value)
    if len(digits) == 4:
        return f"{digits[:2]}+{digits[2:]}"
    if len(digits) == 5:
        return f"{digits[:2]}+{digits[2:]}"
    return value


def normalize_depth(value: str) -> str:
    value = normalize_cell(value)
    value = value.replace(",", ".")
    value = value.replace("O", "0").replace("o", "0")
    match = re.search(r"\d+(?:\.\d+)?", value)
    return match.group(0) if match else value


def normalize_boc(value: str) -> str:
    value = normalize_cell(value)
    value = value.replace("O", "0").replace("o", "0")
    match = re.search(r"\d+", value)
    return match.group(0) if match else value


def looks_like_station(value: str) -> bool:
    return bool(re.match(r"^\d{2}\+\d{2,3}$", value))


def looks_like_depth(value: str) -> bool:
    return bool(re.match(r"^\d+(?:\.\d+)?$", value))


def looks_like_boc(value: str) -> bool:
    return bool(re.match(r"^\d+$", value))


def write_rows(rows):
    with open(BORE_ROWS_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["station", "depth", "boc"])
        for row in rows:
            writer.writerow([row["station"], row["depth"], row["boc"]])


def read_existing_rows():
    if not BORE_ROWS_CSV.exists():
        return []

    rows = []
    with open(BORE_ROWS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            station = normalize_station(r.get("station", ""))
            depth = normalize_depth(r.get("depth", ""))
            boc = normalize_boc(r.get("boc", ""))
            if station or depth or boc:
                rows.append({
                    "station": station,
                    "depth": depth,
                    "boc": boc,
                })
    return rows


def dedupe_rows(rows):
    final_rows = []
    seen = set()

    for row in rows:
        key = (row["station"], row["depth"], row["boc"])
        if key not in seen:
            seen.add(key)
            final_rows.append(row)

    return final_rows


def parse_bulk_line(line: str):
    original = line
    line = normalize_cell(line)

    if not line:
        return None, "Empty line"

    upper = line.upper()
    if "STATION" in upper and "DEPTH" in upper:
        return None, "Header row skipped"

    if "\t" in line:
        parts = [p.strip() for p in line.split("\t") if p.strip()]
    elif "|" in line:
        parts = [p.strip() for p in line.split("|") if p.strip()]
    elif "," in line:
        parts = [p.strip() for p in line.split(",") if p.strip()]
    else:
        station_match = re.search(r"\d{2}\s*\+?\s*\d{2,3}", line)
        number_matches = re.findall(r"\d+(?:\.\d+)?", line)

        if not station_match or len(number_matches) < 3:
            parts = [p.strip() for p in re.split(r"\s+", line) if p.strip()]
        else:
            station = normalize_station(station_match.group(0))
            depth = normalize_depth(number_matches[-2])
            boc = normalize_boc(number_matches[-1])

            if looks_like_station(station) and looks_like_depth(depth) and looks_like_boc(boc):
                return {
                    "station": station,
                    "depth": depth,
                    "boc": boc,
                }, None

            return None, f"Could not normalize line: {original}"

    if len(parts) < 3:
        return None, f"Could not split into 3 values: {original}"

    station = normalize_station(parts[0])
    depth = normalize_depth(parts[1])
    boc = normalize_boc(parts[2])

    if not looks_like_station(station):
        return None, f"Invalid station: {parts[0]}"
    if not looks_like_depth(depth):
        return None, f"Invalid depth: {parts[1]}"
    if not looks_like_boc(boc):
        return None, f"Invalid BOC: {parts[2]}"

    return {
        "station": station,
        "depth": depth,
        "boc": boc,
    }, None


def parse_csv_text(csv_text: str):
    parsed_rows = []
    rejected_rows = []

    reader = csv.reader(io.StringIO(csv_text))
    for idx, row in enumerate(reader, start=1):
        row = [normalize_cell(col) for col in row]

        if not any(row):
            continue

        joined = " ".join(row).upper()
        if "STATION" in joined and "DEPTH" in joined:
            continue

        if len(row) < 3:
            rejected_rows.append({
                "line_number": idx,
                "line": row,
                "reason": "Expected at least 3 columns: station, depth, boc"
            })
            continue

        station = normalize_station(row[0])
        depth = normalize_depth(row[1])
        boc = normalize_boc(row[2])

        if not looks_like_station(station):
            rejected_rows.append({
                "line_number": idx,
                "line": row,
                "reason": f"Invalid station: {row[0]}"
            })
            continue

        if not looks_like_depth(depth):
            rejected_rows.append({
                "line_number": idx,
                "line": row,
                "reason": f"Invalid depth: {row[1]}"
            })
            continue

        if not looks_like_boc(boc):
            rejected_rows.append({
                "line_number": idx,
                "line": row,
                "reason": f"Invalid BOC: {row[2]}"
            })
            continue

        parsed_rows.append({
            "station": station,
            "depth": depth,
            "boc": boc,
        })

    return dedupe_rows(parsed_rows), rejected_rows


@router.post("/save-bore-row")
async def save_bore_row(
    station: str = Form(...),
    depth: str = Form(...),
    boc: str = Form(...),
):
    row = {
        "station": normalize_station(station),
        "depth": normalize_depth(depth),
        "boc": normalize_boc(boc),
    }

    rows = read_existing_rows()
    rows.append(row)
    rows = dedupe_rows(rows)
    write_rows(rows)

    return {
        "status": "bore row saved",
        "saved_to": str(BORE_ROWS_CSV),
        "row": row
    }


@router.post("/save-bore-rows-bulk")
async def save_bore_rows_bulk(payload: BulkBoreRowsRequest):
    parsed_rows = []
    rejected_lines = []

    raw_lines = payload.rows_text.splitlines()

    for idx, line in enumerate(raw_lines, start=1):
        parsed, error = parse_bulk_line(line)

        if parsed:
            parsed_rows.append(parsed)
        elif error and "Header row skipped" not in error and "Empty line" not in error:
            rejected_lines.append({
                "line_number": idx,
                "line": line,
                "reason": error
            })

    parsed_rows = dedupe_rows(parsed_rows)

    if payload.replace_existing:
        final_rows = parsed_rows
    else:
        existing = read_existing_rows()
        final_rows = dedupe_rows(existing + parsed_rows)

    write_rows(final_rows)

    return {
        "status": "bulk bore rows saved",
        "saved_to": str(BORE_ROWS_CSV),
        "replace_existing": payload.replace_existing,
        "accepted_row_count": len(parsed_rows),
        "rejected_line_count": len(rejected_lines),
        "accepted_rows": parsed_rows,
        "rejected_lines": rejected_lines,
    }


@router.post("/upload-bore-csv")
async def upload_bore_csv(
    file: UploadFile = File(...),
    replace_existing: bool = Form(True),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    contents = await file.read()

    try:
        csv_text = contents.decode("utf-8")
    except UnicodeDecodeError:
        try:
            csv_text = contents.decode("utf-8-sig")
        except UnicodeDecodeError:
            csv_text = contents.decode("latin-1")

    parsed_rows, rejected_rows = parse_csv_text(csv_text)

    if replace_existing:
        final_rows = parsed_rows
    else:
        existing = read_existing_rows()
        final_rows = dedupe_rows(existing + parsed_rows)

    write_rows(final_rows)

    return {
        "status": "bore csv uploaded",
        "filename": file.filename,
        "saved_to": str(BORE_ROWS_CSV),
        "replace_existing": replace_existing,
        "accepted_row_count": len(parsed_rows),
        "rejected_row_count": len(rejected_rows),
        "accepted_rows": parsed_rows,
        "rejected_rows": rejected_rows,
    }