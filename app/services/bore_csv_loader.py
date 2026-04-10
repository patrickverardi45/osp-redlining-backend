import csv
from typing import List, Dict


def load_bore_rows_csv(file_path: str) -> List[Dict]:
    """
    Load bore_rows.csv and convert into field_points format
    expected by matching_engine.
    """

    field_points: List[Dict] = []

    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, start=1):
            try:
                lat = float(row.get("lat") or row.get("latitude"))
                lon = float(row.get("lon") or row.get("longitude"))
            except (TypeError, ValueError):
                continue

            def safe_float(val):
                try:
                    return float(val)
                except:
                    return None

            def safe_int(val):
                try:
                    return int(float(val))
                except:
                    return None

            point = {
                "id": row.get("id") or f"P{i}",
                "sequence": safe_int(row.get("sequence") or i),
                "lat": lat,
                "lon": lon,
                "heading_deg": safe_float(row.get("heading") or row.get("heading_deg")),
                "depth_ft": safe_float(row.get("depth") or row.get("depth_ft")),
                "rod": safe_int(row.get("rod") or row.get("rod_number")),
                "timestamp": row.get("timestamp"),
                "source": "bore_csv",
            }

            field_points.append(point)

    return field_points