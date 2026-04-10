from __future__ import annotations

import csv
import io
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core import pipeline_state
from app.services.matching_engine import match_field_points, preprocess_route
from app.services.pdf_report import build_client_ready_pdf

router = APIRouter(prefix="/api", tags=["reporting"])


FLAG_LABELS = {
    "SHALLOW_DEPTH": "Shallow Depth",
    "DEPTH_JUMP": "Depth Change",
    "STATION_GAP": "Station Gap",
    "BOC_JUMP": "Rod Change",
    "HIGH_RISK": "High Risk",
    "OFF_ROUTE": "Off Route",
    "WRONG_STREET": "Wrong Street",
    "OUT_OF_SEQUENCE": "Out of Sequence",
}


def _pretty_flag(flag: str) -> str:
    return FLAG_LABELS.get(flag, flag.replace("_", " ").title())


def _get_route_coords():
    raw = getattr(pipeline_state, "CURRENT_ROUTE", [])

    coords = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lon, lat = item[0], item[1]
            coords.append({"lat": float(lat), "lon": float(lon)})

    return coords


def _parse_station_to_feet(station_text: str) -> float | None:
    if not station_text:
        return None

    text = str(station_text).strip()

    if "+" not in text:
        try:
            return float(text)
        except ValueError:
            return None

    left, right = text.split("+", 1)

    try:
        left_num = int(left.strip())
        right_num = int(right.strip())
        return float(left_num * 100 + right_num)
    except ValueError:
        return None


def _point_on_route_at_progress(route_coords: list[dict], progress_ft: float) -> dict | None:
    if not route_coords or len(route_coords) < 2:
        return None

    route = preprocess_route(route_coords)
    segments = route["segments"]
    route_length_ft = route["route_length_ft"]

    if route_length_ft <= 0:
        return None

    progress_ft = max(0.0, min(progress_ft, route_length_ft))

    for seg in segments:
        if seg.cumulative_start_ft <= progress_ft <= seg.cumulative_end_ft:
            seg_length = max(seg.length_ft, 0.0001)
            local_ft = progress_ft - seg.cumulative_start_ft
            t = local_ft / seg_length

            lat = seg.start_lat + ((seg.end_lat - seg.start_lat) * t)
            lon = seg.start_lon + ((seg.end_lon - seg.start_lon) * t)

            return {
                "lat": lat,
                "lon": lon,
                "segment_index": seg.index,
            }

    last_seg = segments[-1]
    return {
        "lat": last_seg.end_lat,
        "lon": last_seg.end_lon,
        "segment_index": last_seg.index,
    }


def _load_bore_rows_station_points(route_coords: list[dict]) -> list[dict]:
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "bore_rows.csv"

    if not csv_path.exists():
        print("CSV NOT FOUND:", csv_path)
        return []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if not raw_rows:
        return []

    parsed_rows = []
    for i, row in enumerate(raw_rows, start=1):
        station_text = (row.get("station") or "").strip()
        station_ft = _parse_station_to_feet(station_text)

        if station_ft is None:
            continue

        try:
            depth_ft = float(row.get("depth")) if row.get("depth") not in (None, "") else None
        except ValueError:
            depth_ft = None

        try:
            boc = float(row.get("boc")) if row.get("boc") not in (None, "") else None
        except ValueError:
            boc = None

        parsed_rows.append(
            {
                "row_index": i,
                "station_text": station_text,
                "station_ft": station_ft,
                "depth_ft": depth_ft,
                "boc": boc,
            }
        )

    if not parsed_rows:
        return []

    parsed_rows.sort(key=lambda x: x["station_ft"])
    base_station_ft = parsed_rows[0]["station_ft"]

    field_points = []

    for item in parsed_rows:
        relative_progress_ft = item["station_ft"] - base_station_ft
        route_point = _point_on_route_at_progress(route_coords, relative_progress_ft)

        if not route_point:
            continue

        field_points.append(
            {
                "id": f"BR{item['row_index']}",
                "sequence": item["row_index"],
                "lat": route_point["lat"],
                "lon": route_point["lon"],
                "heading_deg": None,
                "depth_ft": item["depth_ft"],
                "rod": item["boc"],
                "timestamp": None,
                "source": "bore_rows_station",
                "meta": {
                    "station_text": item["station_text"],
                    "absolute_station_ft": item["station_ft"],
                    "relative_progress_ft": round(relative_progress_ft, 2),
                    "boc": item["boc"],
                },
            }
        )

    return field_points


def _demo_points(route):
    pts = []
    for i, r in enumerate(route[:8], start=1):
        pts.append(
            {
                "id": f"P{i}",
                "sequence": i,
                "lat": r["lat"],
                "lon": r["lon"],
                "heading_deg": 90,
                "depth_ft": 5,
                "rod": i,
                "source": "demo",
            }
        )
    return pts


def _enrich_report_for_display(report: dict) -> dict:
    results = report.get("results", [])

    for row in results:
        raw_flags = row.get("flags", []) or []
        display_flags = [_pretty_flag(flag) for flag in raw_flags]

        if "HIGH_RISK" in raw_flags:
            qa_color = "red"
        elif any(flag in raw_flags for flag in ["SHALLOW_DEPTH", "DEPTH_JUMP", "STATION_GAP", "BOC_JUMP"]):
            qa_color = "orange"
        elif row.get("offset_color") == "yellow":
            qa_color = "yellow"
        else:
            qa_color = "green"

        row["display_flags"] = display_flags
        row["qa_color"] = qa_color

    return report


def _build_report() -> dict:
    route = _get_route_coords()

    if not route or len(route) < 2:
        raise HTTPException(status_code=400, detail="No route loaded")

    field_points = _load_bore_rows_station_points(route)

    if not field_points:
        field_points = _demo_points(route)
        data_source = "demo"
    else:
        data_source = "bore_rows.csv (station-based)"

    report = match_field_points(route, field_points)
    report["data_source"] = data_source
    report["route_name"] = getattr(pipeline_state, "CURRENT_ROUTE_NAME", None)

    return _enrich_report_for_display(report)


@router.get("/report/data")
def get_report_data():
    return _build_report()


@router.get("/report/pdf")
def get_pdf():
    report = _build_report()

    pdf_bytes = build_client_ready_pdf(report)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=report.pdf"},
    )


@router.get("/report/map-data")
def get_report_map_data():
    report = _build_report()
    route = _get_route_coords()

    if not route:
        raise HTTPException(status_code=400, detail="No route loaded")

    route_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": report.get("route_name"),
                    "type": "design_route",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[pt["lon"], pt["lat"]] for pt in route],
                },
            }
        ],
    }

    point_features = []
    for row in report.get("results", []):
        point_features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": row.get("id"),
                    "sequence": row.get("sequence"),
                    "station": row.get("station"),
                    "depth_ft": row.get("depth_ft"),
                    "boc": row.get("rod"),
                    "flags": row.get("flags", []),
                    "display_flags": row.get("display_flags", []),
                    "qa_color": row.get("qa_color"),
                    "confidence": row.get("confidence"),
                    "notes": row.get("qa_notes", []),
                    "source": row.get("source"),
                    "snapped_lat": row.get("snapped_lat"),
                    "snapped_lon": row.get("snapped_lon"),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [row.get("lon"), row.get("lat")],
                },
            }
        )

    points_geojson = {
        "type": "FeatureCollection",
        "features": point_features,
    }

    return {
        "route_name": report.get("route_name"),
        "data_source": report.get("data_source"),
        "summary": report.get("summary"),
        "coverage": report.get("coverage"),
        "route_geojson": route_geojson,
        "points_geojson": points_geojson,
    }