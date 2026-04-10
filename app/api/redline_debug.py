from fastapi import APIRouter, UploadFile, File
import tempfile, os, json
from app.core.kmz_extractor import extract_routes_from_kmz
from app.core.route_matching_engine import match_bore_to_routes
from app.core.redline_slice import slice_route
from app.core.redline_helpers import get_bore_range, format_station_label

router = APIRouter()

@router.post("/debug/redline")
async def generate_redline(file: UploadFile = File(...), bore_log_json: str = None):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    routes = extract_routes_from_kmz(path)
    os.remove(path)
    routes = [r for r in routes if len(r.get("points", [])) >= 3 and r.get("end_station", 0) > 50]

    if not bore_log_json:
        return {"error": "missing_bore_log_json"}
    try:
        bore_log = json.loads(bore_log_json)
    except Exception as exc:
        return {"error": "invalid_bore_log_json", "detail": str(exc)}

    match = match_bore_to_routes(bore_log, routes)
    if match.get("status") != "matched":
        return {"error": "no_match", "match": match}

    route_id = match["best"]["route_id"]
    route = next((r for r in routes if r.get("id") == route_id), None)
    if not route:
        return {"error": "route_not_found", "route_id": route_id}

    start_ft, end_ft = get_bore_range(bore_log)
    if start_ft is None:
        return {"error": "no_valid_stations"}

    route_len = route.get("end_station", 0)
    start_ft = max(0, min(start_ft, route_len))
    end_ft = max(0, min(end_ft, route_len))
    if end_ft < start_ft:
        start_ft, end_ft = end_ft, start_ft

    coords = slice_route(route["points"], start_ft, end_ft)

    return {
        "route_id": route_id,
        "start_ft": int(start_ft),
        "end_ft": int(end_ft),
        "start_station_label": format_station_label(start_ft),
        "end_station_label": format_station_label(end_ft),
        "segment_length_ft": int(end_ft - start_ft),
        "coords": coords,
        "points_preview": coords[:50],
        "total_points": len(coords),
        "match": match,
    }
