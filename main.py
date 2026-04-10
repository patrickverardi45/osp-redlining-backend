
from __future__ import annotations

import io
import math
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="OSP Redlining Mapping Layer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KML_NS = {
    "kml": "http://www.opengis.net/kml/2.2",
    "gx": "http://www.google.com/kml/ext/2.2",
}

MAX_BUG_REPORTS = 200

STATE: Dict[str, Any] = {
    "route_name": None,
    "route_id": None,
    "route_coords": [],
    "route_length_ft": 0.0,
    "route_catalog": [],
    "map_points": [],
    "committed_rows": [],
    "station_points": [],
    "redline_segments": [],
    "loaded_field_data_files": 0,
    "latest_structured_file": None,
    "station_mapping_mode": None,
    "station_mapping_min_ft": None,
    "station_mapping_max_ft": None,
    "station_mapping_range_ft": None,
    "selected_route_match": None,
    "route_match_candidates": [],
    "verification_summary": {},
    "bug_reports": [],
}


CURRENT_PACKET_PRINT_SHEET_INDEX: Dict[str, Dict[str, Any]] = {
    # Calibrated from the detailed engineering sheets in the 07-15-25 Brenham Phase 5 design set.
    # The new Fieldwire report becomes useful starting at its page 24 because that is where the
    # embedded engineering plan pages begin showing street-level route geometry, matchlines, and
    # sheet continuity. We use those plan sheets as the print-to-street truth layer.
    #
    # Route-id calibration against the current KMZ underground-cable lines:
    # route_476 -> E Stone St corridor
    # route_477 -> E Tom Green St corridor
    # route_478 -> E Mansfield St corridor
    # route_479 / route_480 -> Niebuhr St corridor
    # route_475 -> Glenda Blvd corridor
    "1": {"sheet": 1, "streets": ["E STONE ST"], "route_ids": ["route_476"]},
    "2": {"sheet": 2, "streets": ["E STONE ST"], "route_ids": ["route_476"]},
    "3": {"sheet": 3, "streets": ["E STONE ST"], "route_ids": ["route_476"]},
    "4": {"sheet": 4, "streets": ["E STONE ST", "NIEBUHR ST"], "route_ids": ["route_476", "route_479"]},
    "5": {"sheet": 5, "streets": ["NIEBUHR ST"], "route_ids": ["route_479", "route_480"]},
    "6": {"sheet": 6, "streets": ["NIEBUHR ST"], "route_ids": ["route_479", "route_480"]},
    # For the paired 7,15 bore-log context the design truth is the E Stone St corridor.
    "7": {"sheet": 7, "streets": ["E STONE ST"], "route_ids": ["route_476"]},
    "8": {"sheet": 8, "streets": ["E MANSFIELD ST"], "route_ids": ["route_478"]},
    "9": {"sheet": 9, "streets": ["E TOM GREEN ST"], "route_ids": ["route_477"]},
    "10": {"sheet": 10, "streets": ["E TOM GREEN ST"], "route_ids": ["route_477"]},
    "11": {"sheet": 11, "streets": ["E TOM GREEN ST"], "route_ids": ["route_477"]},
    "12": {"sheet": 12, "streets": ["E TOM GREEN ST"], "route_ids": ["route_477"]},
    "13": {"sheet": 13, "streets": ["E TOM GREEN ST", "BRUCE ST"], "route_ids": ["route_477"]},
    "14": {"sheet": 14, "streets": ["E MANSFIELD ST"], "route_ids": ["route_478"]},
    "15": {"sheet": 15, "streets": ["E STONE ST"], "route_ids": ["route_476"]},
    "16": {"sheet": 16, "streets": ["NIEBUHR ST"], "route_ids": ["route_479", "route_480"]},
    "17": {"sheet": 17, "streets": ["NIEBUHR ST"], "route_ids": ["route_479", "route_480"]},
    "18": {"sheet": 18, "streets": ["NIEBUHR ST", "E TOM GREEN ST"], "route_ids": ["route_477", "route_479", "route_480"]},
    "19": {"sheet": 19, "streets": ["NIEBUHR ST"], "route_ids": ["route_479", "route_480"]},
    "20": {"sheet": 20, "streets": ["NIEBUHR ST"], "route_ids": ["route_479", "route_480"]},
    "21": {"sheet": 21, "streets": ["NIEBUHR ST"], "route_ids": ["route_479", "route_480"]},
    "22": {"sheet": 22, "streets": ["NIEBUHR ST", "E TOM GREEN ST"], "route_ids": ["route_477", "route_479", "route_480"]},
    "23": {"sheet": 23, "streets": ["CARLEE DR"], "route_ids": ["route_478"]},
    "24": {"sheet": 24, "streets": ["POST OAK CT"], "route_ids": ["route_478"]},
    "25": {"sheet": 25, "streets": ["GLENDA BLVD"], "route_ids": ["route_475"]},
    "26": {"sheet": 26, "streets": ["GLENDA BLVD"], "route_ids": ["route_475"]},
    "27": {"sheet": 27, "streets": ["GLENDA BLVD"], "route_ids": ["route_475"]},
    "28": {"sheet": 28, "streets": ["GLENDA BLVD"], "route_ids": ["route_475"]},
    "29": {"sheet": 29, "streets": ["GLENDA BLVD"], "route_ids": ["route_475"]},
    "30": {"sheet": 30, "streets": ["E STONE ST"], "route_ids": ["route_476"]},
}

def _print_sheet_hints(print_tokens: Sequence[str]) -> Dict[str, Any]:
    tokens = [str(token).strip() for token in print_tokens if str(token).strip()]
    streets: List[str] = []
    sheet_numbers: List[int] = []
    route_ids: List[str] = []

    for token in tokens:
        entry = CURRENT_PACKET_PRINT_SHEET_INDEX.get(token)
        if not entry:
            continue
        sheet = entry.get("sheet")
        if isinstance(sheet, int) and sheet not in sheet_numbers:
            sheet_numbers.append(sheet)
        for street in entry.get("streets", []) or []:
            if street not in streets:
                streets.append(street)
        for route_id in entry.get("route_ids", []) or []:
            if route_id not in route_ids:
                route_ids.append(route_id)

    return {
        "print_tokens": tokens,
        "sheet_numbers": sheet_numbers,
        "street_hints": streets,
        "allowed_route_ids": route_ids,
    }




def _store_bug_report(report: Dict[str, Any]) -> Dict[str, Any]:
    reports = STATE.setdefault("bug_reports", [])
    fingerprint = str(report.get("fingerprint") or "").strip()
    if fingerprint:
        for existing in reports:
            if str(existing.get("fingerprint") or "").strip() == fingerprint:
                existing["count"] = int(existing.get("count") or 1) + 1
                existing["timestamp"] = report.get("timestamp") or existing.get("timestamp")
                if report.get("details") is not None:
                    existing["details"] = report.get("details")
                if report.get("context") is not None:
                    existing["context"] = report.get("context")
                return existing
    reports.insert(0, dict(report))
    del reports[MAX_BUG_REPORTS:]
    return report

def _ok(**kwargs: Any) -> JSONResponse:
    return JSONResponse({"success": True, **kwargs})


def _err(message: str, status_code: int = 200, **kwargs: Any) -> JSONResponse:
    return JSONResponse({"success": False, "error": message, **kwargs}, status_code=status_code)


def _safe_filename(value: Any) -> str:
    try:
        return str(value or "").strip()
    except Exception:
        return ""


def _haversine_feet(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_m * c * 3.28084


def _route_length_ft(coords: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for i in range(1, len(coords)):
        total += _haversine_feet(
            float(coords[i - 1][0]),
            float(coords[i - 1][1]),
            float(coords[i][0]),
            float(coords[i][1]),
        )
    return total


def _normalize_station_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if "+" in text:
        left, right = text.split("+", 1)
        left = "".join(ch for ch in left if ch.isdigit())
        right = "".join(ch for ch in right if ch.isdigit())
        if not left or not right:
            return None
        return f"{int(left)}+{int(right):02d}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 3:
        return None
    return f"{int(digits[:-2])}+{int(digits[-2:]):02d}"


def _station_to_feet(value: Any) -> Optional[float]:
    normalized = _normalize_station_text(value)
    if not normalized:
        return None
    left, right = normalized.split("+", 1)
    return float(int(left) * 100 + int(right))


def _parse_coordinate_text(text: str) -> List[List[float]]:
    coords: List[List[float]] = []
    for raw in (text or "").strip().split():
        parts = raw.split(",")
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except Exception:
            continue
        coords.append([lat, lon])
    return coords


def _extract_kml_bytes(file_bytes: bytes, filename: str) -> bytes:
    lower = _safe_filename(filename).lower()
    if lower.endswith(".kml"):
        return file_bytes
    if lower.endswith(".kmz"):
        with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
            kml_names = [name for name in zf.namelist() if name.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError("No KML file found inside KMZ.")
            preferred = next((name for name in kml_names if name.lower().endswith("doc.kml")), kml_names[0])
            return zf.read(preferred)
    raise ValueError("Design upload must be .kmz or .kml")


def _dedupe_consecutive(coords: Sequence[Sequence[float]]) -> List[List[float]]:
    cleaned: List[List[float]] = []
    for pt in coords:
        lat = float(pt[0])
        lon = float(pt[1])
        if not cleaned or abs(cleaned[-1][0] - lat) > 1e-9 or abs(cleaned[-1][1] - lon) > 1e-9:
            cleaned.append([lat, lon])
    return cleaned


def _parent_map(root: ET.Element) -> Dict[int, ET.Element]:
    result: Dict[int, ET.Element] = {}
    for elem in root.iter():
        for child in elem:
            result[id(child)] = elem
    return result


def _folder_path(elem: ET.Element, parent_map: Dict[int, ET.Element]) -> List[str]:
    names: List[str] = []
    current = elem
    while id(current) in parent_map:
        current = parent_map[id(current)]
        tag = current.tag.split("}")[-1]
        if tag in {"Folder", "Document"}:
            name = (current.findtext("kml:name", default="", namespaces=KML_NS) or "").strip()
            if name:
                names.append(name)
    names.reverse()
    return names


def _build_route_catalog(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    kml_bytes = _extract_kml_bytes(file_bytes, filename)
    root = ET.fromstring(kml_bytes)
    parent_map = _parent_map(root)

    routes: List[Dict[str, Any]] = []
    route_counter = 0

    for placemark in root.findall(".//kml:Placemark", KML_NS):
        placemark_name = (placemark.findtext("kml:name", default="", namespaces=KML_NS) or "").strip() or "Unnamed Route"
        folder_names = _folder_path(placemark, parent_map)
        source_folder = " / ".join(folder_names[1:]) if len(folder_names) > 1 else (folder_names[0] if folder_names else "")
        role_hint = f"{source_folder} {placemark_name}".strip().lower()

        for node in placemark.findall(".//kml:LineString/kml:coordinates", KML_NS):
            coords = _dedupe_consecutive(_parse_coordinate_text(node.text or ""))
            if len(coords) < 2:
                continue

            route_counter += 1
            route_length_ft = round(_route_length_ft(coords), 2)
            role = "other"
            if "backbone" in role_hint:
                role = "backbone"
            elif "terminal" in role_hint and "tail" in role_hint:
                role = "terminal_tail"
            elif "house" in role_hint and "drop" in role_hint:
                role = "house_drop"
            elif "vacant" in role_hint:
                role = "vacant_pipe"
            elif "underground" in role_hint and "cable" in role_hint:
                role = "underground_cable"

            routes.append(
                {
                    "route_id": f"route_{route_counter}",
                    "route_name": placemark_name,
                    "name": placemark_name,
                    "source_folder": source_folder,
                    "coords": coords,
                    "length_ft": route_length_ft,
                    "point_count": len(coords),
                    "route_role": role,
                }
            )

    if not routes:
        raise ValueError("No valid LineString routes found in design file.")

    routes.sort(key=lambda route: (-float(route.get("length_ft", 0.0) or 0.0), route.get("route_name", "")))
    return routes


def _choose_default_route(route_catalog: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not route_catalog:
        raise ValueError("Route catalog is empty.")
    return max(route_catalog, key=lambda route: float(route.get("length_ft", 0.0) or 0.0))


def _find_route_by_id(route_id: Any) -> Optional[Dict[str, Any]]:
    target = str(route_id or "").strip()
    for route in STATE.get("route_catalog", []) or []:
        if str(route.get("route_id", "")).strip() == target:
            return route
    return None


def _set_active_route(route: Optional[Dict[str, Any]]) -> None:
    if not route:
        STATE["route_id"] = None
        STATE["route_name"] = None
        STATE["route_coords"] = []
        STATE["route_length_ft"] = 0.0
        STATE["map_points"] = []
        return

    STATE["route_id"] = route.get("route_id")
    STATE["route_name"] = route.get("route_name") or route.get("name")
    STATE["route_coords"] = route.get("coords", []) or []
    STATE["route_length_ft"] = float(route.get("length_ft", 0.0) or 0.0)
    STATE["map_points"] = route.get("coords", []) or []


def _route_chainage(coords: Sequence[Sequence[float]]) -> List[float]:
    chainage = [0.0]
    for i in range(1, len(coords)):
        chainage.append(
            chainage[-1]
            + _haversine_feet(
                float(coords[i - 1][0]),
                float(coords[i - 1][1]),
                float(coords[i][0]),
                float(coords[i][1]),
            )
        )
    return chainage


def _interpolate_point(a: Sequence[float], b: Sequence[float], ratio: float) -> List[float]:
    ratio = max(0.0, min(1.0, float(ratio)))
    return [
        float(a[0]) + (float(b[0]) - float(a[0])) * ratio,
        float(a[1]) + (float(b[1]) - float(a[1])) * ratio,
    ]


def _point_at_distance(route_coords: Sequence[Sequence[float]], chainage: Sequence[float], distance_ft: float) -> List[float]:
    if not route_coords:
        raise ValueError("Route is empty.")
    if len(route_coords) == 1:
        return [float(route_coords[0][0]), float(route_coords[0][1])]

    d = max(0.0, min(float(distance_ft), float(chainage[-1])))
    for idx in range(1, len(chainage)):
        seg_start = float(chainage[idx - 1])
        seg_end = float(chainage[idx])
        if d <= seg_end or idx == len(chainage) - 1:
            seg_len = max(seg_end - seg_start, 1e-9)
            ratio = (d - seg_start) / seg_len
            return _interpolate_point(route_coords[idx - 1], route_coords[idx], ratio)

    last = route_coords[-1]
    return [float(last[0]), float(last[1])]


def _clip_route_segment(route_coords: Sequence[Sequence[float]], start_ft: float, end_ft: float) -> List[List[float]]:
    if len(route_coords) < 2:
        return []
    chainage = _route_chainage(route_coords)
    total = float(chainage[-1])
    start_d = max(0.0, min(float(start_ft), total))
    end_d = max(0.0, min(float(end_ft), total))
    if end_d <= start_d:
        return []

    segment = [_point_at_distance(route_coords, chainage, start_d)]
    for idx in range(1, len(chainage) - 1):
        current_d = float(chainage[idx])
        if start_d < current_d < end_d:
            segment.append([float(route_coords[idx][0]), float(route_coords[idx][1])])
    segment.append(_point_at_distance(route_coords, chainage, end_d))

    cleaned: List[List[float]] = []
    for pt in segment:
        if not cleaned or abs(cleaned[-1][0] - pt[0]) > 1e-9 or abs(cleaned[-1][1] - pt[1]) > 1e-9:
            cleaned.append(pt)
    return cleaned if len(cleaned) >= 2 else []


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        text = "".join(ch for ch in str(value) if ch.isdigit() or ch in ".-")
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None


def _read_bore_log_rows(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [str(col).strip().lower() for col in df.columns]

    required = {"station", "depth", "boc"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{filename} must contain columns: station, depth, boc")

    rows: List[Dict[str, Any]] = []
    for _, rec in df.iterrows():
        station_text = _normalize_station_text(rec.get("station"))
        station_ft = _station_to_feet(station_text)
        if station_ft is None:
            continue
        rows.append(
            {
                "station": station_text,
                "station_ft": float(station_ft),
                "depth_ft": _coerce_float(rec.get("depth")),
                "boc_ft": _coerce_float(rec.get("boc")),
                "date": str(rec.get("date") or "").strip(),
                "crew": str(rec.get("crew") or "").strip(),
                "print": str(rec.get("print") or "").strip(),
                "notes": str(rec.get("notes") or "").strip(),
                "source_file": _safe_filename(filename),
            }
        )

    rows.sort(key=lambda r: float(r["station_ft"]))
    return rows


def _group_rows_for_matching(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not rows:
        return []

    groups: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = [dict(rows[0])]

    def group_key(row: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
        source_file = str(row.get("source_file") or "").strip()
        print_tokens = tuple(sorted(_parse_print_tokens(row.get("print"))))
        return source_file, print_tokens

    for row in rows[1:]:
        row_copy = dict(row)
        previous = current_group[-1]
        same_key = group_key(previous) == group_key(row_copy)
        station_delta = abs(float(row_copy["station_ft"]) - float(previous["station_ft"]))
        contiguous_station = station_delta <= 1500.0

        if same_key and contiguous_station:
            current_group.append(row_copy)
        else:
            groups.append(current_group)
            current_group = [row_copy]

    groups.append(current_group)
    return groups


def _infer_expected_roles(group_rows: Sequence[Dict[str, Any]], expected_length_ft: float) -> List[str]:
    notes_blob = " ".join(str(row.get("notes") or "") for row in group_rows).lower()
    source_blob = " ".join(
        [
            str(group_rows[0].get("source_file") or ""),
            str(group_rows[0].get("print") or ""),
            notes_blob,
        ]
    ).lower()

    expected: List[str] = []
    if "vacant" in source_blob:
        expected.append("vacant_pipe")
    if "drop" in source_blob and "house" in source_blob:
        expected.append("house_drop")
    if "tail" in source_blob:
        expected.append("terminal_tail")
    if "backbone" in source_blob:
        expected.append("backbone")
    if "cable" in source_blob or "fiber" in source_blob:
        expected.append("underground_cable")

    if expected_length_ft <= 160:
        expected.extend(["house_drop", "vacant_pipe", "terminal_tail"])
    elif expected_length_ft <= 1200:
        expected.extend(["terminal_tail", "underground_cable", "vacant_pipe"])
    else:
        expected.extend(["underground_cable", "backbone", "terminal_tail"])

    seen = set()
    ordered: List[str] = []
    for item in expected:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _route_type_bonus(route_role: str, expected_roles: Sequence[str]) -> float:
    normalized = str(route_role or "other").strip().lower()
    if not expected_roles:
        return 0.0
    if normalized == expected_roles[0]:
        return 0.18
    if normalized in expected_roles[:2]:
        return 0.10
    if normalized in expected_roles:
        return 0.04
    return 0.0




def _parse_print_tokens(value: Any) -> List[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    parts = [part.strip() for part in raw.replace(";", ",").split(",")]
    return [part for part in parts if part]


def _collect_group_print_tokens(group_rows: Sequence[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    for row in group_rows:
        for token in _parse_print_tokens(row.get("print")):
            if token not in seen:
                seen.append(token)
    return seen



def _route_filter_for_print_tokens(print_tokens: Sequence[str], route_catalog: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not print_tokens:
        return list(route_catalog), {
            "applied": False,
            "mode": "none",
            "print_tokens": [],
            "sheet_numbers": [],
            "street_hints": [],
            "allowed_route_ids": [],
            "reason": "No print tokens were present on the bore-log group.",
        }

    hint_meta = _print_sheet_hints(print_tokens)
    allowed_route_ids = list(hint_meta.get("allowed_route_ids") or [])
    street_hints = list(hint_meta.get("street_hints") or [])
    sheet_numbers = list(hint_meta.get("sheet_numbers") or [])

    if not allowed_route_ids:
        return list(route_catalog), {
            "applied": False,
            "mode": "none",
            "print_tokens": list(print_tokens),
            "sheet_numbers": sheet_numbers,
            "street_hints": street_hints,
            "allowed_route_ids": [],
            "reason": "No print-to-street extraction hints were available for this print set.",
        }

    allowed_set = set(allowed_route_ids)
    filtered = [route for route in route_catalog if str(route.get("route_id") or "") in allowed_set]

    if not filtered:
        return list(route_catalog), {
            "applied": False,
            "mode": "print_to_street_extraction",
            "print_tokens": list(print_tokens),
            "sheet_numbers": sheet_numbers,
            "street_hints": street_hints,
            "allowed_route_ids": allowed_route_ids,
            "reason": "Print-to-street extraction resolved to route ids, but none were present in the current KMZ catalog.",
        }

    return filtered, {
        "applied": True,
        "mode": "print_to_street_extraction",
        "print_tokens": list(print_tokens),
        "sheet_numbers": sheet_numbers,
        "street_hints": street_hints,
        "allowed_route_ids": allowed_route_ids,
        "reason": "Candidate routes were narrowed by print-to-street extraction calibrated from the detailed engineering sheets.",
    }

def _score_route_for_group(group_rows: Sequence[Dict[str, Any]], route: Dict[str, Any]) -> Dict[str, Any]:
    start_ft = float(group_rows[0].get("station_ft") or 0.0)
    end_ft = float(group_rows[-1].get("station_ft") or start_ft)
    expected_length_ft = max(0.0, end_ft - start_ft)

    route_length_ft = float(route.get("length_ft", 0.0) or 0.0)
    length_gap = abs(route_length_ft - expected_length_ft)
    denominator = max(expected_length_ft, route_length_ft, 1.0)
    length_score = max(0.0, 1.0 - (length_gap / denominator))

    expected_roles = _infer_expected_roles(group_rows, expected_length_ft)
    type_bonus = _route_type_bonus(str(route.get("route_role") or ""), expected_roles)

    point_count = float(route.get("point_count", 0) or 0)
    geometry_bonus = 0.02 if point_count >= 3 else 0.0

    score = round(min(1.0, length_score + type_bonus + geometry_bonus), 6)
    reason_parts = [
        f"Expected span {round(expected_length_ft, 2)} ft vs route length {round(route_length_ft, 2)} ft",
        f"Route role {route.get('route_role', 'other')}",
    ]
    if expected_roles:
        reason_parts.append(f"Expected roles {', '.join(expected_roles)}")

    return {
        "route_id": route.get("route_id"),
        "route_name": route.get("route_name"),
        "source_folder": route.get("source_folder"),
        "route_role": route.get("route_role"),
        "route_length_ft": round(route_length_ft, 2),
        "expected_span_ft": round(expected_length_ft, 2),
        "length_gap_ft": round(length_gap, 2),
        "score": score,
        "reason": " | ".join(reason_parts),
    }


def _candidate_rankings_for_group(group_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    route_catalog = STATE.get("route_catalog", []) or []
    if not route_catalog:
        raise ValueError("No route catalog loaded.")

    print_tokens = _collect_group_print_tokens(group_rows)
    candidate_routes, filter_meta = _route_filter_for_print_tokens(print_tokens, route_catalog)
    if not candidate_routes:
        raise ValueError("Print-to-street filtering produced no valid KMZ route candidates for this bore-log file.")

    rankings = [_score_route_for_group(group_rows, route) for route in candidate_routes]
    rankings.sort(key=lambda item: (-float(item.get("score", 0.0)), float(item.get("length_gap_ft", 0.0)), str(item.get("route_name", ""))))
    return rankings[:5], filter_meta


def _select_route_for_group(group_rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    rankings, filter_meta = _candidate_rankings_for_group(group_rows)
    best = rankings[0]
    matched_route = _find_route_by_id(best.get("route_id"))
    if not matched_route:
        raise ValueError("Matched route could not be resolved.")

    return matched_route, rankings, filter_meta


def _resolve_station_mapping(rows: Sequence[Dict[str, Any]], route_total_ft: float) -> Dict[str, Any]:
    station_values = [float(row["station_ft"]) for row in rows if row.get("station_ft") is not None]
    if not station_values:
        return {
            "mode": "absolute",
            "min_station_ft": None,
            "max_station_ft": None,
            "station_range_ft": None,
            "anchor_offset_ft": 0.0,
            "anchored_start_ft": None,
            "anchored_end_ft": None,
        }

    min_station = min(station_values)
    max_station = max(station_values)
    station_range = max_station - min_station

    if route_total_ft <= 0 or station_range <= 0:
        mode = "absolute"
    else:
        mode = "group_relative"

    return {
        "mode": mode,
        "min_station_ft": round(min_station, 2),
        "max_station_ft": round(max_station, 2),
        "station_range_ft": round(station_range, 2),
        "anchor_offset_ft": 0.0,
        "anchored_start_ft": 0.0 if mode == "group_relative" else round(min_station, 2),
        "anchored_end_ft": round(station_range, 2) if mode == "group_relative" else round(max_station, 2),
    }


def _map_station_to_route_distance(station_ft: float, route_total_ft: float, mapping: Dict[str, Any]) -> float:
    if route_total_ft <= 0:
        return 0.0

    mode = str(mapping.get("mode") or "absolute")
    anchor_offset_ft = float(mapping.get("anchor_offset_ft") or 0.0)

    if mode == "group_relative":
        min_station = float(mapping.get("min_station_ft") or 0.0)
        mapped = anchor_offset_ft + max(0.0, float(station_ft) - min_station)
        return max(0.0, min(mapped, route_total_ft))

    mapped = float(station_ft) + anchor_offset_ft
    return max(0.0, min(mapped, route_total_ft))


def _print_order_key(group_rows: Sequence[Dict[str, Any]], filter_meta: Dict[str, Any]) -> Tuple[int, str, str]:
    sheet_numbers = [int(value) for value in (filter_meta.get("sheet_numbers") or []) if str(value).strip().isdigit()]
    print_tokens = [str(token).strip() for token in _collect_group_print_tokens(group_rows) if str(token).strip()]
    numeric_tokens = [int(token) for token in print_tokens if token.isdigit()]

    if sheet_numbers:
        sheet_order = min(sheet_numbers)
    elif numeric_tokens:
        sheet_order = min(numeric_tokens)
    else:
        sheet_order = 10**9

    source_file = str(group_rows[0].get("source_file") or "").strip().lower()
    first_station = str(group_rows[0].get("station") or "").strip()
    return sheet_order, source_file, first_station


def _sheet_anchor_key(group_rows: Sequence[Dict[str, Any]], filter_meta: Dict[str, Any]) -> str:
    sheet_numbers = [int(value) for value in (filter_meta.get("sheet_numbers") or []) if str(value).strip().isdigit()]
    if sheet_numbers:
        return f"sheet::{min(sheet_numbers)}"

    print_tokens = [str(token).strip() for token in _collect_group_print_tokens(group_rows) if str(token).strip()]
    numeric_tokens = [int(token) for token in print_tokens if token.isdigit()]
    if numeric_tokens:
        return f"sheet::{min(numeric_tokens)}"

    if print_tokens:
        return f"print::{sorted(print_tokens)[0]}"

    return "fallback::unknown"


def _apply_non_overlapping_group_anchors(
    prepared_groups: Sequence[Dict[str, Any]],
    route_total_ft: float,
) -> Dict[int, Dict[str, Any]]:
    adjusted_mappings: Dict[int, Dict[str, Any]] = {}
    for item in prepared_groups:
        group_idx = int(item["group_idx"])
        mapping = dict(item["mapping"])
        group_rows = item["group"]
        mapping["anchor_offset_ft"] = 0.0
        mapping["anchor_strategy"] = "true_station_position_no_fabrication"
        mapping["anchor_basis"] = {
            "source_file": str(group_rows[0].get("source_file") or ""),
            "print_tokens": list(_collect_group_print_tokens(group_rows)),
            "sheet_numbers": list(item["filter_meta"].get("sheet_numbers") or []),
            "route_total_ft": round(float(route_total_ft or 0.0), 2),
        }
        if str(mapping.get("mode") or "") == "group_relative":
            station_range_ft = max(0.0, float(mapping.get("station_range_ft") or 0.0))
            mapping["anchored_start_ft"] = 0.0
            mapping["anchored_end_ft"] = round(station_range_ft, 2)
        else:
            mapping["anchored_start_ft"] = mapping.get("min_station_ft")
            mapping["anchored_end_ft"] = mapping.get("max_station_ft")
        adjusted_mappings[group_idx] = mapping
    return adjusted_mappings

def _confidence_from_rankings(mapping_mode: str, rankings: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    top = float(rankings[0].get("score", 0.0)) if rankings else 0.0
    second = float(rankings[1].get("score", 0.0)) if len(rankings) > 1 else 0.0
    margin = top - second

    if top >= 0.90 and margin >= 0.14:
        return "MEDIUM", "Best candidate selected by independent route scoring with a clear lead over alternate paths."
    if top >= 0.78 and margin >= 0.07:
        return "MEDIUM", "Best candidate selected by independent route scoring, but competing paths remain plausible."
    return "LOW", "Candidate route was selected independently, but the score spread is still too narrow for high trust."


def _build_station_points_for_group(
    rows: Sequence[Dict[str, Any]],
    matched_route: Dict[str, Any],
    rankings: Sequence[Dict[str, Any]],
    filter_meta: Dict[str, Any],
    mapping_override: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    route_coords = matched_route.get("coords", []) or []
    if len(route_coords) < 2:
        return [], {
            "mode": "absolute",
            "min_station_ft": None,
            "max_station_ft": None,
            "station_range_ft": None,
        }

    chainage = _route_chainage(route_coords)
    total = float(chainage[-1])
    mapping = dict(mapping_override or _resolve_station_mapping(rows, total))
    confidence, reason = _confidence_from_rankings(str(mapping.get("mode") or "absolute"), rankings)

    points: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        mapped_ft = _map_station_to_route_distance(float(row["station_ft"]), total, mapping)
        lat, lon = _point_at_distance(route_coords, chainage, mapped_ft)
        role = "station"
        if idx == 0:
            role = "start"
        elif idx == len(rows) - 1:
            role = "end"

        points.append(
            {
                "station": row["station"],
                "station_ft": float(row["station_ft"]),
                "mapped_station_ft": round(mapped_ft, 2),
                "lat": round(float(lat), 8),
                "lon": round(float(lon), 8),
                "depth_ft": row.get("depth_ft"),
                "boc_ft": row.get("boc_ft"),
                "notes": row.get("notes", ""),
                "date": row.get("date", ""),
                "crew": row.get("crew", ""),
                "print": row.get("print", ""),
                "job": row.get("source_file", ""),
                "source_file": row.get("source_file", ""),
                "point_role": role,
                "route_id": matched_route.get("route_id"),
                "matched_route_id": matched_route.get("route_id"),
                "matched_route_name": matched_route.get("route_name"),
                "verification": {
                    "entity_type": "station",
                    "confidence": confidence,
                    "reason": reason,
                    "route_selection_method": "independent_candidate_scoring",
                    "mapping_mode": mapping.get("mode"),
                    "anchor_type": "print_filtered_route_pool" if filter_meta.get("applied") else ("print_included_in_group_scoring" if str(row.get("print") or "").strip() else "station_range_group_scoring"),
                    "print_present": bool(str(row.get("print") or "").strip()),
                    "route_name": matched_route.get("route_name", ""),
                    "route_length_ft": round(total, 2),
                    "source_file": str(row.get("source_file") or ""),
                    "print": str(row.get("print") or ""),
                    "candidate_rankings": list(rankings),
                    "print_filter": dict(filter_meta),
                },
            }
        )

    return points, mapping


def _build_redline_segments_for_group(
    rows: Sequence[Dict[str, Any]],
    matched_route: Dict[str, Any],
    rankings: Sequence[Dict[str, Any]],
    mapping: Dict[str, Any],
    filter_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    route_coords = matched_route.get("coords", []) or []
    if len(route_coords) < 2 or len(rows) < 2:
        return []

    chainage = _route_chainage(route_coords)
    total = float(chainage[-1])
    confidence, reason = _confidence_from_rankings(str(mapping.get("mode") or "absolute"), rankings)

    segments: List[Dict[str, Any]] = []
    for idx in range(len(rows) - 1):
        start_row = rows[idx]
        end_row = rows[idx + 1]

        start_ft = _map_station_to_route_distance(float(start_row["station_ft"]), total, mapping)
        end_ft = _map_station_to_route_distance(float(end_row["station_ft"]), total, mapping)
        if end_ft <= start_ft:
            continue

        coords = _clip_route_segment(route_coords, start_ft, end_ft)
        if len(coords) < 2:
            continue

        segments.append(
            {
                "segment_id": f"{matched_route.get('route_id', 'route')}_redline_{idx + 1}_{str(start_row.get('print') or 'no_print').replace(' ', '_')}",
                "row_index": idx + 1,
                "start_station": start_row["station"],
                "end_station": end_row["station"],
                "source_start_ft": round(float(start_row["station_ft"]), 2),
                "source_end_ft": round(float(end_row["station_ft"]), 2),
                "start_ft": round(start_ft, 2),
                "end_ft": round(end_ft, 2),
                "length_ft": round(end_ft - start_ft, 2),
                "depth_ft": start_row.get("depth_ft"),
                "boc_ft": start_row.get("boc_ft"),
                "notes": start_row.get("notes", ""),
                "date": start_row.get("date", ""),
                "crew": start_row.get("crew", ""),
                "print": start_row.get("print", ""),
                "source_file": start_row.get("source_file", ""),
                "coords": coords,
                "route_id": matched_route.get("route_id"),
                "route_name": matched_route.get("route_name"),
                "matched_route_id": matched_route.get("route_id"),
                "matched_route_name": matched_route.get("route_name"),
                "verification": {
                    "entity_type": "redline",
                    "confidence": confidence,
                    "reason": reason,
                    "route_selection_method": "independent_candidate_scoring",
                    "mapping_mode": mapping.get("mode"),
                    "anchor_type": "print_filtered_route_pool" if filter_meta.get("applied") else ("print_included_in_group_scoring" if str(start_row.get("print") or "").strip() else "station_range_group_scoring"),
                    "print_present": bool(str(start_row.get("print") or "").strip()),
                    "route_name": matched_route.get("route_name", ""),
                    "route_length_ft": round(total, 2),
                    "source_file": str(start_row.get("source_file") or ""),
                    "print": str(start_row.get("print") or ""),
                    "mapped_start_ft": round(start_ft, 2),
                    "mapped_end_ft": round(end_ft, 2),
                    "source_start_station": start_row["station"],
                    "source_end_station": end_row["station"],
                    "candidate_rankings": list(rankings),
                    "print_filter": dict(filter_meta),
                },
            }
        )

    return segments


def _rebuild_field_data_outputs() -> None:
    rows = STATE.get("committed_rows", []) or []
    groups = _group_rows_for_matching(rows)

    bucketed_groups: Dict[Tuple[str, ...], List[Tuple[int, List[Dict[str, Any]]]]] = {}
    for idx, group in enumerate(groups):
        source_file = str(group[0].get("source_file") or "").strip()
        print_tokens = tuple(sorted(_collect_group_print_tokens(group)))
        bucket_key = print_tokens if print_tokens else (f"source::{source_file}",)
        bucketed_groups.setdefault(bucket_key, []).append((idx, group))

    resolved_groups: Dict[int, Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]] = {}

    for bucket_items in bucketed_groups.values():
        prepared: List[Dict[str, Any]] = []
        for group_idx, group in bucket_items:
            rankings, filter_meta = _candidate_rankings_for_group(group)
            top_score = float(rankings[0].get("score", 0.0) or 0.0)
            second_score = float(rankings[1].get("score", 0.0) or 0.0) if len(rankings) > 1 else -1.0
            expected_span = float(rankings[0].get("expected_span_ft", 0.0) or 0.0)
            prepared.append(
                {
                    "group_idx": group_idx,
                    "group": group,
                    "rankings": rankings,
                    "filter_meta": filter_meta,
                    "top_score": top_score,
                    "score_gap": top_score - second_score,
                    "expected_span": expected_span,
                }
            )

        prepared.sort(
            key=lambda item: (
                -float(item.get("score_gap", 0.0) or 0.0),
                -float(item.get("top_score", 0.0) or 0.0),
                -float(item.get("expected_span", 0.0) or 0.0),
                int(item.get("group_idx", 0) or 0),
            )
        )

        for item in prepared:
            rankings = list(item["rankings"])
            chosen = rankings[0]
            chosen_route_id = str(chosen.get("route_id") or "")
            ordered_rankings = list(rankings)
            matched_route = _find_route_by_id(chosen_route_id)
            if not matched_route:
                raise ValueError("Matched route could not be resolved.")

            filter_meta = dict(item["filter_meta"])
            filter_meta["bucket_distinct_route_assignment"] = False
            filter_meta["bucket_group_count"] = len(bucket_items)
            filter_meta["bucket_assigned_route_id"] = chosen_route_id
            filter_meta["assignment_strategy"] = "top_scoring_route_selected_true_position"

            resolved_groups[int(item["group_idx"])] = (matched_route, ordered_rankings[:5], filter_meta)

    prebuilt_mappings: Dict[int, Dict[str, Any]] = {}
    route_group_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for group_idx, group in enumerate(groups):
        matched_route, rankings, filter_meta = resolved_groups[group_idx]
        route_coords = matched_route.get("coords", []) or []
        route_total_ft = _route_length_ft(route_coords) if route_coords else 0.0
        base_mapping = _resolve_station_mapping(group, route_total_ft)
        base_mapping["anchor_strategy"] = "none"
        base_mapping["anchor_basis"] = {
            "source_file": str(group[0].get("source_file") or ""),
            "print_tokens": list(_collect_group_print_tokens(group)),
            "sheet_numbers": list(filter_meta.get("sheet_numbers") or []),
        }
        prebuilt_mappings[group_idx] = base_mapping
        route_id = str(matched_route.get("route_id") or "")
        route_group_buckets.setdefault(route_id, []).append(
            {
                "group_idx": group_idx,
                "group": group,
                "matched_route": matched_route,
                "rankings": rankings,
                "filter_meta": filter_meta,
                "mapping": base_mapping,
            }
        )

    for route_id, route_items in route_group_buckets.items():
        if len(route_items) <= 1:
            continue
        matched_route = route_items[0]["matched_route"]
        route_coords = matched_route.get("coords", []) or []
        route_total_ft = _route_length_ft(route_coords) if route_coords else 0.0
        adjusted = _apply_non_overlapping_group_anchors(route_items, route_total_ft)
        for group_idx, mapping in adjusted.items():
            prebuilt_mappings[group_idx] = mapping

    all_station_points: List[Dict[str, Any]] = []
    all_redline_segments: List[Dict[str, Any]] = []
    group_matches: List[Dict[str, Any]] = []
    mapping_modes: List[str] = []

    for group_idx, group in enumerate(groups):
        matched_route, rankings, filter_meta = resolved_groups[group_idx]
        mapping = prebuilt_mappings[group_idx]
        group_station_points, mapping = _build_station_points_for_group(group, matched_route, rankings, filter_meta, mapping)
        group_redline_segments = _build_redline_segments_for_group(group, matched_route, rankings, mapping, filter_meta)

        all_station_points.extend(group_station_points)
        all_redline_segments.extend(group_redline_segments)
        mapping_modes.append(str(mapping.get("mode") or "absolute"))

        first = rankings[0] if rankings else {}
        confidence, reason = _confidence_from_rankings(str(mapping.get("mode") or "absolute"), rankings)
        group_matches.append(
            {
                "route_id": matched_route.get("route_id"),
                "route_name": matched_route.get("route_name"),
                "source_folder": matched_route.get("source_folder"),
                "confidence": round(float(first.get("score", 0.0) or 0.0), 3),
                "confidence_label": confidence,
                "final_decision": reason,
                "route_role": matched_route.get("route_role"),
                "expected_span_ft": first.get("expected_span_ft"),
                "length_gap_ft": first.get("length_gap_ft"),
                "print": str(group[0].get("print") or ""),
                "source_file": str(group[0].get("source_file") or ""),
                "print_filter": dict(filter_meta),
                "candidate_rankings": list(rankings),
                "mapping": dict(mapping),
            }
        )

    STATE["station_points"] = all_station_points
    STATE["redline_segments"] = all_redline_segments
    STATE["station_mapping_mode"] = ",".join(sorted(set(mapping_modes))) if mapping_modes else None
    STATE["station_mapping_min_ft"] = None
    STATE["station_mapping_max_ft"] = None
    STATE["station_mapping_range_ft"] = None

    unique_route_ids = []
    for match in group_matches:
        route_id = match.get("route_id")
        if route_id and route_id not in unique_route_ids:
            unique_route_ids.append(route_id)

    if len(unique_route_ids) == 1:
        matched_route = _find_route_by_id(unique_route_ids[0])
        if matched_route:
            _set_active_route(matched_route)
        STATE["selected_route_match"] = group_matches[0] if group_matches else None
    else:
        STATE["selected_route_match"] = None

    STATE["route_match_candidates"] = group_matches
    STATE["verification_summary"] = {
        "status": "independent_route_matching_active" if group_matches else "awaiting_bore_logs",
        "version": "v2",
        "route_selection_method": "independent_candidate_scoring_per_group",
        "route_selection_reason": "Each bore-log group is matched independently against KMZ candidate lines, with calibrated print-aware filtering applied when matching hints are available for the current packet. Station placement preserves true mapped position and does not fabricate offset spacing to avoid overlap.",
        "group_count": len(group_matches),
        "unique_matched_routes": len(unique_route_ids),
    }

def _summary_payload() -> Dict[str, Any]:
    route_coords = STATE.get("route_coords", []) or []
    route_length_ft = float(STATE.get("route_length_ft", 0.0) or 0.0)
    redline_segments = STATE.get("redline_segments", []) or []
    covered_length_ft = round(sum(float(seg.get("length_ft", 0.0) or 0.0) for seg in redline_segments), 2)
    completion_pct = round((covered_length_ft / route_length_ft) * 100.0, 2) if route_length_ft > 0 else 0.0

    return {
        "route_name": STATE.get("route_name"),
        "route_catalog": STATE.get("route_catalog", []) or [],
        "suggested_route_id": STATE.get("route_id"),
        "selected_route_id": STATE.get("route_id"),
        "selected_route_name": STATE.get("route_name"),
        "selected_route_match": STATE.get("selected_route_match"),
        "route_match_candidates": STATE.get("route_match_candidates", []) or [],
        "route_coords": route_coords,
        "total_length_ft": route_length_ft,
        "covered_length_ft": covered_length_ft,
        "completion_pct": completion_pct,
        "loaded_field_data_files": int(STATE.get("loaded_field_data_files", 0) or 0),
        "latest_structured_file": STATE.get("latest_structured_file"),
        "committed_rows": STATE.get("committed_rows", []) or [],
        "redline_segments": redline_segments,
        "validation_issues": [],
        "station_points": STATE.get("station_points", []) or [],
        "invalid_redlines": [],
        "map_points": route_coords,
        "station_mapping_mode": STATE.get("station_mapping_mode"),
        "station_mapping_min_ft": STATE.get("station_mapping_min_ft"),
        "station_mapping_max_ft": STATE.get("station_mapping_max_ft"),
        "station_mapping_range_ft": STATE.get("station_mapping_range_ft"),
        "verification_summary": STATE.get("verification_summary", {}) or {},
        "bug_report_count": len(STATE.get("bug_reports", []) or []),
        "recent_bug_reports": (STATE.get("bug_reports", []) or [])[:10],
        "billing": {
            "material_rate_per_ft": 3.5,
            "splicing_rate_per_ft": 1.5,
            "material_total": 0,
            "splicing_total": 0,
            "grand_total": 0,
        },
        "validation_docs": [],
        "sheet_match_suggestions": [],
        "kmz_reference": {
            "folder_summary": [],
            "line_role_summary": [],
            "point_role_summary": [],
            "line_layers": [],
            "explicit_redline_layers": [],
            "visual_reference": {},
            "line_features": [],
            "point_features": [],
        },
    }


@app.post("/api/upload-design")
async def upload_design(file: UploadFile = File(...)) -> JSONResponse:
    try:
        file_bytes = await file.read()
        route_catalog = _build_route_catalog(file_bytes, file.filename or "design.kmz")
        STATE["route_catalog"] = route_catalog

        default_route = _choose_default_route(route_catalog)
        _set_active_route(default_route)

        rebuild_warning: Optional[str] = None

        if STATE.get("committed_rows"):
            try:
                _rebuild_field_data_outputs()
            except Exception as rebuild_exc:
                STATE["station_points"] = []
                STATE["redline_segments"] = []
                STATE["selected_route_match"] = None
                STATE["route_match_candidates"] = []
                STATE["verification_summary"] = {
                    "status": "kmz_loaded_rebuild_pending",
                    "version": "v2",
                    "route_selection_method": "independent_candidate_scoring_per_group",
                    "route_selection_reason": "KMZ loaded successfully, but existing bore-log data needs to be re-uploaded after route rebuild failed.",
                    "group_count": 0,
                    "unique_matched_routes": 0,
                }
                rebuild_warning = f"KMZ uploaded, but previous bore-log overlays were cleared because rebuild failed: {rebuild_exc}"
        else:
            STATE["station_points"] = []
            STATE["redline_segments"] = []
            STATE["selected_route_match"] = None
            STATE["route_match_candidates"] = []
            STATE["verification_summary"] = {
                "status": "awaiting_bore_logs",
                "version": "v2",
                "route_selection_method": "independent_candidate_scoring_per_group",
                "route_selection_reason": "KMZ candidate routes loaded. Bore-log matching will happen independently per group after field data upload.",
                "group_count": 0,
                "unique_matched_routes": 0,
            }

        payload = _summary_payload()
        if rebuild_warning:
            payload["warning"] = rebuild_warning
            payload["message"] = "Design uploaded successfully with previous overlays cleared."
            return _ok(**payload)

        return _ok(message="Design uploaded successfully", **payload)
    except Exception as exc:
        return _err(str(exc))


@app.post("/api/select-active-route")
async def select_active_route(route_id: str = Form(...)) -> JSONResponse:
    try:
        matched_route = _find_route_by_id(route_id)
        if not matched_route:
            return _err("Route not found.", status_code=404)

        _set_active_route(matched_route)
        return _ok(message="Active route updated", **_summary_payload())
    except Exception as exc:
        return _err(str(exc))


@app.post("/api/upload-structured-bore-files")
async def upload_structured_bore_files(files: List[UploadFile] = File(...)) -> JSONResponse:
    try:
        existing_rows = list(STATE.get("committed_rows", []) or [])
        existing_by_file: Dict[str, List[Dict[str, Any]]] = {}
        for row in existing_rows:
            source_file = str(row.get("source_file") or "").strip()
            if not source_file:
                continue
            existing_by_file.setdefault(source_file, []).append(row)

        latest_name: Optional[str] = None

        for file in files:
            file_bytes = await file.read()
            latest_name = file.filename or "structured_file"
            existing_by_file[latest_name] = _read_bore_log_rows(file_bytes, latest_name)

        merged_rows: List[Dict[str, Any]] = []
        for source_file in sorted(existing_by_file.keys()):
            merged_rows.extend(existing_by_file[source_file])

        STATE["committed_rows"] = merged_rows
        STATE["loaded_field_data_files"] = len(existing_by_file)
        STATE["latest_structured_file"] = latest_name

        _rebuild_field_data_outputs()
        return _ok(message="Bore logs uploaded successfully", **_summary_payload())
    except Exception as exc:
        return _err(str(exc))


@app.get("/api/current-state")
def current_state() -> JSONResponse:
    return _ok(**_summary_payload())


@app.post("/api/report-bug")
def report_bug(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    bug_reports = list(STATE.get("bug_reports", []) or [])
    entry = {
        "id": str(payload.get("id") or ""),
        "timestamp": str(payload.get("timestamp") or ""),
        "level": str(payload.get("level") or "info"),
        "category": str(payload.get("category") or "ui"),
        "message": str(payload.get("message") or ""),
        "details": payload.get("details") if isinstance(payload.get("details"), dict) else {},
    }
    bug_reports.insert(0, entry)
    STATE["bug_reports"] = bug_reports[:200]
    return _ok(message="Bug report captured", bug_report_count=len(STATE["bug_reports"]))


@app.get("/api/bug-reports")
def get_bug_reports() -> JSONResponse:
    return _ok(bug_reports=STATE.get("bug_reports", []) or [])
