from __future__ import annotations

import io
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pypdf import PdfReader

STREET_PATTERN = re.compile(
    r"\b(?:[NSEW]\s+)?[A-Z0-9][A-Z0-9'&.-]*(?:\s+[A-Z0-9][A-Z0-9'&.-]*)*\s+(?:ST|STREET|RD|ROAD|DR|DRIVE|LN|LANE|AVE|AVENUE|BLVD|BOULEVARD|CT|COURT|CIR|CIRCLE|TRL|TRAIL|PKWY|PARKWAY|WAY|PL|PLACE|HWY|HIGHWAY)\b",
    re.IGNORECASE,
)
ADDRESS_PATTERN = re.compile(r"\b\d{2,5}\s+[A-Z0-9][A-Z0-9'&.-]*(?:\s+[A-Z0-9][A-Z0-9'&.-]*)*\b", re.IGNORECASE)
SHEET_PATTERN = re.compile(r"\b(\d{1,2})\s+OF\s+(\d{1,2})\b", re.IGNORECASE)
DRAWING_PATTERN = re.compile(r"\b([A-Z0-9_-]+_P_\s*\d+\.DWG)\b", re.IGNORECASE)
STATION_RANGE_PATTERN = re.compile(r"STA\s*([0-9OQIL]+\+[0-9OQIL]{2})\s*(?:TO|-)\s*STA?\s*([0-9OQIL]+\+[0-9OQIL]{2})", re.IGNORECASE)
MATCHLINE_PATTERN = re.compile(r"MATCHLINE\s+STA\s*([0-9OQIL+\/\s]+?)\s*-\s*SEE\s+SHEET\s*([A-Z0-9-]+)", re.IGNORECASE)
EQUATION_PATTERN = re.compile(r"STA\s*([0-9OQIL]+\+[0-9OQIL]{2})\s*=\s*([0-9OQIL]+\+[0-9OQIL]{2})", re.IGNORECASE)
SPLICE_PATTERN = re.compile(r"(?:PROP\.\s*)?SPLICE\s+POINT\s+([A-Z0-9-]+)", re.IGNORECASE)
HANDHOLE_PATTERN = re.compile(r"(?:PLACE\s+)?(?:\d+\"X\d+\"X\d+\"\s+)?([A-Z0-9\"-]+\s*(?:HH|HANDHOLE|FLOWER POT|INSTALLER HH|PORT HH))", re.IGNORECASE)
CONTEXT_PATTERN = re.compile(
    r"\b(?:AP-\d+|PROP\.\s*SPLICE\s+POINT\s+\d+|TERMINAL\s+\d+\s+PORT\s+HH|FLOWER\s+POT|INSTALLER\s+HH|VACANT\s+HDPE|PORT\s+TERMINAL\s+TAIL|\d+CT\s+FIBER\s+OPTIC\s+CABLE|DIR\.\s+BORE)\b",
    re.IGNORECASE,
)


def normalize_station_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    text = text.replace("O", "0").replace("Q", "0").replace("I", "1").replace("L", "1")
    text = re.sub(r"[^0-9+]", "", text)
    if not text:
        return ""
    if "+" in text:
        left, right = text.split("+", 1)
        if not left or not right:
            return ""
        return f"{int(left)}+{int(right):02d}"
    digits = re.sub(r"\D", "", text)
    if len(digits) < 3:
        return ""
    return f"{int(digits[:-2])}+{int(digits[-2:]):02d}"


def station_to_feet(station: Any) -> Optional[float]:
    normalized = normalize_station_text(station)
    if not normalized:
        return None
    left, right = normalized.split("+", 1)
    return float(int(left) * 100 + int(right))


@dataclass
class CorridorSegment:
    corridor_id: str
    sheet_id: str
    chain_id: str
    station_start: float
    station_end: float
    station_start_raw: str
    station_end_raw: str
    length_ft: float
    matchline_prev: Optional[str] = None
    matchline_next: Optional[str] = None
    street_names: List[str] = field(default_factory=list)
    context_labels: List[str] = field(default_factory=list)
    raw_text_block: str = ""
    source_file: str = ""
    page_number: int = 0
    equation_parent_station: Optional[float] = None
    equation_local_start: Optional[float] = None
    see_sheets: List[str] = field(default_factory=list)


@dataclass
class CorridorGraph:
    segments: List[CorridorSegment] = field(default_factory=list)
    chains: Dict[str, List[str]] = field(default_factory=dict)
    adjacency_map: Dict[str, List[str]] = field(default_factory=dict)
    station_index: Dict[str, List[str]] = field(default_factory=dict)
    pages: List[Dict[str, Any]] = field(default_factory=list)
    equations: List[Dict[str, Any]] = field(default_factory=list)


def _unique(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _extract_page_text(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: List[Dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        compact = re.sub(r"\s+", " ", text).strip()
        pages.append(
            {
                "source_file": filename,
                "page_number": page_number,
                "page_total": len(reader.pages),
                "raw_text": text,
                "compact_text": compact,
            }
        )
    return pages


def _sheet_id_for_page(page: Dict[str, Any]) -> str:
    compact = page.get("compact_text", "")
    match = SHEET_PATTERN.search(compact)
    if match:
        return match.group(1)
    match = DRAWING_PATTERN.search(compact)
    if match:
        return match.group(1)
    return str(page.get("page_number") or "")


def _page_street_names(compact: str) -> List[str]:
    return _unique(STREET_PATTERN.findall(compact))


def _page_context_labels(compact: str) -> List[str]:
    labels: List[str] = []
    for pattern in (SPLICE_PATTERN, HANDHOLE_PATTERN, CONTEXT_PATTERN, ADDRESS_PATTERN):
        for match in pattern.findall(compact):
            if isinstance(match, tuple):
                labels.extend([m for m in match if m])
            else:
                labels.append(str(match))
    return _unique(labels)


def _page_equations(compact: str, sheet_id: str, source_file: str, page_number: int) -> List[Dict[str, Any]]:
    equations: List[Dict[str, Any]] = []
    for idx, match in enumerate(EQUATION_PATTERN.finditer(compact), start=1):
        parent_raw = normalize_station_text(match.group(1))
        local_raw = normalize_station_text(match.group(2))
        parent_ft = station_to_feet(parent_raw)
        local_ft = station_to_feet(local_raw)
        if parent_ft is None or local_ft is None:
            continue
        equations.append(
            {
                "equation_id": f"eq_{source_file}_{page_number}_{idx}",
                "sheet_id": sheet_id,
                "source_file": source_file,
                "page_number": page_number,
                "parent_station_raw": parent_raw,
                "parent_station_ft": float(parent_ft),
                "local_station_raw": local_raw,
                "local_station_ft": float(local_ft),
            }
        )
    equations.sort(key=lambda item: float(item.get("parent_station_ft", 0.0)))
    return equations


def _page_matchlines(compact: str) -> List[Dict[str, Any]]:
    matchlines: List[Dict[str, Any]] = []
    for match in MATCHLINE_PATTERN.finditer(compact):
        stations_blob = re.sub(r"\s+", "", match.group(1) or "")
        see_sheet = str(match.group(2) or "").strip()
        values = [station_to_feet(raw) for raw in re.findall(r"[0-9OQIL]+\+[0-9OQIL]{2}", stations_blob, re.IGNORECASE)]
        clean_values = [float(v) for v in values if v is not None]
        matchlines.append(
            {
                "stations_text": stations_blob,
                "station_values_ft": clean_values,
                "see_sheet": see_sheet,
            }
        )
    return matchlines


def _assign_chain_id(start_ft: float, end_ft: float, equations: Sequence[Dict[str, Any]], sheet_id: str) -> Tuple[str, Optional[float], Optional[float]]:
    low = min(start_ft, end_ft)
    high = max(start_ft, end_ft)
    for idx, eq in enumerate(sorted(equations, key=lambda item: float(item.get("parent_station_ft", 0.0)))):
        parent_ft = float(eq.get("parent_station_ft", 0.0) or 0.0)
        local_ft = float(eq.get("local_station_ft", 0.0) or 0.0)
        if local_ft > 0:
            continue
        # Branch segments live in the low-numbered local chain after a reset like 7+40 = 0+00.
        if high <= max(parent_ft - 100.0, 150.0):
            return f"sheet_{sheet_id}_branch_{idx + 1}", parent_ft, local_ft
    return f"sheet_{sheet_id}_main", None, None


def _matchline_neighbors(segment: CorridorSegment, matchlines: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], List[str]]:
    prev_sheet = None
    next_sheet = None
    see_sheets: List[str] = []
    start_ft = float(segment.station_start)
    end_ft = float(segment.station_end)
    for matchline in matchlines:
        values = [float(v) for v in (matchline.get("station_values_ft") or [])]
        if not values:
            continue
        see_sheet = str(matchline.get("see_sheet") or "").strip()
        if see_sheet:
            see_sheets.append(see_sheet)
        if any(abs(v - start_ft) <= 5.0 for v in values):
            prev_sheet = see_sheet or prev_sheet
        if any(abs(v - end_ft) <= 5.0 for v in values):
            next_sheet = see_sheet or next_sheet
    return prev_sheet, next_sheet, _unique(see_sheets)


def build_corridor_graph_from_pdfs(uploaded: Sequence[Tuple[bytes, str]]) -> CorridorGraph:
    graph = CorridorGraph()
    for file_bytes, filename in uploaded:
        for page in _extract_page_text(file_bytes, filename):
            compact = page["compact_text"]
            sheet_id = _sheet_id_for_page(page)
            equations = _page_equations(compact, sheet_id, filename, int(page["page_number"]))
            matchlines = _page_matchlines(compact)
            street_names = _page_street_names(compact)
            context_labels = _page_context_labels(compact)
            page_record = {
                "source_file": filename,
                "page_number": int(page["page_number"]),
                "page_total": int(page["page_total"]),
                "sheet_id": sheet_id,
                "station_ranges": [],
                "matchlines": matchlines,
                "station_values_ft": [float(v) for v in [station_to_feet(raw) for raw in re.findall(r"[0-9OQIL]+\+[0-9OQIL]{2}", compact, re.IGNORECASE)] if v is not None],
                "street_names": street_names,
                "context_labels": context_labels,
                "equations": equations,
                "text_excerpt": compact[:800],
            }
            for idx, match in enumerate(STATION_RANGE_PATTERN.finditer(compact), start=1):
                start_raw = normalize_station_text(match.group(1))
                end_raw = normalize_station_text(match.group(2))
                start_ft = station_to_feet(start_raw)
                end_ft = station_to_feet(end_raw)
                if start_ft is None or end_ft is None:
                    continue
                chain_id, eq_parent, eq_local = _assign_chain_id(float(start_ft), float(end_ft), equations, sheet_id)
                temp_segment = CorridorSegment(
                    corridor_id=f"corr_{filename}_{page['page_number']}_{idx}",
                    sheet_id=sheet_id,
                    chain_id=chain_id,
                    station_start=min(float(start_ft), float(end_ft)),
                    station_end=max(float(start_ft), float(end_ft)),
                    station_start_raw=start_raw,
                    station_end_raw=end_raw,
                    length_ft=abs(float(end_ft) - float(start_ft)),
                    street_names=street_names,
                    context_labels=context_labels,
                    raw_text_block=compact[:2000],
                    source_file=filename,
                    page_number=int(page["page_number"]),
                    equation_parent_station=eq_parent,
                    equation_local_start=eq_local,
                )
                prev_sheet, next_sheet, see_sheets = _matchline_neighbors(temp_segment, matchlines)
                temp_segment.matchline_prev = prev_sheet
                temp_segment.matchline_next = next_sheet
                temp_segment.see_sheets = see_sheets
                graph.segments.append(temp_segment)
                page_record["station_ranges"].append(
                    {
                        "corridor_id": temp_segment.corridor_id,
                        "start_station": start_raw,
                        "end_station": end_raw,
                        "start_ft": temp_segment.station_start,
                        "end_ft": temp_segment.station_end,
                        "chain_id": chain_id,
                    }
                )
            graph.pages.append(page_record)
            graph.equations.extend(equations)

    chain_map: Dict[str, List[CorridorSegment]] = {}
    for segment in graph.segments:
        chain_map.setdefault(segment.chain_id, []).append(segment)
        bucket_key = f"{int(segment.station_start // 100)}xx"
        graph.station_index.setdefault(bucket_key, []).append(segment.corridor_id)

    segment_by_id = {segment.corridor_id: segment for segment in graph.segments}
    for chain_id, items in chain_map.items():
        ordered = sorted(items, key=lambda seg: (seg.station_start, seg.station_end, seg.page_number))
        graph.chains[chain_id] = [segment.corridor_id for segment in ordered]
        for idx, segment in enumerate(ordered):
            neighbors = graph.adjacency_map.setdefault(segment.corridor_id, [])
            if idx > 0:
                neighbors.append(ordered[idx - 1].corridor_id)
            if idx < len(ordered) - 1:
                neighbors.append(ordered[idx + 1].corridor_id)

    # Link segments across sheets using explicit see-sheet references and near-touching station boundaries.
    for segment in graph.segments:
        neighbors = graph.adjacency_map.setdefault(segment.corridor_id, [])
        for other in graph.segments:
            if other.corridor_id == segment.corridor_id:
                continue
            if other.sheet_id in (segment.matchline_prev, segment.matchline_next) or segment.sheet_id in other.see_sheets:
                if abs(other.station_start - segment.station_end) <= 10.0 or abs(other.station_end - segment.station_start) <= 10.0:
                    neighbors.append(other.corridor_id)
        graph.adjacency_map[segment.corridor_id] = _unique(neighbors)

    return graph


def corridor_graph_to_serializable(graph: CorridorGraph) -> Dict[str, Any]:
    return {
        "segments": [asdict(segment) for segment in graph.segments],
        "chains": graph.chains,
        "adjacency_map": graph.adjacency_map,
        "station_index": graph.station_index,
        "pages": graph.pages,
        "equations": graph.equations,
    }


def _segment_tokens(segment: CorridorSegment) -> List[str]:
    text = " ".join(segment.street_names + segment.context_labels + [segment.raw_text_block])
    return [tok for tok in re.sub(r"[^a-z0-9]+", " ", text.lower()).split() if len(tok) >= 3 and not tok.isdigit()]


def _rows_tokens(rows: Sequence[Dict[str, Any]]) -> List[str]:
    pieces: List[str] = []
    for row in rows or []:
        for key in ("notes", "reason", "print", "crew", "source_file", "date"):
            pieces.append(str(row.get(key) or ""))
    return [tok for tok in re.sub(r"[^a-z0-9]+", " ", " ".join(pieces).lower()).split() if len(tok) >= 3 and not tok.isdigit()]


def _detect_station_reset(rows: Sequence[Dict[str, Any]], best_match: Optional[Dict[str, Any]] = None) -> bool:
    ordered: List[float] = []
    for row in rows or []:
        for key in ("station", "start_station", "end_station"):
            ft = station_to_feet(row.get(key))
            if ft is not None:
                ordered.append(float(ft))
    for prev, curr in zip(ordered, ordered[1:]):
        if curr + 1.0 < prev:
            return True
    if best_match and best_match.get("equation_parent_station_ft") is not None:
        return True
    return False


def match_rows_to_corridors(graph: CorridorGraph, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not graph.segments:
        return {"best_match": None, "candidates": [], "needs_review": True, "reason": ["No corridor segments were extracted from the PDFs."], "debug": {"station_reset_detected": False}}

    station_values: List[float] = []
    for row in rows or []:
        for key in ("station", "start_station", "end_station"):
            ft = station_to_feet(row.get(key))
            if ft is not None:
                station_values.append(float(ft))
    if not station_values:
        return {"best_match": None, "candidates": [], "needs_review": True, "reason": ["No station values were found in the field rows."], "debug": {"station_reset_detected": False}}

    min_station = min(station_values)
    max_station = max(station_values)
    span_ft = max_station - min_station
    row_tokens = set(_rows_tokens(rows))

    scored: List[Dict[str, Any]] = []
    raw_scores: List[float] = []
    for segment in graph.segments:
        overlap = max(0.0, min(segment.station_end, max_station) - max(segment.station_start, min_station))
        containment = overlap / max(span_ft, 1.0)
        if span_ft <= 0:
            containment = 1.0 if segment.station_start <= min_station <= segment.station_end else 0.0
        segment_span_delta = abs(segment.length_ft - span_ft)
        score = containment * 650.0 - segment_span_delta * 0.55
        reasons: List[str] = []
        if overlap > 0:
            reasons.append("station containment overlap")
        fully_contains = segment.station_start <= min_station + 5.0 and segment.station_end >= max_station - 5.0
        if fully_contains:
            score += 240.0
            reasons.append("corridor fully contains bore-log span")
        elif overlap > 0:
            score += 120.0
        if segment.equation_parent_station is not None and max_station <= float(segment.equation_parent_station):
            score += 90.0
            reasons.append("chain/reset relationship fits the low-station branch")
        seg_tokens = set(_segment_tokens(segment))
        token_overlap = sorted(seg_tokens & row_tokens)
        if token_overlap:
            score += min(len(token_overlap), 5) * 24.0
            reasons.append(f"context token overlap ({', '.join(token_overlap[:4])})")
        if segment.street_names:
            reasons.append(f"street context: {', '.join(segment.street_names[:3])}")
        scored.append(
            {
                "corridor_id": segment.corridor_id,
                "sheet_id": segment.sheet_id,
                "chain_id": segment.chain_id,
                "source_file": segment.source_file,
                "page_number": segment.page_number,
                "station_start": segment.station_start_raw,
                "station_end": segment.station_end_raw,
                "station_start_ft": segment.station_start,
                "station_end_ft": segment.station_end,
                "length_ft": segment.length_ft,
                "overlap_ft": round(overlap, 2),
                "containment_score": round(containment, 4),
                "segment_span_delta_ft": round(segment_span_delta, 2),
                "street_names": segment.street_names,
                "context_labels": segment.context_labels[:12],
                "match_reasons": reasons[:6],
                "raw_score": round(score, 4),
                "equation_parent_station_ft": segment.equation_parent_station,
                "equation_local_start_ft": segment.equation_local_start,
                "matchline_prev": segment.matchline_prev,
                "matchline_next": segment.matchline_next,
                "see_sheets": segment.see_sheets,
            }
        )
        raw_scores.append(score)

    scored.sort(key=lambda item: float(item.get("raw_score", 0.0)), reverse=True)
    if not scored:
        return {"best_match": None, "candidates": [], "needs_review": True, "reason": ["No corridor candidates scored."], "debug": {"station_reset_detected": False}}

    min_score = min(raw_scores)
    max_score = max(raw_scores)
    score_range = max(max_score - min_score, 1e-9)
    top_score = float(scored[0].get("raw_score", 0.0) or 0.0)
    second_score = float(scored[1].get("raw_score", top_score) or top_score) if len(scored) > 1 else top_score
    for idx, item in enumerate(scored):
        confidence = (float(item.get("raw_score", 0.0) or 0.0) - min_score) / score_range
        item["confidence"] = round(confidence, 4)
        if idx == 0:
            margin = (top_score - second_score) / max(abs(top_score), 1.0)
            item["top_margin"] = round(margin, 4)
            item["strong_match"] = bool((confidence >= 0.68 and margin >= 0.08) or (confidence >= 0.84 and margin >= 0.03))
        else:
            item["strong_match"] = False
    best = scored[0]
    station_reset_detected = _detect_station_reset(rows, best)
    best["matched_station_range"] = f"{normalize_station_text(best.get('station_start'))} -> {normalize_station_text(best.get('station_end'))}"
    best["station_reset_detected"] = station_reset_detected
    return {
        "best_match": best,
        "candidates": scored[:8],
        "needs_review": not bool(best.get("strong_match")),
        "reason": best.get("match_reasons", []),
        "debug": {
            "matched_pdf_sheet": best.get("sheet_id"),
            "matched_chain_id": best.get("chain_id"),
            "matched_corridor_ids": [best.get("corridor_id")],
            "matched_station_range": best.get("matched_station_range"),
            "matched_street_names": best.get("street_names", []),
            "station_reset_detected": station_reset_detected,
        },
    }


def _preferred_roles_for_corridor(corridor_match: Optional[Dict[str, Any]]) -> List[str]:
    text = " ".join((corridor_match or {}).get("context_labels", []) + (corridor_match or {}).get("street_names", []) + (corridor_match or {}).get("match_reasons", []))
    lowered = text.lower()
    roles: List[str] = []
    terminal_context = "port terminal tail" in lowered or "terminal" in lowered
    vacant_context = "vacant" in lowered
    cable_context = any(token in lowered for token in ["288ct", "48ct", "12ct", "fiber optic cable", "splice point", "dir. bore"])

    if terminal_context:
        roles.append("terminal_tail")
    if vacant_context:
        roles.append("vacant_pipe")
    if cable_context:
        roles.append("underground_cable")
    if cable_context and not terminal_context and not vacant_context:
        roles.append("backbone")
    if not roles:
        roles.extend(["terminal_tail", "vacant_pipe", "underground_cable"])
    return _unique(roles)


def _normalize_text_tokens(value: Any) -> List[str]:
    return [tok for tok in re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split() if tok and len(tok) >= 2]


def _corridor_anchor_tokens(best: Dict[str, Any]) -> List[str]:
    pieces: List[str] = []
    for value in (best.get("street_names") or []) + (best.get("context_labels") or []):
        pieces.append(str(value or ""))
    joined = " ".join(pieces)
    extra = re.findall(r"(?:SPLICE\s+LOC\s*\d+|AP-\d+|\d{3})", joined, flags=re.IGNORECASE)
    tokens: List[str] = []
    for piece in pieces + extra:
        tokens.extend(_normalize_text_tokens(piece))
    return _unique(tokens)


def _point_matches_tokens(point: Dict[str, Any], tokens: Sequence[str]) -> bool:
    haystack = " ".join([
        str(point.get("name") or ""),
        str(point.get("folder_path") or ""),
        str(point.get("role") or ""),
    ]).lower()
    if not haystack.strip():
        return False
    return any(token in haystack for token in tokens if len(token) >= 2)


def _route_min_anchor_distance_ft(route: Dict[str, Any], anchors: Sequence[Tuple[float, float]]) -> float:
    coords = route.get("geometry", []) or []
    best = float("inf")
    for lat, lon in coords:
        for a_lat, a_lon in anchors:
            dist = ((69.0 * (float(lat) - float(a_lat))) ** 2 + (math.cos(math.radians((float(lat) + float(a_lat)) / 2.0)) * 69.0 * (float(lon) - float(a_lon))) ** 2) ** 0.5 * 5280.0
            if dist < best:
                best = dist
    return best


def _route_points_inside_bbox(route: Dict[str, Any], bbox: Optional[Tuple[float, float, float, float]]) -> Tuple[int, int]:
    coords = route.get("geometry", []) or []
    if not coords or not bbox:
        return 0, len(coords)
    min_lat, max_lat, min_lon, max_lon = bbox
    inside = 0
    for lat, lon in coords:
        if min_lat <= float(lat) <= max_lat and min_lon <= float(lon) <= max_lon:
            inside += 1
    return inside, len(coords)


def _corridor_bbox_from_anchors(anchors: Sequence[Tuple[float, float]], margin_ft: float) -> Optional[Tuple[float, float, float, float]]:
    if not anchors:
        return None
    lats = [float(lat) for lat, _lon in anchors]
    lons = [float(lon) for _lat, lon in anchors]
    center_lat = sum(lats) / len(lats)
    lat_margin = float(margin_ft) / 364000.0
    lon_margin = float(margin_ft) / max(364000.0 * max(math.cos(math.radians(center_lat)), 0.2), 1.0)
    return (
        min(lats) - lat_margin,
        max(lats) + lat_margin,
        min(lons) - lon_margin,
        max(lons) + lon_margin,
    )


def _generic_route_name_penalty(route: Dict[str, Any]) -> float:
    route_name = str(route.get("route_name") or "").strip().lower()
    folder = str(route.get("source_folder") or "").strip().lower()
    role = str(route.get("role") or "").strip().lower().replace("_", " ")
    penalty = 0.0
    if route_name == "local corridor chain":
        return 5.0
    if not route_name:
        return 220.0
    if route_name == role or route_name == folder:
        penalty += 220.0
    if role and route_name.startswith(role):
        penalty += 120.0
    if folder and folder in route_name:
        penalty += 80.0
    if any(term in route_name for term in ["underground cable", "connections", "backbone", "terminal tail", "vacant pipe"]):
        penalty += 60.0
    return penalty


def _route_identity_key(route: Dict[str, Any]) -> str:
    route_name = str(route.get("route_name") or "").strip().lower()
    folder = str(route.get("source_folder") or "").strip().lower()
    role = str(route.get("role") or "").strip().lower()
    return f"{folder}|{role}|{route_name}"


def _prune_duplicate_generic_routes(routes: Sequence[Dict[str, Any]], span_ft: Optional[float]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(routes) <= 1:
        return list(routes), {"duplicate_groups_pruned": 0, "duplicate_routes_removed": 0}

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    ordered_keys: List[str] = []
    for route in routes:
        key = _route_identity_key(route)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(route)

    kept: List[Dict[str, Any]] = []
    groups_pruned = 0
    routes_removed = 0
    for key in ordered_keys:
        group = grouped[key]
        if len(group) == 1:
            kept.append(group[0])
            continue

        groups_pruned += 1
        routes_removed += max(len(group) - 1, 0)
        group_sorted = sorted(
            group,
            key=lambda route: (
                -float(route.get("_corridor_locality_score", 0.0) or 0.0),
                -float(route.get("_corridor_bbox_ratio", 0.0) or 0.0),
                float(route.get("_corridor_gate_distance_ft") or 10**9),
                abs(float(route.get("total_length_ft", 0.0) or 0.0) - float(span_ft or 0.0)),
                float(route.get("_generic_name_penalty", 0.0) or 0.0),
                str(route.get("route_id") or ""),
            ),
        )
        winner = dict(group_sorted[0])
        winner["_duplicate_name_family_size"] = len(group)
        kept.append(winner)

    kept.sort(
        key=lambda route: (
            -float(route.get("_corridor_locality_score", 0.0) or 0.0),
            -float(route.get("_corridor_bbox_ratio", 0.0) or 0.0),
            float(route.get("_corridor_gate_distance_ft") or 10**9),
            abs(float(route.get("total_length_ft", 0.0) or 0.0) - float(span_ft or 0.0)),
            float(route.get("_generic_name_penalty", 0.0) or 0.0),
            str(route.get("route_id") or ""),
        )
    )
    return kept, {
        "duplicate_groups_pruned": groups_pruned,
        "duplicate_routes_removed": routes_removed,
    }


def _route_centroid(route: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    coords = route.get("geometry", []) or []
    if not coords:
        return None
    lat = sum(float(pt[0]) for pt in coords) / len(coords)
    lon = sum(float(pt[1]) for pt in coords) / len(coords)
    return (lat, lon)


def _point_distance_ft(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((69.0 * (float(a[0]) - float(b[0]))) ** 2 + (math.cos(math.radians((float(a[0]) + float(b[0])) / 2.0)) * 69.0 * (float(a[1]) - float(b[1]))) ** 2) ** 0.5 * 5280.0


def _cluster_routes_to_local_family(routes: Sequence[Dict[str, Any]], span_ft: Optional[float]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(routes) <= 1:
        return list(routes), {"cluster_radius_ft": None, "cluster_seed_route_id": None, "cluster_size_before_cap": len(routes), "cluster_size_after_cap": len(routes)}

    cluster_radius_ft = max(220.0, min(575.0, float(span_ft or 0.0) * 0.55 + 160.0))
    centroid_rows: List[Tuple[Dict[str, Any], Tuple[float, float]]] = []
    for route in routes:
        centroid = _route_centroid(route)
        if centroid is not None:
            centroid_rows.append((route, centroid))
    if not centroid_rows:
        return list(routes), {"cluster_radius_ft": round(cluster_radius_ft, 2), "cluster_seed_route_id": None, "cluster_size_before_cap": len(routes), "cluster_size_after_cap": len(routes)}

    scored_clusters: List[Tuple[int, float, float, str, List[Dict[str, Any]]]] = []
    for seed_route, seed_centroid in centroid_rows:
        local_members: List[Dict[str, Any]] = []
        distance_total = 0.0
        for route, centroid in centroid_rows:
            distance_ft = _point_distance_ft(seed_centroid, centroid)
            if distance_ft <= cluster_radius_ft:
                clone = dict(route)
                clone["_cluster_distance_ft"] = round(distance_ft, 2)
                local_members.append(clone)
                distance_total += distance_ft
        local_members.sort(key=lambda item: (
            float(item.get("_cluster_distance_ft", 10**9) or 10**9),
            -float(item.get("_corridor_locality_score", 0.0) or 0.0),
            -float(item.get("_corridor_bbox_ratio", 0.0) or 0.0),
            float(item.get("_generic_name_penalty", 0.0) or 0.0),
            abs(float(item.get("total_length_ft", 0.0) or 0.0) - float(span_ft or 0.0)),
        ))
        scored_clusters.append((
            len(local_members),
            -distance_total,
            -float(seed_route.get("_corridor_locality_score", 0.0) or 0.0),
            str(seed_route.get("route_id") or ""),
            local_members,
        ))

    best_cluster = max(scored_clusters, key=lambda item: (item[0], item[1], item[2], item[3]))
    cluster_members = best_cluster[4]
    capped_members = cluster_members[:8]
    return capped_members, {
        "cluster_radius_ft": round(cluster_radius_ft, 2),
        "cluster_seed_route_id": str(capped_members[0].get("route_id") or "") if capped_members else None,
        "cluster_size_before_cap": len(cluster_members),
        "cluster_size_after_cap": len(capped_members),
    }


def _route_endpoints(route: Dict[str, Any]) -> List[Tuple[float, float]]:
    coords = route.get("geometry", []) or []
    if len(coords) < 2:
        return []
    try:
        return [
            (float(coords[0][0]), float(coords[0][1])),
            (float(coords[-1][0]), float(coords[-1][1])),
        ]
    except Exception:
        return []


def _routes_are_chainable(a: Dict[str, Any], b: Dict[str, Any], max_link_ft: float) -> bool:
    a_endpoints = _route_endpoints(a)
    b_endpoints = _route_endpoints(b)
    if not a_endpoints or not b_endpoints:
        return False
    for a_pt in a_endpoints:
        for b_pt in b_endpoints:
            if _point_distance_ft(a_pt, b_pt) <= max_link_ft:
                return True
    a_centroid = _route_centroid(a)
    b_centroid = _route_centroid(b)
    if a_centroid and b_centroid and _point_distance_ft(a_centroid, b_centroid) <= max_link_ft * 1.45:
        return True
    # also allow chaining when one route endpoint lands close to the other route body
    for a_pt in a_endpoints:
        if _route_min_anchor_distance_ft(b, [a_pt]) <= max_link_ft * 1.10:
            return True
    for b_pt in b_endpoints:
        if _route_min_anchor_distance_ft(a, [b_pt]) <= max_link_ft * 1.10:
            return True
    return False


def _merge_route_geometries(routes: Sequence[Dict[str, Any]]) -> List[List[float]]:
    merged: List[List[float]] = []
    seen: set[Tuple[float, float]] = set()
    for route in routes:
        for lat, lon in route.get("geometry", []) or []:
            key = (round(float(lat), 8), round(float(lon), 8))
            if key in seen:
                continue
            seen.add(key)
            merged.append([float(lat), float(lon)])
    return merged


def _synthesize_local_route_chains(routes: Sequence[Dict[str, Any]], span_ft: Optional[float], farthest_station: Optional[float]) -> List[Dict[str, Any]]:
    base_routes = [dict(route) for route in (routes or [])]
    if len(base_routes) < 2:
        return []

    target_length = float(farthest_station or span_ft or 0.0)
    if target_length <= 0:
        return []

    max_link_ft = max(90.0, min(240.0, target_length * 0.14 + 30.0))
    candidates = sorted(
        base_routes,
        key=lambda route: (
            -float(route.get("_corridor_locality_score", 0.0) or 0.0),
            -float(route.get("_corridor_bbox_ratio", 0.0) or 0.0),
            abs(float(route.get("total_length_ft", 0.0) or 0.0) - target_length),
            float(route.get("_corridor_gate_distance_ft") or 10**9),
        ),
    )[:18]

    synthetic: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, ...]] = set()

    def add_combo(combo: Sequence[Dict[str, Any]]) -> None:
        key = tuple(sorted(str(item.get("route_id") or "") for item in combo))
        if len(key) < 2 or key in seen_keys:
            return
        seen_keys.add(key)
        total_length = sum(float(item.get("total_length_ft", 0.0) or 0.0) for item in combo)
        min_required = max(target_length - 80.0, target_length * 0.84)
        max_allowed = max(target_length * 1.38, target_length + 220.0)
        if total_length < min_required or total_length > max_allowed:
            return
        geometry = _merge_route_geometries(combo)
        if len(geometry) < 2:
            return
        locality_score = sum(float(item.get("_corridor_locality_score", 0.0) or 0.0) for item in combo) / len(combo)
        bbox_ratio = sum(float(item.get("_corridor_bbox_ratio", 0.0) or 0.0) for item in combo) / len(combo)
        priority_score = max(float(item.get("priority_score", 0.0) or 0.0) for item in combo) + len(combo) * 0.85
        dominant = max((str(item.get("role") or "other") for item in combo), key=lambda role: sum(1 for item in combo if str(item.get("role") or "other") == role))
        synthetic.append({
            "route_id": f"synthetic_chain_{'_'.join(key)}",
            "route_name": "Local corridor chain",
            "source_folder": "corridor_chain",
            "role": dominant,
            "geometry": geometry,
            "total_length_ft": round(float(total_length), 2),
            "priority_score": round(float(priority_score), 2),
            "feature_id": f"synthetic_chain_{'_'.join(key)}",
            "line_color_hex": "",
            "line_width": 0.0,
            "_synthetic_chain": True,
            "_chain_route_ids": list(key),
            "_corridor_locality_score": round(float(locality_score), 4),
            "_corridor_bbox_ratio": round(float(bbox_ratio), 4),
            "_corridor_gate_distance_ft": min(float(item.get("_corridor_gate_distance_ft") or 10**9) for item in combo if item.get("_corridor_gate_distance_ft") is not None) if any(item.get("_corridor_gate_distance_ft") is not None for item in combo) else None,
            "_corridor_anchor_hit": any(bool(item.get("_corridor_anchor_hit")) for item in combo),
            "_corridor_endpoint_inside": any(bool(item.get("_corridor_endpoint_inside")) for item in combo),
            "_generic_name_penalty": 5.0,
            "_corridor_role_bonus": max(float(item.get("_corridor_role_bonus", 0.0) or 0.0) for item in combo),
        })

    # brute-force small local combinations instead of relying only on greedy growth
    from itertools import combinations
    for size in (2, 3, 4, 5):
        for combo in combinations(candidates, size):
            if not all(any(_routes_are_chainable(a, b, max_link_ft) for b in combo if b is not a) for a in combo):
                continue
            add_combo(combo)

    synthetic.sort(
        key=lambda route: (
            abs(float(route.get("total_length_ft", 0.0) or 0.0) - target_length),
            -float(route.get("_corridor_locality_score", 0.0) or 0.0),
            -float(route.get("_corridor_bbox_ratio", 0.0) or 0.0),
            -float(route.get("priority_score", 0.0) or 0.0),
        )
    )
    return synthetic[:10]

def filter_route_catalog_by_corridor_match(route_catalog: Sequence[Dict[str, Any]], corridor_result: Dict[str, Any], rows: Sequence[Dict[str, Any]], kmz_reference: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not route_catalog:
        return [], {"filter_mode": "empty_catalog", "preferred_roles": [], "kept_route_ids": [], "candidate_count_before_filter": 0, "candidate_count_after_filter": 0}

    best = corridor_result.get("best_match") or {}
    debug = dict(corridor_result.get("debug") or {})
    if not best:
        fallback_routes = [dict(route) for route in route_catalog]
        debug.update({
            "matched_pdf_sheet": None,
            "matched_chain_id": None,
            "matched_corridor_ids": [],
            "matched_station_range": None,
            "matched_street_names": [],
            "station_reset_detected": False,
            "kmz_candidates_before_filter": len(route_catalog),
            "kmz_candidates_after_filter": len(fallback_routes),
            "corridor_gate_status": "no_corridor_match_fallback",
            "corridor_anchor_tokens": [],
            "corridor_anchor_point_ids": [],
            "corridor_anchor_count": 0,
        })
        return fallback_routes, {
            "filter_mode": "corridor_graph_fallback_all_routes",
            "preferred_roles": [],
            "kept_route_ids": [str(route.get("route_id") or "") for route in fallback_routes[:50]],
            "candidate_count_before_filter": len(route_catalog),
            "candidate_count_after_filter": len(fallback_routes),
            "needs_review": True,
            "debug": debug,
        }

    preferred_roles = _preferred_roles_for_corridor(best)
    station_values: List[float] = []
    for row in rows or []:
        for key in ("station", "start_station", "end_station"):
            ft = station_to_feet(row.get(key))
            if ft is not None:
                station_values.append(float(ft))
    farthest_station = max(station_values) if station_values else None
    span_ft = (max(station_values) - min(station_values)) if len(station_values) >= 2 else None

    anchor_tokens = _corridor_anchor_tokens(best)
    point_features = (kmz_reference or {}).get("point_features", []) or []
    anchor_points: List[Tuple[float, float]] = []
    matched_point_ids: List[str] = []
    for point in point_features:
        if _point_matches_tokens(point, anchor_tokens):
            try:
                anchor_points.append((float(point.get("lat")), float(point.get("lon"))))
                matched_point_ids.append(str(point.get("feature_id") or ""))
            except Exception:
                continue
    anchor_points = list(dict.fromkeys(anchor_points))

    center_lat = None
    center_lon = None
    spread_ft = None
    locality_radius_ft = None
    corridor_bbox = None
    corridor_bbox_margin_ft = None
    if anchor_points:
        center_lat = sum(p[0] for p in anchor_points) / len(anchor_points)
        center_lon = sum(p[1] for p in anchor_points) / len(anchor_points)
        spread_ft = max(_route_min_anchor_distance_ft({"geometry": [[center_lat, center_lon]]}, [pt]) for pt in anchor_points)
        span_hint = float(span_ft or 0.0)
        farthest_hint = float(farthest_station or 0.0)
        locality_radius_ft = max(140.0, min(650.0, max(spread_ft + 120.0, min(span_hint * 0.22, 260.0), min(farthest_hint * 0.18, 240.0))))
        corridor_bbox_margin_ft = max(120.0, min(320.0, (spread_ft or 0.0) * 0.35 + 120.0))
        corridor_bbox = _corridor_bbox_from_anchors(anchor_points, corridor_bbox_margin_ft)

    filtered: List[Dict[str, Any]] = []
    chain_seed_candidates: List[Dict[str, Any]] = []
    rejected_by_reason: Dict[str, int] = {
        "role": 0,
        "length_short": 0,
        "length_long": 0,
        "span_delta": 0,
        "anchor_distance": 0,
        "bbox": 0,
    }
    for route in route_catalog:
        route_length = float(route.get("total_length_ft", 0.0) or 0.0)
        role = str(route.get("role") or "other")
        keep = True
        locality_distance_ft = None
        locality_score = 0.0
        anchor_hit = False
        inside_points = 0
        total_points = 0
        bbox_ratio = 0.0
        endpoint_inside = False

        role_preferred = role in preferred_roles if preferred_roles else True
        if preferred_roles and not role_preferred:
            # Do not hard-reject by role alone anymore.
            # Some real matches are mixed local chains that include nearby support segments.
            rejected_by_reason["role"] += 1
        if keep and farthest_station is not None and route_length < farthest_station - 25.0:
            rejected_by_reason["length_short"] += 1
            keep = False
        if keep and farthest_station is not None:
            long_cap = max(float(farthest_station) * 1.6, float(span_ft or 0.0) * 1.95, 950.0)
            if route_length > long_cap:
                rejected_by_reason["length_long"] += 1
                keep = False
        if keep and span_ft is not None and span_ft > 0:
            max_span_delta = max(140.0, min(360.0, float(span_ft) * 0.28 + 40.0))
            if abs(route_length - float(span_ft)) > max_span_delta:
                rejected_by_reason["span_delta"] += 1
                keep = False
        if keep and anchor_points:
            locality_distance_ft = _route_min_anchor_distance_ft(route, anchor_points)
            anchor_hit = bool(locality_distance_ft <= float(locality_radius_ft or 0.0))
            if not anchor_hit:
                rejected_by_reason["anchor_distance"] += 1
                keep = False
            else:
                locality_score = max(0.0, 1.0 - (locality_distance_ft / max(float(locality_radius_ft or 1.0), 1.0)))
        if keep and corridor_bbox:
            inside_points, total_points = _route_points_inside_bbox(route, corridor_bbox)
            bbox_ratio = (inside_points / max(total_points, 1)) if total_points else 0.0
            geometry = route.get("geometry", []) or []
            endpoint_inside = False
            if geometry:
                min_lat, max_lat, min_lon, max_lon = corridor_bbox
                for lat, lon in (geometry[0], geometry[-1]):
                    if min_lat <= float(lat) <= max_lat and min_lon <= float(lon) <= max_lon:
                        endpoint_inside = True
                        break
            if bbox_ratio < 0.55 and not endpoint_inside:
                rejected_by_reason["bbox"] += 1
                keep = False
            # Mixed local chains sometimes include one support segment that clips the corridor window
            # only at an endpoint. Preserve that information for later synthetic-chain assembly.

        chain_seed_ok = True
        if chain_seed_ok and anchor_points and locality_distance_ft is not None and locality_radius_ft is not None:
            if locality_distance_ft <= max(float(locality_radius_ft) * 1.35, float(locality_radius_ft) + 90.0):
                if corridor_bbox is None or bbox_ratio >= 0.28 or endpoint_inside:
                    seed_clone = dict(route)
                    seed_clone["_corridor_gate_distance_ft"] = round(float(locality_distance_ft), 2) if locality_distance_ft is not None else None
                    seed_clone["_corridor_locality_score"] = round(float(locality_score), 4)
                    seed_clone["_corridor_anchor_hit"] = anchor_hit
                    seed_clone["_corridor_bbox_ratio"] = round(float(bbox_ratio), 4)
                    seed_clone["_corridor_endpoint_inside"] = endpoint_inside
                    seed_clone["_generic_name_penalty"] = round(_generic_route_name_penalty(route) * 1.4, 2)
                    chain_seed_candidates.append(seed_clone)

        if keep:
            clone = dict(route)
            role_rank = preferred_roles.index(role) if role in preferred_roles else len(preferred_roles)
            role_bonus = max(0.0, float((len(preferred_roles) - role_rank) * 85.0)) if preferred_roles and role in preferred_roles else 0.0
            clone["_corridor_gate_distance_ft"] = round(float(locality_distance_ft), 2) if locality_distance_ft is not None else None
            clone["_corridor_locality_score"] = round(float(locality_score), 4)
            clone["_corridor_anchor_hit"] = anchor_hit
            clone["_corridor_bbox_ratio"] = round(float(bbox_ratio), 4)
            clone["_corridor_endpoint_inside"] = endpoint_inside
            clone["_generic_name_penalty"] = round(_generic_route_name_penalty(route) * 1.6, 2)
            clone["_corridor_role_bonus"] = round(role_bonus, 2)
            filtered.append(clone)

    filtered.sort(key=lambda route: (
        -float(route.get("_corridor_locality_score", 0.0) or 0.0),
        -float(route.get("_corridor_bbox_ratio", 0.0) or 0.0),
        float(route.get("_corridor_gate_distance_ft") or 10**9),
        float(route.get("_generic_name_penalty", 0.0) or 0.0),
        abs(float(route.get("total_length_ft", 0.0) or 0.0) - float(span_ft or 0.0)),
    ))

    cluster_debug: Dict[str, Any] = {
        "cluster_radius_ft": None,
        "cluster_seed_route_id": None,
        "cluster_size_before_cap": len(filtered),
        "cluster_size_after_cap": len(filtered),
    }
    too_broad = False
    duplicate_debug: Dict[str, Any] = {"duplicate_groups_pruned": 0, "duplicate_routes_removed": 0}
    if not anchor_points:
        filtered = [dict(route) for route in route_catalog]
        for route in filtered:
            route.setdefault("_corridor_gate_distance_ft", None)
            route.setdefault("_corridor_locality_score", 0.0)
            route.setdefault("_corridor_anchor_hit", False)
            route.setdefault("_corridor_bbox_ratio", 0.0)
            route.setdefault("_corridor_endpoint_inside", False)
            route.setdefault("_generic_name_penalty", round(_generic_route_name_penalty(route) * 1.6, 2))
            route.setdefault("_corridor_role_bonus", 0.0)
        filtered.sort(key=lambda route: (
            float(route.get("_generic_name_penalty", 0.0) or 0.0),
            abs(float(route.get("total_length_ft", 0.0) or 0.0) - float(span_ft or farthest_station or 0.0)),
            str(route.get("route_name") or ""),
        ))
    elif len(filtered) > 8:
        filtered, cluster_debug = _cluster_routes_to_local_family(filtered, span_ft)
        too_broad = len(filtered) > 8

    synthetic_source = filtered if len(filtered) >= 2 else (chain_seed_candidates or filtered)
    synthetic_chains = _synthesize_local_route_chains(synthetic_source, span_ft, farthest_station)
    if synthetic_chains:
        filtered.extend(synthetic_chains)
        filtered.sort(key=lambda route: (
            -int(bool(route.get("_synthetic_chain"))),
            -float(route.get("_corridor_locality_score", 0.0) or 0.0),
            -float(route.get("_corridor_bbox_ratio", 0.0) or 0.0),
            abs(float(route.get("total_length_ft", 0.0) or 0.0) - float(span_ft or farthest_station or 0.0)),
            float(route.get("_corridor_gate_distance_ft") or 10**9),
        ))

    filtered, duplicate_debug = _prune_duplicate_generic_routes(filtered, span_ft)
    if len(filtered) > 6:
        too_broad = True

    debug.update({
        "matched_pdf_sheet": best.get("sheet_id"),
        "matched_chain_id": best.get("chain_id"),
        "matched_corridor_ids": [best.get("corridor_id")],
        "matched_station_range": best.get("matched_station_range") or f"{best.get('station_start')} -> {best.get('station_end')}",
        "matched_street_names": best.get("street_names", []),
        "station_reset_detected": bool(best.get("station_reset_detected")),
        "kmz_candidates_before_filter": len(route_catalog),
        "kmz_candidates_after_filter": len(filtered),
        "corridor_anchor_tokens": anchor_tokens[:20],
        "corridor_anchor_point_ids": matched_point_ids[:25],
        "corridor_anchor_count": len(anchor_points),
        "corridor_locality_radius_ft": round(float(locality_radius_ft), 2) if locality_radius_ft is not None else None,
        "corridor_bbox_margin_ft": round(float(corridor_bbox_margin_ft), 2) if corridor_bbox_margin_ft is not None else None,
        "corridor_rejections": rejected_by_reason,
        "corridor_cluster_radius_ft": cluster_debug.get("cluster_radius_ft"),
        "corridor_cluster_seed_route_id": cluster_debug.get("cluster_seed_route_id"),
        "corridor_cluster_size_before_cap": cluster_debug.get("cluster_size_before_cap"),
        "corridor_cluster_size_after_cap": cluster_debug.get("cluster_size_after_cap"),
        "duplicate_groups_pruned": duplicate_debug.get("duplicate_groups_pruned"),
        "duplicate_routes_removed": duplicate_debug.get("duplicate_routes_removed"),
        "synthetic_chain_candidates": len(synthetic_chains),
        "chain_seed_candidates": len(chain_seed_candidates),
        "corridor_gate_status": "too_broad" if too_broad else (("no_anchor_fallback" if not anchor_points else "hard_gated") if filtered else "no_local_subset"),
    })

    return filtered, {
        "filter_mode": "corridor_graph_hard_gate",
        "preferred_roles": preferred_roles,
        "kept_route_ids": [str(route.get("route_id") or "") for route in filtered[:50]],
        "candidate_count_before_filter": len(route_catalog),
        "candidate_count_after_filter": len(filtered),
        "needs_review": bool(too_broad or not filtered),
        "debug": debug,
    }

