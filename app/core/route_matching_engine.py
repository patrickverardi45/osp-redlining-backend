import re

def normalize_station(station_str):
    if not station_str:
        return None
    station_str = station_str.strip().replace(" ", "")
    match = re.match(r"(\d+)\+(\d+)", station_str)
    if not match:
        return None
    return int(match.group(1)) * 100 + int(match.group(2))

def extract_bore_log_range(bore_log):
    stations = []
    for row in bore_log.get("rows", []):
        val = normalize_station(row.get("station"))
        if val is not None:
            stations.append(val)
    if not stations:
        return None, None
    return min(stations), max(stations)

def score_route(bore_start, bore_end, route):
    score = 0
    breakdown = {}
    rs = route.get("start_station")
    re_ = route.get("end_station")

    if rs is None or re_ is None:
        return 0, {"invalid": True}

    overlap = max(0, min(bore_end, re_) - max(bore_start, rs))
    bore_len = bore_end - bore_start

    overlap_ratio = overlap / bore_len if bore_len else 0
    overlap_score = overlap_ratio * 60
    score += overlap_score

    route_len = re_ - rs
    length_diff = abs(route_len - bore_len)
    length_score = max(0, 30 - (length_diff / route_len) * 30) if route_len else 0
    score += length_score

    name_score = 10 if route.get("name") else 0
    score += name_score

    return score, {
        "overlap": overlap_score,
        "length": length_score,
        "name": name_score
    }

def match_bore_to_routes(bore_log, routes):
    start, end = extract_bore_log_range(bore_log)
    if start is None:
        return {"status": "rejected", "reason": "no_stations"}

    results = []
    for r in routes:
        s, b = score_route(start, end, r)
        results.append({"route_id": r.get("id"), "score": s, "breakdown": b})

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0] if results else None

    if not best or best["score"] < 40:
        return {"status": "rejected", "candidates": results}

    return {"status": "matched", "best": best, "candidates": results}
