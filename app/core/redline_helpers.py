from app.core.route_matching_engine import normalize_station

def get_bore_range(bore_log):
    stations = []
    for row in bore_log.get("rows", []):
        val = normalize_station(row.get("station"))
        if val is not None:
            stations.append(val)
    if not stations:
        return None, None
    return min(stations), max(stations)

def format_station_label(value):
    if value is None:
        return ""
    value = int(round(float(value)))
    left = value // 100
    right = value % 100
    return f"{left:02d}+{right:02d}"
