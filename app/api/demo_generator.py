import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import pandas as pd


EARTH_METERS_PER_DEG_LAT = 111_320.0


@dataclass
class RoutePoint:
    lat: float
    lon: float


def _meters_per_deg_lon(lat_deg: float) -> float:
    return EARTH_METERS_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def _to_local_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """
    Convert lat/lon to local X/Y meters using a simple local tangent approximation.
    Good enough for short demo routes.
    """
    x = (lon - origin_lon) * _meters_per_deg_lon(origin_lat)
    y = (lat - origin_lat) * EARTH_METERS_PER_DEG_LAT
    return x, y


def _to_latlon(x: float, y: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    lat = origin_lat + (y / EARTH_METERS_PER_DEG_LAT)
    lon = origin_lon + (x / _meters_per_deg_lon(origin_lat))
    return lat, lon


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _interpolate(a: Tuple[float, float], b: Tuple[float, float], t: float) -> Tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _resample_polyline_latlon(route_latlon: List[Tuple[float, float]], spacing_ft: float = 25.0) -> List[Dict[str, Any]]:
    """
    Resample a route polyline into evenly spaced points.
    Returns rows with local x/y meters plus station info.
    """
    if len(route_latlon) < 2:
        raise ValueError("Route must have at least 2 points.")

    origin_lat, origin_lon = route_latlon[0]
    spacing_m = spacing_ft * 0.3048

    xy = [_to_local_xy(lat, lon, origin_lat, origin_lon) for lat, lon in route_latlon]

    segments = []
    total_len = 0.0
    for i in range(len(xy) - 1):
        a = xy[i]
        b = xy[i + 1]
        seg_len = _distance(a, b)
        if seg_len <= 0:
            continue
        segments.append((a, b, seg_len))
        total_len += seg_len

    if total_len <= 0:
        raise ValueError("Route length is zero.")

    output = []
    target_d = 0.0
    running_d = 0.0

    seg_index = 0
    seg_start_d = 0.0

    while target_d <= total_len + 0.001:
        while seg_index < len(segments):
            a, b, seg_len = segments[seg_index]
            seg_end_d = seg_start_d + seg_len
            if target_d <= seg_end_d or seg_index == len(segments) - 1:
                break
            seg_start_d = seg_end_d
            seg_index += 1

        a, b, seg_len = segments[seg_index]
        local_d = min(max(target_d - seg_start_d, 0.0), seg_len)
        t = 0.0 if seg_len == 0 else local_d / seg_len
        x, y = _interpolate(a, b, t)

        # unit tangent
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        mag = math.hypot(dx, dy)
        tx = 1.0 if mag == 0 else dx / mag
        ty = 0.0 if mag == 0 else dy / mag

        # left normal
        nx = -ty
        ny = tx

        lat, lon = _to_latlon(x, y, origin_lat, origin_lon)

        output.append(
            {
                "station_m": target_d,
                "station_ft": target_d / 0.3048,
                "plan_x_m": x,
                "plan_y_m": y,
                "plan_lat": lat,
                "plan_lon": lon,
                "tangent_x": tx,
                "tangent_y": ty,
                "normal_x": nx,
                "normal_y": ny,
            }
        )

        target_d += spacing_m

    return output


def build_demo_route() -> List[Tuple[float, float]]:
    """
    Built-in demo route near Brenham-style geometry.
    This gives a believable customer demo route with mild bends
    instead of a perfectly straight line.
    """
    return [
        (30.16520, -96.40380),
        (30.16555, -96.40325),
        (30.16595, -96.40270),
        (30.16630, -96.40210),
        (30.16675, -96.40155),
        (30.16710, -96.40095),
        (30.16745, -96.40035),
        (30.16782, -96.39980),
        (30.16810, -96.39925),
    ]


def _build_deviation_windows(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """
    Create a few realistic deviation windows.
    They are short, clustered, and stay near the route.
    """
    if n < 25:
        return []

    windows = []

    count = rng.choice([2, 3, 3, 4])

    protected_margin = 5
    used_ranges = []

    for _ in range(count):
        attempts = 0
        while attempts < 30:
            start = rng.randint(protected_margin, max(protected_margin + 1, n - 12))
            length = rng.randint(3, 7)
            end = min(n - protected_margin, start + length)

            overlaps = False
            for a, b in used_ranges:
                if not (end < a or start > b):
                    overlaps = True
                    break

            if not overlaps:
                used_ranges.append((start, end))

                dev_type = rng.choices(
                    ["minor_offset", "utility_avoidance", "shallow_adjustment", "deep_adjustment"],
                    weights=[45, 25, 15, 15],
                    k=1,
                )[0]

                if dev_type == "minor_offset":
                    max_offset_ft = rng.uniform(1.0, 3.5)
                    depth_delta_ft = rng.uniform(-0.2, 0.2)
                elif dev_type == "utility_avoidance":
                    max_offset_ft = rng.uniform(3.0, 7.0)
                    depth_delta_ft = rng.uniform(0.0, 1.2)
                elif dev_type == "shallow_adjustment":
                    max_offset_ft = rng.uniform(1.0, 2.5)
                    depth_delta_ft = rng.uniform(-1.0, -0.4)
                else:
                    max_offset_ft = rng.uniform(1.5, 4.0)
                    depth_delta_ft = rng.uniform(0.8, 2.0)

                windows.append(
                    {
                        "start": start,
                        "end": end,
                        "type": dev_type,
                        "max_offset_ft": max_offset_ft,
                        "depth_delta_ft": depth_delta_ft,
                        "direction": rng.choice([-1, 1]),
                    }
                )
                break

            attempts += 1

    return sorted(windows, key=lambda x: x["start"])


def generate_realistic_bore_demo(
    route_latlon: List[Tuple[float, float]] = None,
    spacing_ft: float = 25.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      bore_df: point-by-point plan vs as-built bore data
      redlines_df: grouped redline events/windows
    """
    rng = random.Random(seed)

    if route_latlon is None:
        route_latlon = build_demo_route()

    sampled = _resample_polyline_latlon(route_latlon, spacing_ft=spacing_ft)
    n = len(sampled)

    windows = _build_deviation_windows(n, rng)
    window_lookup = {}
    for idx, w in enumerate(windows):
        for i in range(w["start"], w["end"] + 1):
            window_lookup[i] = (idx, w)

    rows = []
    current_side_bias_ft = 0.0

    for i, p in enumerate(sampled):
        station_ft = p["station_ft"]

        # base noise stays tight to route
        lateral_noise_ft = rng.uniform(-0.45, 0.45)
        longitudinal_noise_ft = rng.uniform(-0.85, 0.85)
        depth_noise_ft = rng.uniform(-0.20, 0.20)

        # smooth drift so path feels real, not jittery
        current_side_bias_ft = (0.85 * current_side_bias_ft) + rng.uniform(-0.18, 0.18)
        lateral_ft = lateral_noise_ft + current_side_bias_ft

        deviation_type = "on_plan"
        event_id = None
        is_flagged = False

        if i in window_lookup:
            event_id, w = window_lookup[i]
            deviation_type = w["type"]

            center = (w["start"] + w["end"]) / 2.0
            half = max((w["end"] - w["start"]) / 2.0, 1.0)
            taper = 1.0 - abs(i - center) / half
            taper = max(0.15, min(1.0, taper))

            lateral_ft += w["direction"] * w["max_offset_ft"] * taper
            depth_noise_ft += w["depth_delta_ft"] * taper
            is_flagged = True

        lateral_m = lateral_ft * 0.3048
        longitudinal_m = longitudinal_noise_ft * 0.3048

        as_x = (
            p["plan_x_m"]
            + (p["normal_x"] * lateral_m)
            + (p["tangent_x"] * longitudinal_m)
        )
        as_y = (
            p["plan_y_m"]
            + (p["normal_y"] * lateral_m)
            + (p["tangent_y"] * longitudinal_m)
        )

        origin_lat, origin_lon = route_latlon[0]
        as_lat, as_lon = _to_latlon(as_x, as_y, origin_lat, origin_lon)

        plan_depth_ft = 5.5 + 0.35 * math.sin(i / 6.0)
        as_built_depth_ft = plan_depth_ft + depth_noise_ft

        offset_ft = math.hypot(as_x - p["plan_x_m"], as_y - p["plan_y_m"]) / 0.3048

        rows.append(
            {
                "station_ft": round(station_ft, 1),
                "station_label": f"{int(station_ft // 100)}+{int(station_ft % 100):02d}",
                "plan_lat": p["plan_lat"],
                "plan_lon": p["plan_lon"],
                "asbuilt_lat": as_lat,
                "asbuilt_lon": as_lon,
                "plan_depth_ft": round(plan_depth_ft, 2),
                "asbuilt_depth_ft": round(as_built_depth_ft, 2),
                "depth_delta_ft": round(as_built_depth_ft - plan_depth_ft, 2),
                "offset_ft": round(offset_ft, 2),
                "deviation_type": deviation_type,
                "flagged": is_flagged,
                "event_id": None if event_id is None else int(event_id),
            }
        )

    bore_df = pd.DataFrame(rows)

    redline_events = []
    for idx, w in enumerate(windows):
        window_df = bore_df[bore_df["event_id"] == idx].copy()
        if window_df.empty:
            continue

        max_offset_ft = float(window_df["offset_ft"].max())
        avg_offset_ft = float(window_df["offset_ft"].mean())
        max_depth_delta_ft = float(window_df["depth_delta_ft"].abs().max())

        reason = {
            "minor_offset": "Minor field alignment adjustment",
            "utility_avoidance": "Adjusted alignment to avoid field obstruction / utility conflict",
            "shallow_adjustment": "Depth adjusted due to shallow field condition",
            "deep_adjustment": "Depth increased to maintain clearance / constructability",
        }.get(w["type"], "Field deviation")

        redline_events.append(
            {
                "event_id": idx,
                "start_station_ft": float(window_df["station_ft"].min()),
                "end_station_ft": float(window_df["station_ft"].max()),
                "start_station_label": window_df.iloc[0]["station_label"],
                "end_station_label": window_df.iloc[-1]["station_label"],
                "deviation_type": w["type"],
                "reason": reason,
                "max_offset_ft": round(max_offset_ft, 2),
                "avg_offset_ft": round(avg_offset_ft, 2),
                "max_depth_delta_ft": round(max_depth_delta_ft, 2),
                "status": "Pending Foreman Review",
                "pay_item_impact": "Review for as-built acceptance",
            }
        )

    redlines_df = pd.DataFrame(redline_events)

    return bore_df, redlines_df