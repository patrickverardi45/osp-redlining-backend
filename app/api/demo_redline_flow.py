import pandas as pd
import streamlit as st
import folium
from streamlit.components.v1 import html
import math

from demo_generator import build_demo_route, generate_realistic_bore_demo


st.set_page_config(page_title="OSP Redline Demo Flow", layout="wide")


def render_map(m):
    html(m._repr_html_(), height=750)


def compute_center(coords):
    lat = sum(c[0] for c in coords) / len(coords)
    lon = sum(c[1] for c in coords) / len(coords)
    return lat, lon


# 🔥 TRUE PERPENDICULAR OFFSET FUNCTION
def offset_line(coords, offset=0.00008):
    new_coords = []

    for i in range(len(coords)):
        lat, lon = coords[i]

        if i < len(coords) - 1:
            lat2, lon2 = coords[i + 1]
        else:
            lat2, lon2 = coords[i - 1]

        dx = lon2 - lon
        dy = lat2 - lat

        length = math.hypot(dx, dy)
        if length == 0:
            new_coords.append([lat, lon])
            continue

        # perpendicular vector
        px = -dy / length
        py = dx / length

        new_lat = lat + py * offset
        new_lon = lon + px * offset

        new_coords.append([new_lat, new_lon])

    return new_coords


def create_map(bore_df, redlines_df, accepted_ids, show_points):

    plan = bore_df[["plan_lat", "plan_lon"]].values.tolist()
    asbuilt = bore_df[["asbuilt_lat", "asbuilt_lon"]].values.tolist()

    center = compute_center(plan)
    m = folium.Map(location=center, zoom_start=19)

    # --------------------------------------------------
    # OFFSET BOTH LINES IN OPPOSITE DIRECTIONS
    # --------------------------------------------------
    plan_offset = offset_line(plan, 0.00010)
    asbuilt_offset = offset_line(asbuilt, -0.00002)

    # -----------------------------
    # PLANNED (blue dashed)
    # -----------------------------
    folium.PolyLine(
        plan_offset,
        color="#2563eb",
        weight=5,
        dash_array="6,6",
        opacity=0.9,
        tooltip="Planned Route",
    ).add_to(m)

    # -----------------------------
    # BASE AS-BUILT (gray center)
    # -----------------------------
    folium.PolyLine(
        asbuilt_offset,
        color="#6b7280",
        weight=6,
        opacity=0.6,
    ).add_to(m)

    # -----------------------------
    # REDLINE SEGMENTS (clear + separated)
    # -----------------------------
    for _, event in redlines_df.iterrows():

        pts = bore_df[bore_df["event_id"] == event["event_id"]]
        if pts.empty:
            continue

        coords = pts[["asbuilt_lat", "asbuilt_lon"]].values.tolist()
        coords = offset_line(coords, -0.00002)

        mid = pts.iloc[len(pts)//2]

        is_accepted = event["event_id"] in accepted_ids

        if is_accepted:
            color = "#16a34a"
            label = "APPROVED"
        else:
            color = "#dc2626"
            label = "PENDING"

        folium.PolyLine(
            coords,
            color=color,
            weight=10,
            opacity=0.95,
            tooltip=f'{label} | {event["start_station_label"]} → {event["end_station_label"]}',
        ).add_to(m)

        folium.CircleMarker(
            location=[mid["asbuilt_lat"], mid["asbuilt_lon"]],
            radius=8,
            color="#111",
            fill=True,
            fill_color=color,
            fill_opacity=1,
        ).add_to(m)

    # -----------------------------
    # OPTIONAL POINTS (tiny)
    # -----------------------------
    if show_points:
        for _, row in bore_df.iterrows():
            folium.CircleMarker(
                [row["asbuilt_lat"], row["asbuilt_lon"]],
                radius=1.5,
                color="#15803d" if not row["flagged"] else "#b91c1c",
                fill=True,
                fill_opacity=0.5,
            ).add_to(m)

    return m


def main():
    st.title("OSP Fiber Redlining / Construction QA — Payment View Demo")

    with st.sidebar:
        seed = st.number_input("Seed", value=42)
        spacing = st.slider("Spacing (ft)", 10, 50, 25)
        show_points = st.checkbox("Show Points", True)

        st.markdown("---")
        accepted_raw = st.text_input("Approved Event IDs", "0,1")

    accepted_ids = set()
    for x in accepted_raw.split(","):
        if x.strip().isdigit():
            accepted_ids.add(int(x.strip()))

    route = build_demo_route()

    bore_df, redlines_df = generate_realistic_bore_demo(
        route_latlon=route,
        spacing_ft=spacing,
        seed=seed,
    )

    st.markdown("### 💰 Billable Map View")

    m = create_map(bore_df, redlines_df, accepted_ids, show_points)
    render_map(m)

    total_len = bore_df["station_ft"].max()
    approved = sum([1 for e in redlines_df["event_id"] if e in accepted_ids])
    pending = len(redlines_df) - approved

    st.markdown("### Job Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Footage", f"{int(total_len)} ft")
    col2.metric("Approved Events", approved)
    col3.metric("Pending Events", pending)

    if pending == 0:
        st.success("✅ READY FOR BILLING")
    else:
        st.warning("⚠️ Pending review required")

    st.dataframe(redlines_df, width="stretch")


if __name__ == "__main__":
    main()