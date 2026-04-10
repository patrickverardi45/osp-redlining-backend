import json
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_coordinates(coord_text: str):
    coords = []
    raw_parts = coord_text.strip().split()

    for part in raw_parts:
        pieces = part.split(",")
        if len(pieces) >= 2:
            lon = float(pieces[0])
            lat = float(pieces[1])
            coords.append([lon, lat])

    return coords


def extract_kml_text(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() == ".kml":
        return path.read_text(encoding="utf-8", errors="ignore")

    if path.suffix.lower() == ".kmz":
        with zipfile.ZipFile(path, "r") as kmz:
            for name in kmz.namelist():
                if name.lower().endswith(".kml"):
                    with kmz.open(name) as f:
                        return f.read().decode("utf-8", errors="ignore")

    raise ValueError("Unsupported GIS file type. Use .kml or .kmz")


def process_gis(file_path: str):
    try:
        kml_text = extract_kml_text(file_path)
        root = ET.fromstring(kml_text)

        ns = {"kml": "http://www.opengis.net/kml/2.2"}

        features = []
        all_points = []

        placemarks = root.findall(".//kml:Placemark", ns)

        for placemark in placemarks:
            name_el = placemark.find("kml:name", ns)
            name = name_el.text.strip() if name_el is not None and name_el.text else "Unnamed Route"

            line_strings = placemark.findall(".//kml:LineString", ns)
            for line in line_strings:
                coord_el = line.find("kml:coordinates", ns)
                if coord_el is None or not coord_el.text:
                    continue

                coords = parse_coordinates(coord_el.text)
                if not coords:
                    continue

                all_points.extend(coords)

                features.append({
                    "type": "Feature",
                    "properties": {
                        "name": name
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    }
                })

        if not features:
            return {
                "status": "error",
                "message": "No LineString routes found in the KML/KMZ file."
            }

        lons = [pt[0] for pt in all_points]
        lats = [pt[1] for pt in all_points]

        bbox = {
            "min_lon": min(lons),
            "min_lat": min(lats),
            "max_lon": max(lons),
            "max_lat": max(lats),
        }

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        geojson_path = OUTPUT_DIR / "latest_route.geojson"
        geojson_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")

        summary_path = OUTPUT_DIR / "latest_route_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "route_count": len(features),
                    "point_count": len(all_points),
                    "bbox": bbox,
                },
                indent=2
            ),
            encoding="utf-8"
        )

        return {
            "status": "GIS route processed",
            "route_count": len(features),
            "point_count": len(all_points),
            "bbox": bbox,
            "geojson_saved_to": str(geojson_path),
            "summary_saved_to": str(summary_path),
            "map_preview_url": "/map-preview"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }