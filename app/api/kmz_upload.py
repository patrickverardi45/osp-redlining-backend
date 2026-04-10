from fastapi import APIRouter, UploadFile, File
import zipfile
import tempfile
import xml.etree.ElementTree as ET

from app.core import pipeline_state

router = APIRouter()

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def parse_coordinates_text(text):
    coords = []

    if not text:
        return coords

    chunks = text.strip().split()

    for chunk in chunks:
        parts = chunk.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lon, lat))
            except ValueError:
                pass

    return coords


def extract_route_from_kmz(file_path):
    """
    Extract only LineString-style route candidates from the KMZ,
    then choose the longest one as the main route.
    """
    line_candidates = []

    with zipfile.ZipFile(file_path, "r") as z:
        kml_name = None

        for name in z.namelist():
            if name.lower().endswith(".kml"):
                kml_name = name
                break

        if not kml_name:
            return [], 0

        with z.open(kml_name) as f:
            xml_data = f.read()

    root = ET.fromstring(xml_data)

    # 1) Plain LineString
    for elem in root.findall(".//kml:LineString/kml:coordinates", KML_NS):
        coords = parse_coordinates_text(elem.text)
        if len(coords) >= 2:
            line_candidates.append(coords)

    # 2) gx:Track fallback (some KMZ/KML files use gx:coord)
    gx_ns = {
        "kml": "http://www.opengis.net/kml/2.2",
        "gx": "http://www.google.com/kml/ext/2.2",
    }

    for track in root.findall(".//gx:Track", gx_ns):
        track_coords = []
        for coord_elem in track.findall("gx:coord", gx_ns):
            text = (coord_elem.text or "").strip()
            parts = text.split()
            if len(parts) >= 2:
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    track_coords.append((lon, lat))
                except ValueError:
                    pass
        if len(track_coords) >= 2:
            line_candidates.append(track_coords)

    if not line_candidates:
        return [], 0

    # Pick the longest candidate route
    best_route = max(line_candidates, key=len)

    return best_route, len(line_candidates)


@router.post("/upload-kmz")
async def upload_kmz(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    route_coords, candidate_count = extract_route_from_kmz(tmp_path)

    pipeline_state.CURRENT_ROUTE = route_coords
    pipeline_state.CURRENT_ROUTE_NAME = file.filename

    return {
        "uploaded_file": file.filename,
        "route_candidates_found": candidate_count,
        "points_extracted": len(route_coords),
        "sample": route_coords[:10],
        "route_loaded": len(route_coords) > 0
    }