import zipfile
import xml.etree.ElementTree as ET
import math

def haversine(p1, p2):
    lat1, lon1 = p1
    lat2, lon2 = p2
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def extract_routes_from_kmz(file_path):
    with zipfile.ZipFile(file_path, 'r') as z:
        kml_file = [f for f in z.namelist() if f.endswith('.kml')][0]
        root = ET.fromstring(z.read(kml_file))

    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    routes = []
    idx = 1

    for coords in root.findall(".//kml:LineString/kml:coordinates", ns):
        pts = []
        for c in coords.text.strip().split():
            lon, lat, *_ = map(float, c.split(","))
            pts.append((lat, lon))

        length = sum(haversine(pts[i], pts[i+1]) for i in range(len(pts)-1))

        routes.append({
            "id": f"route_{idx}",
            "points": pts,
            "start_station": 0,
            "end_station": int(length / 0.3048)
        })
        idx += 1

    return routes
