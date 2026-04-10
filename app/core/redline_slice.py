import math

def haversine(p1, p2):
    lat1, lon1 = p1
    lat2, lon2 = p2
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def slice_route(points, start_ft, end_ft):
    sliced = []
    total = 0.0
    for i in range(len(points) - 1):
        seg_ft = haversine(points[i], points[i + 1]) / 0.3048
        next_total = total + seg_ft
        if next_total >= start_ft and total <= end_ft:
            if not sliced:
                sliced.append(points[i])
            sliced.append(points[i + 1])
        total = next_total
    return sliced
