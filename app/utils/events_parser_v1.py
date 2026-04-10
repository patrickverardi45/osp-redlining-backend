# events_parser_v1.py
# Simple Event Parser V1 for OSP Redlining System

import re

def parse_station(text):
    match = re.search(r'(?:STA\s*)?(\d+)\+(\d+)', text)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2))
        return major * 100 + minor
    return None

def classify_event(text):
    t = text.lower()
    if "rock" in t:
        return "rock", "high", True
    if "damage" in t:
        return "damage", "critical", True
    if "coupler" in t:
        return "coupler", "medium", True
    if "sod" in t or "restoration" in t:
        return "restoration", "medium", False
    if "homeowner" in t:
        return "complaint", "high", False
    return "note", "low", False

def parse_offset(text):
    match = re.search(r"(\d+)\s*[’']?\s*(?:ft|feet)?\s*from station\s*(\d+\+\d+)", text.lower())
    if match:
        distance = int(match.group(1))
        station_raw = match.group(2)
        parts = station_raw.split("+")
        base_station = int(parts[0]) * 100 + int(parts[1])
        return distance, base_station
    return None, None

def parse_events(lines):
    events = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        station = parse_station(line)
        event_type, severity, billable = classify_event(line)
        offset_distance, offset_station = parse_offset(line)

        event = {
            "type": event_type,
            "description": line,
            "station": station,
            "offset": {
                "distance_ft": offset_distance,
                "from_station": offset_station
            },
            "severity": severity,
            "billable": billable
        }

        events.append(event)

    return events


if __name__ == "__main__":
    sample = [
        "Rock Adder - STA 2+12",
        "Coupler at 50’ from station 2+72",
        "No structures set to station 3+50",
        "Homeowner complaint about yard",
        "Restoration - add sod at 1+57"
    ]

    results = parse_events(sample)

    for r in results:
        print(r)
