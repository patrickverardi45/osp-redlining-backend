def _wrong_street_analysis(
    best: Dict[str, Any],
    second: Optional[Dict[str, Any]],
    segments: List[Segment],
    point_heading_deg: Optional[float],
    settings: Dict[str, Any],
) -> Dict[str, bool]:
    wrong_street = False

    if best is None:
        return {"wrong_street": wrong_street}

    # Check perpendicular offset
    if best.perpendicular_offset_ft > settings["wrong_street_perpendicular_offset_threshold"]:
        # Check heading difference using circular angle difference
        segment_heading_deg = segments[best.segment_index].heading_deg
        heading_diff = abs((point_heading_deg - segment_heading_deg + 180) % 360 - 180)
        
        # Allow a small heading difference (e.g., 30 degrees) before marking as wrong street
        if heading_diff > settings["wrong_street_heading_threshold"]:
            wrong_street = True

    return {"wrong_street": wrong_street}
