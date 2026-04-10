import re
import pandas as pd


# ==============================
# STATION FORMAT DETECTION
# ==============================

def is_station_format(value):
    """
    Check if value looks like station format:
    14+20, 123+45, 1+5, etc.
    """
    if pd.isna(value):
        return False

    value = str(value).strip()
    pattern = r"^\d{1,4}\+\d{1,2}$"
    return bool(re.match(pattern, value))


# ==============================
# STATION → NUMERIC CONVERSION
# ==============================

def station_to_number(station):
    """
    Convert station string:
    14+20 -> 1420
    """
    try:
        station = str(station).strip()
        parts = station.split("+")
        if len(parts) != 2:
            return None
        return int(parts[0]) * 100 + int(parts[1])
    except Exception:
        return None


# ==============================
# HEADER NORMALIZATION
# ==============================

def normalize_header(header):
    """
    Lowercase and remove spaces/symbols
    """
    return re.sub(r"[^a-z0-9]", "", str(header).lower())


# ==============================
# HELPER: CLEAN DATAFRAME
# ==============================

def clean_dataframe(df):
    """
    Remove fully empty rows/columns
    """
    if df is None or df.empty:
        return df

    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df


# ==============================
# HELPER: FIND STATION VALUE TO RIGHT
# ==============================

def find_station_value_to_right(df, row_idx, col_idx, max_lookahead=6):
    """
    Starting from a label cell like 'Start Station',
    scan right in the same row to find first station-formatted value.
    """
    max_col = df.shape[1]

    for offset in range(1, max_lookahead + 1):
        target_col = col_idx + offset
        if target_col >= max_col:
            break

        value = df.iat[row_idx, target_col]
        if is_station_format(value):
            return str(value).strip()

    return None


# ==============================
# FORM-STYLE STATION DETECTION
# ==============================

def detect_form_station_pair(df):
    """
    Detect form-style station layout such as:
    Start Station .... 19+76 .... End Station .... 20+47
    """
    if df is None or df.empty:
        return {
            "success": False,
            "error": "Empty dataframe."
        }

    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            cell_value = df.iat[r, c]
            if pd.isna(cell_value):
                continue

            norm = normalize_header(cell_value)

            # Look for start station label
            if norm in ["startstation", "beginstation", "fromstation"]:
                start_station = find_station_value_to_right(df, r, c)

                # Search same row for end station label
                end_station = None
                for c2 in range(c + 1, df.shape[1]):
                    cell2 = df.iat[r, c2]
                    if pd.isna(cell2):
                        continue

                    norm2 = normalize_header(cell2)
                    if norm2 in ["endstation", "tostation"]:
                        end_station = find_station_value_to_right(df, r, c2)
                        break

                if start_station and end_station:
                    return {
                        "success": True,
                        "mode": "form",
                        "start_station": start_station,
                        "end_station": end_station
                    }

    return {
        "success": False,
        "error": "No form-style start/end station pair found."
    }


# ==============================
# TABULAR COLUMN DETECTION
# ==============================

def detect_station_columns(df):
    """
    Detect start/end station columns using:
    - known names
    - fuzzy matching
    - station value scanning
    """

    if df is None or df.empty:
        return {
            "success": False,
            "error": "Empty dataframe."
        }

    normalized_columns = {col: normalize_header(col) for col in df.columns}

    start_candidates = []
    end_candidates = []

    for col, norm in normalized_columns.items():
        if any(x in norm for x in ["startstation", "fromstation", "beginstation"]):
            start_candidates.append(col)

        if any(x in norm for x in ["endstation", "tostation"]):
            end_candidates.append(col)

    # Fallback: scan values for station-like content
    if not start_candidates or not end_candidates:
        station_like_cols = []

        for col in df.columns:
            values = df[col].dropna().astype(str).head(30)
            station_matches = sum(is_station_format(v) for v in values)

            if station_matches >= 2:
                station_like_cols.append(col)

        if len(station_like_cols) >= 2:
            if not start_candidates:
                start_candidates.append(station_like_cols[0])
            if not end_candidates:
                end_candidates.append(station_like_cols[1])

    if not start_candidates or not end_candidates:
        return {
            "success": False,
            "error": "Could not confidently detect station columns."
        }

    return {
        "success": True,
        "start_col": start_candidates[0],
        "end_col": end_candidates[0]
    }


# ==============================
# EXTRACT TABULAR STATION RANGES
# ==============================

def extract_tabular_station_ranges(df):
    """
    Standard tabular extraction:
    start station column + end station column
    """
    detection = detect_station_columns(df)

    if not detection["success"]:
        return {
            "success": False,
            "error": detection["error"]
        }

    start_col = detection["start_col"]
    end_col = detection["end_col"]

    rows = []

    for _, row in df.iterrows():
        start_raw = row[start_col]
        end_raw = row[end_col]

        if not is_station_format(start_raw) or not is_station_format(end_raw):
            continue

        start_num = station_to_number(start_raw)
        end_num = station_to_number(end_raw)

        if start_num is None or end_num is None:
            continue

        rows.append({
            "start_station": str(start_raw).strip(),
            "end_station": str(end_raw).strip(),
            "start_dist": start_num,
            "end_dist": end_num,
            "source_mode": "tabular"
        })

    if len(rows) == 0:
        return {
            "success": False,
            "error": "No valid tabular station rows found."
        }

    return {
        "success": True,
        "mode": "tabular",
        "data": rows,
        "start_col": start_col,
        "end_col": end_col
    }


# ==============================
# MAIN EXTRACTION ENTRY
# ==============================

def extract_station_ranges(df):
    """
    Try:
    1. form-style bore log extraction
    2. tabular station-column extraction

    Return safe structured result.
    """
    df = clean_dataframe(df)

    if df is None or df.empty:
        return {
            "success": False,
            "error": "No usable data found in sheet."
        }

    # --------------------------------
    # Try form-style detection first
    # --------------------------------
    form_result = detect_form_station_pair(df)
    if form_result["success"]:
        start_station = form_result["start_station"]
        end_station = form_result["end_station"]

        start_num = station_to_number(start_station)
        end_num = station_to_number(end_station)

        if start_num is not None and end_num is not None:
            return {
                "success": True,
                "mode": "form",
                "start_col": "form_label_scan",
                "end_col": "form_label_scan",
                "data": [
                    {
                        "start_station": start_station,
                        "end_station": end_station,
                        "start_dist": start_num,
                        "end_dist": end_num,
                        "source_mode": "form"
                    }
                ]
            }

    # --------------------------------
    # Fallback to normal tabular mode
    # --------------------------------
    tabular_result = extract_tabular_station_ranges(df)
    if tabular_result["success"]:
        return tabular_result

    return {
        "success": False,
        "error": "Could not detect valid station data in form-style or tabular layout."
    }