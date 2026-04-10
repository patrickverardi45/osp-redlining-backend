from fastapi import APIRouter, UploadFile, File
import tempfile, os, json
from app.core.kmz_extractor import extract_routes_from_kmz
from app.core.route_matching_engine import match_bore_to_routes

router = APIRouter()

@router.post("/debug/auto-match")
async def auto_match(file: UploadFile = File(...), bore_log_json: str = None):
    '''
    Upload KMZ + pass bore_log_json as raw JSON string
    '''

    # save KMZ
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    routes = extract_routes_from_kmz(path)
    os.remove(path)

    # filter weak routes
    routes = [r for r in routes if len(r["points"]) >= 3 and r["end_station"] > 50]

    if not bore_log_json:
        return {"error": "missing bore_log_json"}

    bore_log = json.loads(bore_log_json)

    result = match_bore_to_routes(bore_log, routes)

    return {
        "filtered_routes": len(routes),
        "match_result": result
    }
