from fastapi import APIRouter, UploadFile, File
import tempfile, os
from app.core.kmz_extractor import extract_routes_from_kmz

router = APIRouter()

@router.post("/debug/kmz-routes")
async def debug_kmz_routes(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    routes = extract_routes_from_kmz(path)
    os.remove(path)

    return {"routes":[
        {"id":r["id"],"num_points":len(r["points"]),"estimated_length":r["end_station"]}
        for r in routes
    ]}
