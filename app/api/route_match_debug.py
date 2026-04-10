from fastapi import APIRouter
from app.core.route_matching_engine import match_bore_to_routes

router = APIRouter()

@router.post("/debug/route-match")
def debug_route_match(payload: dict):
    return match_bore_to_routes(payload.get("bore_log"), payload.get("routes"))
