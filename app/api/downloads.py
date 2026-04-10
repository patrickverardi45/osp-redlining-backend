from fastapi import APIRouter
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(tags=["downloads"])

@router.get("/download-csv")
def download_csv():
    file_path = Path("bore_logs.csv")

    if not file_path.exists():
        return {"error": "CSV file not found"}

    return FileResponse(
        path=file_path,
        filename="bore_logs.csv",
        media_type="text/csv"
    )