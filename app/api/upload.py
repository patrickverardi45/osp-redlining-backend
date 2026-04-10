import csv
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.pdf_service import process_pdf
from app.services.gis_service import process_gis
from app.services.bore_log_service import process_bore_log

router = APIRouter(tags=["uploads"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

BORE_LOG_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}
GIS_EXTENSIONS = {".kmz", ".kml"}
PDF_EXTENSIONS = {".pdf"}


def detect_file_category(filename: str) -> str:
    extension = Path(filename).suffix.lower()

    if extension in BORE_LOG_EXTENSIONS:
        return "bore_log"
    if extension in GIS_EXTENSIONS:
        return "gis_route"
    if extension in PDF_EXTENSIONS:
        return "design_pdf"

    return "unknown"


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    date: str = Form(None),
    crew: str = Form(None),
    job_name: str = Form(None),
    print_number: str = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    destination = UPLOAD_DIR / file.filename
    contents = await file.read()
    destination.write_bytes(contents)

    category = detect_file_category(file.filename)
    extension = Path(file.filename).suffix.lower()

    if category == "design_pdf":
        result = process_pdf(str(destination))
    elif category == "gis_route":
        result = process_gis(str(destination))
    elif category == "bore_log":
        result = process_bore_log(str(destination))
    else:
        result = {"status": "no processor available"}

    csv_file = Path("bore_logs.csv")
    file_exists = csv_file.exists()

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["date", "crew", "job_name", "print_number"])

        writer.writerow([date, crew, job_name, print_number])

    return {
        "message": "File uploaded and saved successfully",
        "filename": file.filename,
        "saved_to": str(destination),
        "content_type": file.content_type,
        "extension": extension,
        "category": category,
        "final_data": {
            "date": date,
            "crew": crew,
            "job_name": job_name,
            "print_number": print_number,
        },
        "csv_saved_to": str(csv_file),
        "processing_result": result,
    }