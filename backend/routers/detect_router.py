from fastapi import APIRouter, UploadFile, File
import shutil
import uuid
from cv_engine.detector import ObjectDetector
from backend.models.detect_model import DetectResponse


router = APIRouter()
detector = ObjectDetector()  # load YOLO once


@router.post("/", response_model=DetectResponse)
async def detect_image(file: UploadFile = File(...)):
    # save temporary file
    ext = file.filename.split(".")[-1]
    temp_path = f"temp_{uuid.uuid4()}.{ext}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # run detector
    result = detector.load(temp_path)

    return DetectResponse(**result)
