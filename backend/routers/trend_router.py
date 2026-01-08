from fastapi import APIRouter, UploadFile, File
import uuid
import shutil

from backend.models.trend_model import TrendResponse
from cv_engine.trend_similarity import TrendSimilarity

router = APIRouter()
ts = TrendSimilarity()   # CLIP model loaded once


@router.post("/", response_model=TrendResponse)
async def trend_similarity(file: UploadFile = File(...)):
    # save temporary file
    ext = file.filename.split(".")[-1]
    temp_path = f"temp_{uuid.uuid4()}.{ext}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # compute similarity
    score = ts.similarity_score(temp_path)

    return TrendResponse(trend_similarity=score)

