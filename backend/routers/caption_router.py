from fastapi import APIRouter
from backend.models.caption_model import CaptionRequest, CaptionResponse
from text_engine.caption_analysis import CaptionAnalyzer

router = APIRouter()
analyzer = CaptionAnalyzer()  # load once


@router.post("/", response_model=CaptionResponse)
async def analyze_caption(data: CaptionRequest):
    result = analyzer.analyze(data.caption)
    return CaptionResponse(**result.to_dict())

