from fastapi import APIRouter
from backend.models.virality_model import ViralityRequest, ViralityResponse
from scoring.virality_score import ViralityScorer

router = APIRouter()
scorer = ViralityScorer()


@router.post("/", response_model=ViralityResponse)
async def calculate_virality(data: ViralityRequest):
    result = scorer.compute(
        aesthetic_score=data.aesthetic_score,
        geometry_scores=data.geometry_scores,
        color_scores=data.color_scores,
        caption_score=data.caption_score,
        trend_similarity=data.trend_similarity
    )

    return ViralityResponse(**result.to_dict())

