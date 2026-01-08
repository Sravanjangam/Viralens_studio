from pydantic import BaseModel
from typing import Dict


class ViralityRequest(BaseModel):
    aesthetic_score: float
    geometry_scores: Dict[str, float]
    color_scores: Dict[str, float]
    caption_score: float
    trend_similarity: float


class ViralityResponse(BaseModel):
    aesthetic: float
    geometry: float
    color_light: float
    caption: float
    trend: float
    final_score: float
