from pydantic import BaseModel
from typing import Dict, List


class FullAnalysisResponse(BaseModel):
    # Raw modules
    objects: List[Dict] | None
    geometry: Dict[str, float]
    color: Dict[str, float]
    aesthetic: float
    caption: Dict[str, float]
    trend_similarity: float

    # Final
    virality_score: float
