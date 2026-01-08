from pydantic import BaseModel

class TrendResponse(BaseModel):
    trend_similarity: float
