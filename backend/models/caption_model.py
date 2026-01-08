from pydantic import BaseModel


class CaptionRequest(BaseModel):
    caption: str


class CaptionResponse(BaseModel):
    sentiment_score: float
    hook_score: float
    cta_score: float
    length_score: float
    emoji_score: float
    overall_caption_score: float
