from pydantic import BaseModel
from typing import List, Optional


class DetectResponse(BaseModel):
    boxes: List[List[float]]
    classes: List[float]
    confidences: List[float]
    main_box: Optional[List[float]]
    image_size: List[int]
