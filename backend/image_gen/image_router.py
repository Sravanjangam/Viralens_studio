import uuid
import cv2
from fastapi import APIRouter, UploadFile, File, Form

from backend.image_gen.prompt_builder import build_improvement_prompt
from backend.image_gen.pollinations_client import generate_images

router = APIRouter(prefix="/image", tags=["Image Generation"])


@router.post("/improve/")
async def improve_image(
    file: UploadFile = File(...),
    user_prompt: str = Form(None)
):
    # Save uploaded image
    temp_path = f"tmp_{uuid.uuid4()}.png"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    img = cv2.imread(temp_path)
    if img is None:
        return {"error": "Invalid image file"}

    # ✅ SAFE DEFAULT METRICS (NO external dependency)
    metrics = {
        "geometry": {
            "rule_of_thirds": 0.7,
            "symmetry": 0.6,
            "clutter": 0.3
        },
        "aesthetic_score": 0.6,
        "trend_similarity": 0.5
    }

    prompt = build_improvement_prompt(metrics, user_prompt)

    image_urls = generate_images(
        prompt=prompt,
        image_url=None,
        n=4
    )

    return {
        "prompt_used": prompt,
        "generated_images": image_urls
    }
