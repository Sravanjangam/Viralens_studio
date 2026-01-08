import uuid
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from scoring.virality_score import ViralityScorer
from cv_engine.detector import ObjectDetector
from cv_engine.geometry import (
    rule_of_thirds_score,
    symmetry_score,
    clutter_score,
    brightness_score,
    contrast_score
)
from cv_engine.color import ColorAnalyzer
from cv_engine.trend_similarity import TrendSimilarity
from text_engine.caption_analysis import CaptionAnalyzer

router = APIRouter()

# Load engines once
detector = ObjectDetector()
color_engine = ColorAnalyzer()
trend_engine = TrendSimilarity()
caption_engine = CaptionAnalyzer()
scorer = ViralityScorer()


@router.post("/full/")
async def full_analysis(
    file: UploadFile = File(...),
    caption: str = Form(...)
):
    # --------------------------------------------
    # 1. Save file temporarily
    # --------------------------------------------
    temp_path = f"tmp_{uuid.uuid4()}.webp"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Load image with OpenCV
    img = cv2.imread(temp_path)
    if img is None:
        return {"error": "Could not read image"}

    h, w = img.shape[:2]

    # --------------------------------------------
    # 2. Object Detection
    # --------------------------------------------
    det = detector.load(temp_path)
    main_box = det["main_box"]

    # If no objects detected → fallback center
    if main_box:
        x1, y1, x2, y2 = main_box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    else:
        cx, cy = w // 2, h // 2

    # --------------------------------------------
    # 3. Geometry Metrics
    # --------------------------------------------
    rule = rule_of_thirds_score(cx, cy, w, h)
    sym = symmetry_score(img)
    clt = clutter_score(img)
    bright = brightness_score(img)
    cont = contrast_score(img)

    geometry_scores = {
        "rule_of_thirds": rule,
        "symmetry": sym,
        "clutter": clt
    }

    color_scores = {
        "brightness": bright,
        "contrast": cont
    }

    # --------------------------------------------
    # 4. Color + Lighting
    # --------------------------------------------
    dominant_colors = color_engine.extract_colors(img)

    # --------------------------------------------
    # 5. Trend Similarity
    # --------------------------------------------
    trend_score = trend_engine.similarity_score(temp_path)


    # --------------------------------------------
    # 6. Caption Analysis
    # --------------------------------------------
    cap = caption_engine.analyze(caption)
    caption_score = cap.overall_caption_score


    # --------------------------------------------
    # 7. Aesthetic score (simple version: brightness + contrast)
    # (You can replace with CLIP model later)
    # --------------------------------------------
    aesthetic_score = (bright * 0.4) + (cont * 0.6)

    # --------------------------------------------
    # 8. Virality Score (FINAL FIX)
    # --------------------------------------------
    result = scorer.compute(
        aesthetic_score=aesthetic_score,
        geometry_scores=geometry_scores,
        color_scores=color_scores,
        caption_score=caption_score,
        trend_similarity=trend_score,
    )

    # --------------------------------------------
    # 9. Return combined output
    # --------------------------------------------
    return {
        "dominant_colors": dominant_colors,
        "geometry": geometry_scores,
        "color": color_scores,
        "trend_similarity": trend_score,
        "caption_analysis": cap,
        "aesthetic_score": aesthetic_score,
        "virality": result.to_dict()
    }
