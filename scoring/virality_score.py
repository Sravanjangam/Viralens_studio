from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


@dataclass
class ViralityBreakdown:
    aesthetic: float
    geometry: float
    color_light: float
    caption: float
    trend: float
    final_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aesthetic": self.aesthetic,
            "geometry": self.geometry,
            "color_light": self.color_light,
            "caption": self.caption,
            "trend": self.trend,
            "final_score": self.final_score,
        }


class ViralityScorer:
    def __init__(self):
        pass

    def compute(
        self,
        aesthetic_score: float,
        geometry_scores: Dict[str, float],
        color_scores: Dict[str, float],
        caption_score: float,
        trend_similarity: float,
    ) -> ViralityBreakdown:
        """
        Inputs should already be normalized to [0,1].
        """

        # Geometry score (5 metrics combined)
        rule = geometry_scores.get("rule_of_thirds", 0)
        sym = geometry_scores.get("symmetry", 0)
        clutter = geometry_scores.get("clutter", 0)
        brightness = color_scores.get("brightness", 0)
        contrast = color_scores.get("contrast", 0)

        geometry_score = (
            0.40 * rule +
            0.20 * sym +
            0.20 * (1 - clutter) +   # lower clutter is better
            0.10 * brightness +
            0.10 * contrast
        )

        # Color-light score (brightness & contrast summarized)
        color_light_score = (
            0.5 * brightness +
            0.5 * contrast
        )

        # Visual score (overall image quality)
        visual_score = (
            0.40 * aesthetic_score +
            0.30 * geometry_score +
            0.30 * color_light_score
        )

        # Combine all dimensions
        final_score = (
            0.60 * visual_score +
            0.25 * caption_score +
            0.15 * trend_similarity
        ) * 100

        return ViralityBreakdown(
            aesthetic=float(aesthetic_score),
            geometry=float(geometry_score),
            color_light=float(color_light_score),
            caption=float(caption_score),
            trend=float(trend_similarity),
            final_score=float(final_score),
        )
