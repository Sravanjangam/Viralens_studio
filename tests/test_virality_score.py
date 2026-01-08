from scoring.virality_score import ViralityScorer

scorer = ViralityScorer()

result = scorer.compute(
    aesthetic_score=0.70,
    geometry_scores={
        "rule_of_thirds": 0.6,
        "symmetry": 0.4,
        "clutter": 0.2,
    },
    color_scores={
        "brightness": 0.5,
        "contrast": 0.6,
    },
    caption_score=0.85,
    trend_similarity=0.72,
)

print("Virality breakdown:")
print(result.to_dict())
