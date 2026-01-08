def build_improvement_prompt(metrics: dict, user_prompt: str | None = None):
    """
    Builds a high-quality image generation prompt
    using Virality metrics + optional user input
    """

    geometry = metrics.get("geometry", {})
    aesthetic = metrics.get("aesthetic_score", 0.5)
    trend = metrics.get("trend_similarity", 0.5)

    prompt = f"""
Improve this image for high social media virality.

Visual goals:
- Strong visual focus on the main subject
- Clean, modern, high-quality aesthetic
- Good contrast and sharp details
- Proper lighting (no dull or flat look)
- Rule of thirds composition
- Clear subject-background separation

Optimization hints:
- Geometry quality: {geometry}
- Aesthetic quality score: {round(aesthetic, 2)}
- Trend similarity score: {round(trend, 2)}

Style:
- Professional
- Scroll-stopping
- Platform-friendly (Instagram / TikTok)

Avoid:
- Blurry details
- Overcrowded frames
- Flat lighting
- Washed out colors
"""

    if user_prompt:
        prompt += f"\nUser preference:\n{user_prompt}\n"

    return prompt.strip()
