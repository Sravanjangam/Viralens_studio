import os
from typing import Optional, Dict, Any
from openai import OpenAI


class ChatEngine:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing")

        # Groq OpenAI-compatible client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        # You can change this to any Groq-supported model later
        self.model = "openai/gpt-oss-20b"

        # System prompt: high-expertise, evaluative, warm tone
        self.system_prompt = """
You are **ViraLens AI** — a senior AI strategist and visual intelligence expert.
You specialize in:
- Content virality
- Visual geometry & composition
- Aesthetics (lighting, color, contrast)
- Trend alignment
- Caption psychology & creator branding

You MUST:
1. Think analytically before answering.
2. Identify ROOT CAUSES, not just surface issues.
3. Benchmark the content against what truly goes viral.
4. Prioritize actions by IMPACT, not by convenience.
5. Be clear, confident, kind, and practical.
6. Avoid generic advice. Always be specific and grounded in the metrics.
7. Sound human, warm, and expert — never robotic.
8. Always give the user a feeling of: “I know exactly what to do next.”

Your responses should be:
- Well structured (use headings, bullet points, and tables where helpful).
- Actionable (exact changes to make in composition, style, caption, and posting strategy).
- Encouraging but honest.
"""

    def _build_analysis_context(self, metrics: Dict[str, Any]) -> str:
        """
        Convert raw metrics dict into a readable context block for the model.
        Handles both flat and nested virality structures.
        """
        if not metrics:
            return ""

        # Try to infer values from different possible shapes
        virality = metrics.get("virality", {}) if isinstance(metrics.get("virality"), dict) else {}

        final_score = (
            virality.get("final_score")
            or metrics.get("final_score")
        )

        geometry_score = (
            virality.get("geometry")
            or metrics.get("geometry")
            or metrics.get("geometry_score")
        )

        aesthetic_score = (
            virality.get("aesthetic")
            or metrics.get("aesthetic")
            or metrics.get("aesthetic_score")
        )

        trend_score = (
            virality.get("trend")
            or metrics.get("trend")
            or metrics.get("trend_similarity")
        )

        caption_score = (
            virality.get("caption")
            or metrics.get("caption")
            or metrics.get("overall_caption_score")
        )

        # Build a readable block for the model
        context = f"""
CONTENT ANALYSIS DATA (from ViraLens engine):

- Virality Score (0–100): {final_score}
- Geometry Score (0–1):  {geometry_score}
- Aesthetic Score (0–1): {aesthetic_score}
- Trend Similarity (0–1): {trend_score}
- Caption Score (0–1):   {caption_score}

INTERPRETATION:
• Geometry → framing, rule-of-thirds, balance, and visual hierarchy.
• Aesthetics → lighting, contrast, color harmony, and visual polish.
• Trend similarity → how much this matches current platform + cultural patterns.
• Caption score → hook strength, clarity, emotional pull, and CTA quality.
"""

        return context

    def chat(self, message: str, metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        High-value chat interface:
        - Explains metrics
        - Evaluates weaknesses
        - Suggests prioritized improvements
        - Speaks with expert, warm tone
        """
        analysis_context = ""
        if metrics:
            analysis_context = self._build_analysis_context(metrics)

        user_prompt = f"""
User request:
{message}

TASK FOR YOU:
1. First, diagnose what is most limiting virality (geometry, aesthetics, trend, caption, or something else).
2. Explain clearly WHY the current scores are where they are.
3. Propose the TOP 3–5 highest-impact improvements:
   - Composition / framing
   - Color & lighting
   - Style and vibe
   - Caption & hook
   - Trend and format choices
4. Where helpful, give examples of:
   - Better captions (3 options)
   - Better visual styles or shooting directions (2–3 ideas)
5. End with a short, motivational summary that feels like a coach talking to a creator.

Tone:
- Empathetic, practical, and expert.
- Avoid fluff. Every sentence should add value.
"""

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": analysis_context + "\n" + user_prompt,
                },
            ],
        )

        return response.output_text
