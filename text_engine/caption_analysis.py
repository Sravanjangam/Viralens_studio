from __future__ import annotations
import re
import math
from dataclasses import dataclass
from typing import Dict, Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


# ---------- Utilities ----------

EMOJI_PATTERN = re.compile(
    "["                     # broad Unicode emoji ranges
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+"
)


HOOK_PHRASES = [
    "did you know", "stop scrolling", "you won't believe", "you wont believe",
    "this is crazy", "this changed my life", "no one talks about", "the secret to",
    "here's why", "heres why", "what nobody tells you", "things i wish i knew",
]

CTA_PHRASES = [
    "link in bio", "check the link", "click the link", "buy now", "shop now",
    "use code", "tap to", "swipe up", "follow for more", "share this", "save this",
    "comment below", "tag a friend",
]


@dataclass
class CaptionAnalysisResult:
    sentiment_score: float     # [0,1]
    hook_score: float          # 0 or 1
    cta_score: float           # 0 or 1
    length_score: float        # [0,1]
    emoji_score: float         # [0,1]
    overall_caption_score: float  # [0,1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentiment_score": self.sentiment_score,
            "hook_score": self.hook_score,
            "cta_score": self.cta_score,
            "length_score": self.length_score,
            "emoji_score": self.emoji_score,
            "overall_caption_score": self.overall_caption_score,
        }


class CaptionAnalyzer:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME).to(self.device)
        self.model.eval()

    # ----- public -----

    def analyze(self, caption: str) -> CaptionAnalysisResult:
        caption = (caption or "").strip()
        if not caption:
            # handle empty caption gracefully
            return CaptionAnalysisResult(
                sentiment_score=0.5,
                hook_score=0.0,
                cta_score=0.0,
                length_score=0.0,
                emoji_score=0.0,
                overall_caption_score=0.0,
            )

        sentiment = self._sentiment_score(caption)
        hook = self._hook_score(caption)
        cta = self._cta_score(caption)
        length = self._length_score(caption)
        emojis = self._emoji_score(caption)

        overall = (
            0.40 * sentiment +
            0.25 * hook +
            0.15 * cta +
            0.10 * length +
            0.10 * emojis
        )

        return CaptionAnalysisResult(
            sentiment_score=sentiment,
            hook_score=hook,
            cta_score=cta,
            length_score=length,
            emoji_score=emojis,
            overall_caption_score=float(overall),
        )

    # ----- internals -----

    def _sentiment_score(self, caption: str) -> float:
        """Maps sentiment to [0,1], where 1 ~ very positive, 0 ~ very negative."""
        inputs = self.tokenizer(caption, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]

        probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
        # cardiffnlp ordering: [negative, neutral, positive]
        negative, neutral, positive = probs
        score = 0.2 * negative + 0.5 * neutral + 1.0 * positive
        # normalize to [0,1] (max possible is 1.0, min ~0.2)
        return float(max(0.0, min(score, 1.0)))

    def _hook_score(self, caption: str) -> float:
        text = caption.lower().strip()
        # look mostly at the first ~8 words
        first_part = " ".join(text.split()[:8])
        for phrase in HOOK_PHRASES:
            if phrase in first_part:
                return 1.0
        return 0.0

    def _cta_score(self, caption: str) -> float:
        text = caption.lower()
        for phrase in CTA_PHRASES:
            if phrase in text:
                return 1.0
        return 0.0

    def _length_score(self, caption: str) -> float:
        words = caption.split()
        n = len(words)

        # heuristic sweet spot: 8–30 words
        if n == 0:
            return 0.0
        if 8 <= n <= 30:
            return 1.0
        # linearly decay outside sweet range
        if n < 8:
            return n / 8.0
        # n > 30
        # 30–60 → 1 → 0, >60 → clamp at 0
        if n >= 60:
            return 0.0
        # between 30 and 60
        return float(max(0.0, 1 - (n - 30) / 30.0))

    def _emoji_score(self, caption: str) -> float:
        emojis = EMOJI_PATTERN.findall(caption)
        if not emojis:
            return 0.0
        emoji_count = sum(len(e) for e in emojis)
        # saturation at ~10 emojis
        return float(max(0.0, min(1.0, emoji_count / 10.0)))
