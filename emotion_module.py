# emotion_module.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import re

# Simple keyword-based emotion intent from user text
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "adventure": ["adventure", "thrill", "hike", "hiking", "explore", "exploring", "climb", "climbing", "extreme"],
    "excitement": ["exciting", "party", "nightlife", "festival", "fun", "crowd", "music", "energy"],
    "happiness": ["happy", "joy", "sun", "sunny", "bright", "smile", "cheerful", "vibrant"],
    "curiosity": ["curious", "discover", "discovering", "museum", "culture", "history", "heritage", "local"],
    "nostalgia": ["nostalgia", "nostalgic", "old town", "memories", "traditional", "heritage"],
    "romance": ["romance", "romantic", "honeymoon", "couple", "date", "sunset", "candle"],
    "melancholy": ["quiet", "calm", "rain", "rainy", "fog", "mist", "winter", "moody"],
}

def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]

@dataclass
class EmotionScorer:
    """
    Produces:
    - user_emotion_scores: based on user query keywords
    - destination_emotion_boost: based on destination mood field match
    """
    def score_user_text(self, user_text: str) -> Dict[str, float]:
        tokens = set(_tokenize(user_text))
        scores: Dict[str, float] = {}
        for emo, kws in EMOTION_KEYWORDS.items():
            hit = sum(1 for kw in kws if kw in tokens or kw in (user_text or "").lower())
            # normalize lightly
            scores[emo] = min(1.0, hit / 4.0)
        return scores

    def best_user_emotion(self, user_text: str) -> str:
        scores = self.score_user_text(user_text)
        best = max(scores.items(), key=lambda x: x[1])
        return best[0] if best[1] > 0 else "neutral"

    def destination_boost(self, destination_mood: str, user_emotion: str) -> float:
        """
        destination_mood: value from dataset column 'mood'
        user_emotion: detected from user text
        """
        dm = (destination_mood or "").strip().lower()
        ue = (user_emotion or "").strip().lower()
        if not dm or ue == "neutral":
            return 0.0
        # exact match gets bigger boost; partial match smaller
        if dm == ue:
            return 0.25
        if ue in dm or dm in ue:
            return 0.10
        return 0.0
