# recommender.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from activity_model import ActivityModel
from emotion_module import EmotionScorer
from scene_module import SceneDetector


TEXT_COL = "description"
COUNTRY_COL = "country"
ACTIVITY_COL = "activity"
MOOD_COL = "mood"
IMG_COL = "image_url"


@dataclass
class UnifiedRecommender:
    activity_model: ActivityModel
    emotion_scorer: EmotionScorer
    scene_detector: Optional[SceneDetector] = None

    # TF-IDF for retrieval / ranking (separate from classifier)
    retriever_vectorizer: Optional[TfidfVectorizer] = None
    retriever_matrix: Optional[Any] = None
    df_: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> "UnifiedRecommender":
        df = df.copy()

        # Basic cleaning
        for col in [TEXT_COL, COUNTRY_COL, ACTIVITY_COL, MOOD_COL, IMG_COL]:
            if col not in df.columns:
                df[col] = ""

        df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("").str.strip()
        df[COUNTRY_COL] = df[COUNTRY_COL].astype(str).fillna("").str.strip()
        df[ACTIVITY_COL] = df[ACTIVITY_COL].astype(str).fillna("").str.strip().str.lower()
        df[MOOD_COL] = df[MOOD_COL].astype(str).fillna("").str.strip().str.lower()
        df[IMG_COL] = df[IMG_COL].astype(str).fillna("").str.strip()

        df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)

        # Fit activity model (evaluated)
        self.activity_model.fit(df)

        # Fit retriever TF-IDF (for similarity ranking)
        self.retriever_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=8000,
            stop_words="english"
        )
        self.retriever_matrix = self.retriever_vectorizer.fit_transform(df[TEXT_COL])
        self.df_ = df
        return self

    def recommend(
        self,
        user_text: str,
        preferred_country: str = "Any",
        top_n: int = 8,
        use_scene: bool = False
    ) -> Dict[str, Any]:
        if self.df_ is None or self.retriever_vectorizer is None or self.retriever_matrix is None:
            raise RuntimeError("UnifiedRecommender is not fitted.")

        user_text = (user_text or "").strip()
        if not user_text:
            return {
                "activity": "unknown",
                "emotion": "neutral",
                "scene": [],
                "results": [],
                "note": "Please enter some text about what you like."
            }

        # 1) Predict activity (official evaluated model)
        pred_activity = self.activity_model.predict(user_text)

        # 2) Emotion from user text (aux)
        user_emotion = self.emotion_scorer.best_user_emotion(user_text)

        # 3) Candidate filtering
        df = self.df_.copy()

        # Country filter
        pc = (preferred_country or "Any").strip()
        if pc.lower() != "any" and pc != "":
            df = df[df[COUNTRY_COL].str.lower() == pc.lower()].copy()

        # Activity filter (strong)
        if pred_activity != "unknown":
            df = df[df[ACTIVITY_COL] == pred_activity].copy()

        # If no results in that strict filter, fallback: loosen to all countries/activity
        if df.empty:
            df = self.df_.copy()

        # 4) Similarity ranking
        user_vec = self.retriever_vectorizer.transform([user_text])
        sims = cosine_similarity(user_vec, self.retriever_matrix).ravel()

        # 5) Build ranking score with emotion boost
        # emotion boost based on destination mood match
        mood_boost = df[MOOD_COL].apply(lambda m: self.emotion_scorer.destination_boost(m, user_emotion)).to_numpy()

        # map sims to df indices (since df might be filtered)
        idxs = df.index.to_numpy()
        sim_filtered = sims[idxs]

        # final score: mostly similarity + emotion boost
        final_score = (0.80 * sim_filtered) + mood_boost

        df = df.assign(_score=final_score)

        # 6) Remove duplicates (same description/country/image repeating)
        df["_dedup_key"] = (
            df[COUNTRY_COL].str.lower().fillna("")
            + "||" + df[TEXT_COL].str.lower().str.slice(0, 80).fillna("")
            + "||" + df[IMG_COL].str.lower().fillna("")
        )
        df = df.drop_duplicates(subset=["_dedup_key"], keep="first")

        # 7) Sort and take top_n
        df_top = df.sort_values("_score", ascending=False).head(top_n).copy()

        # 8) Optional Scene detection
        scene_tags_map: Dict[int, List[str]] = {}
        if use_scene and self.scene_detector is not None:
            for ridx, row in df_top.iterrows():
                tags = self.scene_detector.detect(row.get(IMG_COL, ""), topk=5)
                scene_tags_map[int(ridx)] = tags

        # 9) Prepare results
        results: List[Dict[str, Any]] = []
        for ridx, row in df_top.iterrows():
            tags = scene_tags_map.get(int(ridx), [])
            reason_parts = []
            if pred_activity != "unknown":
                reason_parts.append(f"matches your predicted activity: {pred_activity}")
            if user_emotion != "neutral":
                if row.get(MOOD_COL, "") == user_emotion:
                    reason_parts.append(f"matches your mood: {user_emotion}")
                else:
                    reason_parts.append(f"similar to your mood: {user_emotion}")
            if tags:
                reason_parts.append(f"scene: {', '.join(tags)}")

            results.append({
                "country": row.get(COUNTRY_COL, ""),
                "activity": row.get(ACTIVITY_COL, ""),
                "mood": row.get(MOOD_COL, ""),
                "description": row.get(TEXT_COL, ""),
                "image_url": row.get(IMG_COL, ""),
                "score": float(row.get("_score", 0.0)),
                "scene_tags": tags,
                "reason": "; ".join(reason_parts) if reason_parts else "high text similarity",
            })

        return {
            "activity": pred_activity,
            "emotion": user_emotion,
            "scene": [],
            "results": results
        }
