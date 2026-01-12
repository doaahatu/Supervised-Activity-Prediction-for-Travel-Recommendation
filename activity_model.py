# activity_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


TEXT_COL = "description"
LABEL_COL = "activity"


@dataclass
class ActivityModel:
    """Supervised activity classifier (this is the evaluated ML task)."""
    pipeline: Optional[Pipeline] = None
    labels_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "ActivityModel":
        df = df.copy()

        # Basic cleaning
        df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("").str.strip()
        df[LABEL_COL] = df[LABEL_COL].astype(str).fillna("").str.strip().str.lower()

        # Drop empty labels/text
        df = df[(df[TEXT_COL].str.len() > 0) & (df[LABEL_COL].str.len() > 0)].reset_index(drop=True)

        # Train model (best you already found: LR C=10)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=6000,
                stop_words="english"
            )),
            ("clf", LogisticRegression(
                C=10.0,
                max_iter=2000,
                n_jobs=None,
                class_weight=None,
                solver="liblinear"
            ))
        ])

        self.pipeline.fit(df[TEXT_COL], df[LABEL_COL])
        self.labels_ = sorted(df[LABEL_COL].unique().tolist())
        return self

    def predict(self, text: str) -> str:
        if not self.pipeline:
            raise RuntimeError("ActivityModel is not fitted yet.")
        text = (text or "").strip()
        if not text:
            return "unknown"
        return str(self.pipeline.predict([text])[0])

    def predict_proba_topk(self, text: str, k: int = 3):
        """Return top-k labels with probabilities (useful for UI)."""
        if not self.pipeline:
            raise RuntimeError("ActivityModel is not fitted yet.")
        vec = self.pipeline.named_steps["tfidf"].transform([text])
        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "predict_proba"):
            return []
        probs = clf.predict_proba(vec)[0]
        classes = clf.classes_
        pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:k]
        return [(str(a), float(p)) for a, p in pairs]
