import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


# =========================
# CONFIG 
# =========================
DATA_PATH = "data.csv"         
OUTPUT_DIR = "outputs"
TEXT_COL = "description"
LABEL_COL = "activity"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Hyperparameter tuning (4 values as required)
SVM_C_VALUES = [0.01, 0.1, 1.0, 10.0]
LR_C_VALUES  = [0.01, 0.1, 1.0, 10.0]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(s: str) -> str:
    """Light text cleaning (safe even if data already cleaned)."""
    s = "" if pd.isna(s) else str(s).lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    return df


def preprocess_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal checks + drops NaNs. Keeps your preprocessing intact."""
    needed = [TEXT_COL, LABEL_COL]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'. Found columns: {list(df.columns)}")

    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(clean_text)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()

    df = df[df[TEXT_COL].str.len() > 0]
    df = df[df[LABEL_COL].str.len() > 0]

    # Optional stability: remove labels with <3 samples (helps stratify + evaluation)
    counts = df[LABEL_COL].value_counts()
    keep = counts[counts >= 3].index
    df = df[df[LABEL_COL].isin(keep)].reset_index(drop=True)

    return df


def run_eda(df: pd.DataFrame):
    summary = {
        "num_rows": int(len(df)),
        "num_classes": int(df[LABEL_COL].nunique()),
        "top_classes": df[LABEL_COL].value_counts().head(10).to_dict(),
        "avg_text_len": float(df[TEXT_COL].str.len().mean()),
        "median_text_len": float(df[TEXT_COL].str.len().median()),
    }
    with open(os.path.join(OUTPUT_DIR, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    vc = df[LABEL_COL].value_counts()
    plt.figure()
    vc.head(15).plot(kind="bar")
    plt.title("Top 15 Activities (count)")
    plt.xlabel("Activity")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_activity_distribution.png"))
    plt.close()

    plt.figure()
    df[TEXT_COL].str.len().plot(kind="hist", bins=30)
    plt.title("Description Length Distribution")
    plt.xlabel("Length (characters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_text_length_hist.png"))
    plt.close()


def evaluate(name: str, pipe: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")

    # Save report
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(
        os.path.join(OUTPUT_DIR, f"{name}_classification_report.csv"),
        index=True
    )

    # Save confusion matrix (image)
    labels = sorted(pd.unique(pd.concat([y_test, pd.Series(y_pred)])))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_confusion_matrix.png"))
    plt.close()

    return {"model": name, "accuracy": float(acc), "macro_f1": float(f1m)}


def save_misclassified(best_pipe: Pipeline, X_test, y_test, out_path: str, max_rows=80):
    y_pred = best_pipe.predict(X_test)
    wrong = np.where(y_pred != y_test)[0]
    rows = []
    for i in wrong[:max_rows]:
        rows.append({
            "description": X_test.iloc[i],
            "true_activity": y_test.iloc[i],
            "pred_activity": y_pred[i],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    print("Loading data...")
    raw = load_data(DATA_PATH)

    print("Minimal preprocessing...")
    df = preprocess_minimal(raw)
    df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_dataset.csv"), index=False)

    print("EDA...")
    run_eda(df)

    X = df[TEXT_COL]
    y = df[LABEL_COL]

    print("Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    results = []

    # =========================
    # 3) BASELINE: kNN (k=1 and k=3) with cosine distance
    # =========================
    print("Baseline kNN...")
    for k in [1, 3]:
        knn_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("knn", KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")),
        ])
        results.append(evaluate(f"knn_k={k}", knn_pipe, X_train, X_test, y_train, y_test))

    # =========================
    # 4) MODEL 1: Linear SVM + tuning C (4 values)
    # =========================
    print("Model 1: Linear SVM tuning...")
    best_svm_pipe, best_svm_f1, best_svm_name = None, -1, None
    for C in SVM_C_VALUES:
        svm_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("svm", LinearSVC(C=C)),
        ])
        name = f"linear_svm_C={C}"
        res = evaluate(name, svm_pipe, X_train, X_test, y_train, y_test)
        results.append(res)
        if res["macro_f1"] > best_svm_f1:
            best_svm_f1 = res["macro_f1"]
            best_svm_pipe = svm_pipe
            best_svm_name = name

    # =========================
    # 4) MODEL 2: Logistic Regression + tuning C (4 values)
    # =========================
    print("Model 2: Logistic Regression tuning...")
    best_lr_pipe, best_lr_f1, best_lr_name = None, -1, None
    for C in LR_C_VALUES:
        lr_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("lr", LogisticRegression(C=C, max_iter=4000)),
        ])
        name = f"logreg_C={C}"
        res = evaluate(name, lr_pipe, X_train, X_test, y_train, y_test)
        results.append(res)
        if res["macro_f1"] > best_lr_f1:
            best_lr_f1 = res["macro_f1"]
            best_lr_pipe = lr_pipe
            best_lr_name = name

    # Save results table
    results_df = pd.DataFrame(results).sort_values(by=["macro_f1", "accuracy"], ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results_table.csv"), index=False)
    print("\n=== RESULTS (sorted by Macro-F1) ===")
    print(results_df)

    # Pick best model overall
    best_model_row = results_df.iloc[0]
    best_name = best_model_row["model"]
    print(f"\nBest model selected: {best_name}")

    # Rebuild best pipeline reference
    if best_name.startswith("linear_svm"):
        best_pipe = best_svm_pipe
    elif best_name.startswith("logreg"):
        best_pipe = best_lr_pipe
    else:
        # best is knn baseline
        k = 1 if "knn_k=1" in best_name else 3
        best_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("knn", KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")),
        ])

    best_pipe.fit(X_train, y_train)

    # Save best model for Streamlit app
    joblib.dump(best_pipe, os.path.join(OUTPUT_DIR, "best_activity_model.joblib"))

    # Error analysis outputs
    save_misclassified(
        best_pipe,
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
        os.path.join(OUTPUT_DIR, "misclassified_examples.csv"),
        max_rows=80
    )

    # Per-class F1 (patterns)
    y_pred = best_pipe.predict(X_test)
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    per_class = (
        pd.DataFrame(rep).transpose()
        .drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
        .sort_values("f1-score")
    )
    per_class.to_csv(os.path.join(OUTPUT_DIR, "per_class_f1_sorted.csv"))

    print("\nSaved outputs in ./outputs")
    print("- outputs/results_table.csv")
    print("- outputs/best_activity_model.joblib")
    print("- outputs/misclassified_examples.csv")
    print("- outputs/per_class_f1_sorted.csv")
    print("\nDONE ✅")


if __name__ == "__main__":
    main()
