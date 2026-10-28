"""Evaluate recommendation relevance using collected user ratings.

Ratings of 4-5 are treated as relevant; ratings 1-2 as not relevant.
Ratings of 3 are neutral and excluded from binary relevance metrics.
This is feedback-based evaluation, not a substitute for a representative user study.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_RATINGS = ROOT / "data" / "user_ratings.csv"

def precision_at_k(relevance, k):
    rel = np.asarray(relevance[:k], dtype=int)
    return float(rel.sum() / k) if k else 0.0

def recall_at_k(relevance, total_relevant, k):
    if total_relevant == 0:
        return np.nan
    return float(np.asarray(relevance[:k], dtype=int).sum() / total_relevant)

def ndcg_at_k(grades, k):
    grades = np.asarray(grades[:k], dtype=float)
    if len(grades) == 0:
        return np.nan
    discounts = 1 / np.log2(np.arange(2, len(grades) + 2))
    dcg = float(np.sum((2 ** grades - 1) * discounts))
    ideal = np.sort(grades)[::-1]
    idcg = float(np.sum((2 ** ideal - 1) * discounts))
    return dcg / idcg if idcg > 0 else np.nan

def evaluate(path=DEFAULT_RATINGS, k=5):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No ratings file found at {path}. Run app.py, rate recommendations, then retry."
        )
    ratings = pd.read_csv(path)
    required = {"home_name", "rating"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"Ratings file must contain columns: {sorted(required)}")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating", "home_name"])
    if ratings.empty:
        raise ValueError("No valid ratings found.")

    # A simple grouped evaluation by user query; query represents the preference prompt.
    if "query" not in ratings.columns:
        ratings["query"] = ""
    metrics = []
    for query, group in ratings.groupby("query", dropna=False):
        group = group.drop_duplicates(subset=["home_name"], keep="last")
        if "recommendation_rank" in group and group["recommendation_rank"].notna().any():
            group = group.sort_values("recommendation_rank", ascending=True, na_position="last")
        elif "timestamp_utc" in group:
            group = group.sort_values("timestamp_utc")
        grades_all = group["rating"].astype(float).to_numpy()
        # Rating 3 is neutral/uncertain. Exclude it from binary relevance judgments.
        judged_mask = grades_all != 3
        judged_grades = grades_all[judged_mask]
        judged_binary = (judged_grades >= 4).astype(int)
        total_relevant = int(judged_binary.sum())
        metrics.append({
            "query": str(query),
            "judged_ratings": int(judged_mask.sum()),
            "mean_rating": float(np.mean(grades_all)) if len(grades_all) else np.nan,
            "relevant_share_rating_4_5": float(np.mean(grades_all >= 4)) if len(grades_all) else np.nan,
            f"precision_at_{k}_among_judged": precision_at_k(judged_binary, min(k, len(judged_binary))) if len(judged_binary) else np.nan,
            f"recall_at_{k}_among_judged": recall_at_k(judged_binary, total_relevant, min(k, len(judged_binary))) if len(judged_binary) else np.nan,
            f"ndcg_at_{k}_rating_grade": ndcg_at_k(judged_grades, k),
        })
    report = pd.DataFrame(metrics)
    print("Ratings collected:", len(ratings))
    print("Unique homes rated:", ratings["home_name"].nunique())
    print("Unique preference queries:", ratings["query"].nunique(dropna=False))
    print(report.to_string(index=False))
    output = ROOT / "artifacts" / "evaluation_report.csv"
    output.parent.mkdir(exist_ok=True)
    report.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved report to {output}")
    print(
        "\\nCaution: these metrics use only rated homes, so unrated homes are unknown rather than irrelevant. "
        "Treat the results as preliminary feedback indicators, not a rigorous offline benchmark."
    )
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default=str(DEFAULT_RATINGS))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    evaluate(args.ratings, args.k)
