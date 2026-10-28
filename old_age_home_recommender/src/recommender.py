"""Recommendation scoring and rating utilities."""
from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DISPLAY_COLS = [
    "Home Name", "State", "District", "City / Taluka", "Address",
    "Facility Type", "Care Type", "Gender", "Mgmt", "Bed Capacity",
    "Telephone", "Email", "Source"
]

def _norm(value):
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()

def recommend(bundle, query="", state="Any", district="Any",
              facility_type="Any", care_type="Any", gender="Any",
              management="Any", top_n=10, prioritize_contact=True):
    """Apply hard filters first, then rank eligible records by content similarity.

    State/district/facility/care/gender/management selections are strict filters.
    Text query and contact completeness affect ranking only.
    """
    df = bundle["data"].copy()
    mask = pd.Series(True, index=df.index)

    filters = [
        ("State", state), ("District", district), ("Facility Type", facility_type),
        ("Care Type", care_type), ("Gender", gender), ("Mgmt", management)
    ]
    for col, selected in filters:
        if selected and selected != "Any":
            # Exact matching, case-insensitive. For gender, permit a home labelled
            # "Men & Women" to satisfy either Men or Women.
            values = df[col].map(_norm) if col in df else pd.Series("", index=df.index)
            wanted = _norm(selected)
            if col == "Gender" and wanted in {"men", "women"}:
                mask &= values.eq(wanted) | values.str.contains("men & women|men & woman", regex=True)
            else:
                mask &= values.eq(wanted)

    candidates = df.loc[mask].copy()
    if candidates.empty:
        return candidates.assign(similarity_score=pd.Series(dtype=float),
                                 contact_bonus=pd.Series(dtype=float),
                                 recommendation_score=pd.Series(dtype=float))

    if query.strip():
        q_vec = bundle["vectorizer"].transform([query])
        positions = candidates.index.to_numpy()
        similarities = cosine_similarity(q_vec, bundle["tfidf_matrix"][positions]).ravel()
    else:
        similarities = np.zeros(len(candidates), dtype=float)

    candidates["similarity_score"] = similarities
    if prioritize_contact:
        phone = candidates.get("has_phone", pd.Series(False, index=candidates.index)).fillna(False).astype(float)
        email = candidates.get("has_email", pd.Series(False, index=candidates.index)).fillna(False).astype(float)
        address = candidates.get("has_address", pd.Series(False, index=candidates.index)).fillna(False).astype(float)
        candidates["contact_bonus"] = (phone * 0.5 + email * 0.2 + address * 0.3)
    else:
        candidates["contact_bonus"] = 0.0

    # Content match remains the primary ranking signal when a query is provided.
    # When no query is provided, use contact completeness as a modest tie-breaker.
    candidates["recommendation_score"] = (
        candidates["similarity_score"] * 0.85 +
        candidates["contact_bonus"] * 0.15
    )
    candidates = candidates.sort_values(
        ["recommendation_score", "similarity_score"], ascending=False
    ).head(int(top_n)).copy()
    candidates["recommendation_rank"] = np.arange(1, len(candidates) + 1)
    return candidates

def log_rating(rating_path, home_id, home_name, rating, query="", notes="", recommendation_rank=None, recommendation_score=None):
    """Append a user rating. Use a row index as a local session identifier only."""
    rating_path = Path(rating_path)
    rating_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "home_id": int(home_id),
        "home_name": str(home_name),
        "rating": int(rating),
        "query": str(query),
        "notes": str(notes),
        "recommendation_rank": recommendation_rank,
        "recommendation_score": recommendation_score,
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }])
    row.to_csv(rating_path, mode="a", header=not rating_path.exists(),
               index=False, encoding="utf-8-sig")
