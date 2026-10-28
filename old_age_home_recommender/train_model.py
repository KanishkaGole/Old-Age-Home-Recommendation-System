"""Train and save the content-based old-age-home recommendation model."""
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = [
    ROOT / "data" / "All_India_Old_Age_Homes.xlsx",
    ROOT.parent / "All_India_Old_Age_Homes.xlsx",
]
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = ARTIFACT_DIR / "homes_cleaned.csv"
MODEL_PATH = ARTIFACT_DIR / "content_model.joblib"

def normalize_text(value):
    if pd.isna(value):
        return ""
    value = re.sub(r"\s+", " ", str(value)).strip()
    if value.casefold() in {"", "na", "n/a", "nan", "none", "null", "-", "--"}:
        return ""
    return value

def train(input_path=None):
    if input_path:
        source = Path(input_path)
    else:
        source = next((p for p in DEFAULT_INPUTS if p.exists()), None)
    if source is None or not source.exists():
        raise FileNotFoundError(
            "Dataset not found. Put All_India_Old_Age_Homes.xlsx in data/ "
            "or pass its path to train_model.py."
        )

    df = pd.read_excel(source, sheet_name="Master_Data")
    df.columns = [str(c).strip() for c in df.columns]
    required = ["Home Name", "State", "District"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    text_cols = [
        "Home Name", "State", "District", "City / Taluka",
        "P.O./ Locality / Village", "Address", "Facility Type",
        "Mgmt", "Gender", "Care Type", "Remarks"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].map(normalize_text).replace("", np.nan)

    if "Bed Capacity" in df.columns:
        df["Bed Capacity"] = pd.to_numeric(df["Bed Capacity"], errors="coerce")
        df.loc[(df["Bed Capacity"] < 0) | (df["Bed Capacity"] % 1 != 0), "Bed Capacity"] = np.nan

    # Keep source columns and only remove exact duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)
    for col in ["Telephone", "Email", "Address", "Source"]:
        if col not in df:
            df[col] = np.nan

    def make_text(row):
        parts = []
        for col in text_cols:
            val = row.get(col, "")
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        return " ".join(parts)

    df["recommendation_text"] = df.apply(make_text, axis=1)
    df["has_phone"] = df["Telephone"].notna() & df["Telephone"].astype(str).str.strip().ne("")
    df["has_email"] = df["Email"].notna() & df["Email"].astype(str).str.strip().ne("")
    df["has_address"] = df["Address"].notna() & df["Address"].astype(str).str.strip().ne("")
    df["capacity_needs_verification"] = df["Bed Capacity"].isna() | (df["Bed Capacity"] <= 0)

    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), max_features=8000,
        stop_words="english", sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(df["recommendation_text"].fillna("").astype(str))

    # Normalize categories for reliable matching without altering display labels.
    for col in ["State", "District", "Facility Type", "Care Type", "Gender", "Mgmt"]:
        if col not in df:
            df[col] = np.nan
        df[f"{col}_normalized"] = df[col].map(normalize_text).str.casefold()

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    bundle = {
        "data": df,
        "vectorizer": vectorizer,
        "tfidf_matrix": matrix,
        "source_file": str(source),
        "model_type": "TF-IDF content-based retrieval",
        "version": 1,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Loaded source: {source}")
    print(f"Rows retained: {len(df)}")
    print(f"TF-IDF matrix: {matrix.shape}")
    print(f"Saved cleaned data: {OUTPUT_CSV}")
    print(f"Saved model bundle: {MODEL_PATH}")
    return bundle

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to All_India_Old_Age_Homes.xlsx")
    args = parser.parse_args()
    train(args.input)
