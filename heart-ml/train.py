"""
Train a baseline heart-disease classifier (Cleveland-style label).
- Loads any CSV in data/ (or a specific path)
- Cleans, encodes, and trains a model (RF or Logistic Regression)
- Prints metrics and saves the model to models/<name>.joblib
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def find_csv(datadir: Path) -> Path:
    """Pick a CSV from data/ (prefers common names, otherwise first CSV)."""
    preferred = [
        "heart-disease-cleveland-uci.csv",
        "heart.csv",
        "cleveland.csv",
        "heart_cleveland_upload.csv",
    ]
    for name in preferred:
        p = datadir / name
        if p.exists():
            return p
    csvs = sorted(datadir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {datadir}.")
    return csvs[0]


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, coerce '?' to NaN, and create binary target."""
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df = df.replace("?", np.nan)

    # Locate target column (common aliases)
    target_aliases = ["target", "num", "class", "condition"]
    target_col = next((c for c in target_aliases if c in df.columns), None)
    if target_col is None:
        raise ValueError(f"Could not find target column among {target_aliases}")

    # Cleveland convention: 0 = no disease; 1..4 = disease
    y = pd.to_numeric(df[target_col], errors="coerce")
    df["target_bin"] = (y > 0).astype(int)

    return df


def build_pipeline(model: str, num_cols, cat_cols) -> Pipeline:
    numeric_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ]
    )

    if model == "logreg":
        clf = LogisticRegression(max_iter=2000)
    else:
        clf = RandomForestClassifier(
            n_estimators=500, random_state=42, n_jobs=-1
        )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data", help="Folder with CSV")
    parser.add_argument("--csv", type=str, default="/Users/murad/Desktop/mldoc/heart-ml/data/Heart_disease_cleveland_new.csv", help="Optional explicit CSV path")
    parser.add_argument("--model", choices=["rf", "logreg"], default="rf")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = Path(args.csv) if args.csv else find_csv(data_dir)

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    df = standardize_dataframe(df)

    # Expected Cleveland columns (we’ll use what’s available)
    numeric_candidates = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    categorical_candidates = ["sex", "fbs", "exang", "cp", "restecg", "slope", "thal"]

    avail_num = [c for c in numeric_candidates if c in df.columns]
    avail_cat = [c for c in categorical_candidates if c in df.columns]

    if not avail_num and not avail_cat:
        raise ValueError("No known feature columns found. Check your CSV headers.")

    X = df[avail_num + avail_cat].copy()
    y = df["target_bin"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipe = build_pipeline(args.model, avail_num, avail_cat)
    pipe.fit(X_train, y_train)

    # quick stability check with CV (on train split)
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
    print(f"\n5-fold CV F1 (train): mean={cv.mean():.3f} ± {cv.std():.3f}")

    # test metrics
    y_pred = pipe.predict(X_test)
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred))

    # save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"heart_{args.model}.joblib"
    dump(pipe, model_path)
    print(f"\nSaved model → {model_path.resolve()}")

    # Save the feature schema for inference
    schema = {
        "numeric": avail_num,
        "categorical": avail_cat,
    }
    schema_path = out_dir / "feature_schema.json"
    pd.Series(schema).to_json(schema_path)
    print(f"Saved feature schema → {schema_path.resolve()}")


if __name__ == "__main__":
    main()
