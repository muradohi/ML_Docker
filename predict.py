"""
Simple batch prediction script.
- Loads models/heart_<model>.joblib
- Reads a CSV of new samples with the same feature columns used in training
- Writes predictions to predictions.csv
"""

import argparse
from pathlib import Path
import pandas as pd
from joblib import load
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/heart_rf.joblib")
    parser.add_argument("--schema-path", type=str, default="models/feature_schema.json")
    parser.add_argument("--input-csv", type=str, required=True, help="CSV with rows to predict")
    parser.add_argument("--output-csv", type=str, default="predictions.csv")
    args = parser.parse_args()

    model = load(args.model_path)

    # Load feature schema saved during training
    with open(args.schema_path, "r") as f:
        schema = json.load(f)
    num_cols = schema.get("numeric", [])
    cat_cols = schema.get("categorical", [])
    expected = num_cols + cat_cols

    df = pd.read_csv(args.input_csv)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for inference: {missing}")

    X = df[expected].copy()
    preds = model.predict(X)

    out = df.copy()
    out["prediction"] = preds
    out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions → {Path(args.output_csv).resolve()}")

if __name__ == "__main__":
    main()
