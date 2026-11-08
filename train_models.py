"""Train the 5 classifiers on ONE dataset and save everything the app needs.

Usage:
    python train_models.py --data data/dataset.csv --target <target_column>
"""
import argparse, json, os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import build_models, compute_metrics, MODEL_NAMES, METRIC_COLS

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to the full dataset CSV")
ap.add_argument("--target", required=True, help="name of the target column")
ap.add_argument("--drop", nargs="*", default=[], help="ID-like columns to drop")
ap.add_argument("--test_size", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

df = pd.read_csv(args.data).drop(columns=args.drop).dropna(subset=[args.target])
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]-1} features")
assert df.shape[0] >= 500 and df.shape[1] - 1 >= 12, "Need >=500 rows and >=12 features"

X, y_raw = df.drop(columns=[args.target]), df[args.target]
le = LabelEncoder().fit(y_raw)
y = le.transform(y_raw)
n_classes = len(le.classes_)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

# Save the held-out test set (features + target, original labels) for the app
os.makedirs("model", exist_ok=True)
test_df = X_te.copy()
test_df[args.target] = le.inverse_transform(y_te)
test_df.to_csv("test_data.csv", index=False)

rows = []
for name, pipe in build_models(X_tr, args.seed).items():
    pipe.fit(X_tr, y_tr)
    m = compute_metrics(y_te, pipe.predict(X_te), pipe.predict_proba(X_te), n_classes)
    rows.append({"ML Model Name": name, **m})
    joblib.dump(pipe, f"model/{name.lower().replace(' ', '_')}.joblib")
    print(name, {k: round(v, 4) for k, v in m.items()})

joblib.dump(le, "model/label_encoder.joblib")
json.dump({"target": args.target, "features": list(X.columns),
           "classes": [str(c) for c in le.classes_]},
          open("model/meta.json", "w"), indent=2)

res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics_summary.csv", index=False)
print("\n", res.to_markdown(index=False))
