"""Train and evaluate a LightGBM baseline on the synthetic v2 dataset."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from feature_extraction import FEATURE_COLUMNS


DATASET_PATH = Path("payguard_razorpay_synthetic_v2.csv")
METRICS_PATH = Path("synthetic_evaluation_metrics.json")


def main() -> None:
    data = pd.read_csv(DATASET_PATH)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        data[FEATURE_COLUMNS], data["outcome"], test_size=0.2, random_state=42, stratify=data["outcome"]
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    categorical = [column for column in FEATURE_COLUMNS if is_string_dtype(data[column])]
    numeric = [column for column in FEATURE_COLUMNS if column not in categorical]
    preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    model = lgb.LGBMClassifier(
        n_estimators=350, learning_rate=0.05, num_leaves=31,
        subsample=0.85, colsample_bytree=0.85, random_state=42,
        verbosity=-1,
    )
    pipeline = Pipeline([("features", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)
    validation_probabilities = pipeline.predict_proba(x_validation)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_validation, validation_probabilities)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    threshold = float(thresholds[f1_scores.argmax()])
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "rows": int(len(data)),
        "positive_rate": round(float(data["outcome"].mean()), 5),
        "decision_threshold": round(threshold, 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, predictions, digits=4))
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
