"""Train, select a threshold and save a LightGBM payment-risk pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATASET_PATH = Path("payguard_razorpay_risk_dataset.csv")
MODEL_PATH = Path("payguard_lightgbm_risk_model.joblib")
SCHEMA_PATH = Path("payguard_model_schema.json")
METRICS_PATH = Path("payguard_risk_model_metrics.json")
TARGET = "outcome"
TIMESTAMP = "event_timestamp_utc"
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def score_at_threshold(y_true: pd.Series, probabilities, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def main() -> None:
    data = pd.read_csv(DATASET_PATH)
    data = data.sort_values(TIMESTAMP).reset_index(drop=True)
    feature_columns = [column for column in data.columns if column not in {TARGET, TIMESTAMP}]
    categorical_columns = [column for column in feature_columns if is_string_dtype(data[column])]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    # Chronological 70% train, 15% validation, 15% completely held-out test.
    total = len(data)
    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)
    train, validation, test = data.iloc[:train_end], data.iloc[train_end:validation_end], data.iloc[validation_end:]

    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_columns),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_columns),
    ])
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.035, num_leaves=31,
        min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
        random_state=42, verbosity=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(train[feature_columns], train[TARGET])

    validation_probabilities = pipeline.predict_proba(validation[feature_columns])[:, 1]
    precision, recall, thresholds = precision_recall_curve(validation[TARGET], validation_probabilities)
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    recommended_threshold = float(thresholds[f1_values.argmax()])

    test_probabilities = pipeline.predict_proba(test[feature_columns])[:, 1]
    threshold_table = [score_at_threshold(test[TARGET], test_probabilities, threshold) for threshold in THRESHOLDS]
    selected = score_at_threshold(test[TARGET], test_probabilities, recommended_threshold)
    metrics = {
        "dataset_rows": total,
        "feature_count": len(feature_columns),
        "positive_rate": round(float(data[TARGET].mean()), 5),
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "decision_threshold": round(recommended_threshold, 4),
        "precision": selected["precision"], "recall": selected["recall"], "f1": selected["f1"],
        "average_precision": round(float(average_precision_score(test[TARGET], test_probabilities)), 4),
        "confusion_matrix": [[selected["true_negatives"], selected["false_positives"]], [selected["false_negatives"], selected["true_positives"]]],
        "false_positives": selected["false_positives"], "false_negatives": selected["false_negatives"],
        "threshold_table": threshold_table,
    }
    schema = {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "timestamp_column": TIMESTAMP,
        "target_column": TARGET,
        "recommended_threshold": round(recommended_threshold, 4),
        "dataset_kind": "synthetic hackathon prototype",
    }
    joblib.dump(pipeline, MODEL_PATH)
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved schema to {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
