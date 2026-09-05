"""Load and score the persisted PayGuard LightGBM pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("payguard_lightgbm_risk_model.joblib")
SCHEMA_PATH = Path("payguard_model_schema.json")


def load_risk_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the fitted preprocessing + LightGBM pipeline without retraining."""
    return joblib.load(model_path)


def load_model_schema(schema_path: Path = SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def prepare_transaction(transaction_features: dict[str, Any], schema: dict[str, Any]) -> pd.DataFrame:
    """Create one model-ready row using the persisted training column order."""
    feature_columns = schema["feature_columns"]
    missing = [column for column in feature_columns if column not in transaction_features]
    if missing:
        raise ValueError(f"Missing model features: {', '.join(missing)}")
    return pd.DataFrame([{column: transaction_features[column] for column in feature_columns}])


def predict_risk(model: Pipeline, transaction_features: dict[str, Any], schema: dict[str, Any]) -> float:
    """Return only the positive-class LightGBM risk probability."""
    row = prepare_transaction(transaction_features, schema)
    return float(model.predict_proba(row)[0, 1])


def risk_level(risk_score: float) -> str:
    """Map a score to a display level; this does not change the ML score."""
    if risk_score < 0.25:
        return "LOW"
    if risk_score < 0.60:
        return "MEDIUM"
    return "HIGH"


def transformed_feature_sources(model: Pipeline, schema: dict[str, Any]) -> list[str]:
    """Map each fitted encoded feature back to its original model field."""
    preprocessor = model.named_steps["preprocessor"]
    encoded_names = preprocessor.get_feature_names_out()
    categorical = schema["categorical_columns"]
    numeric = set(schema["numeric_columns"])
    sources: list[str] = []
    for encoded_name in encoded_names:
        local_name = encoded_name.split("__", 1)[-1]
        if local_name in numeric:
            sources.append(local_name)
            continue
        matched = next((column for column in categorical if local_name.startswith(f"{column}_")), None)
        if matched is None:
            raise ValueError(f"Could not map encoded feature '{encoded_name}' to an input field")
        sources.append(matched)
    return sources
