"""TreeSHAP explanations for the existing PayGuard LightGBM pipeline."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from risk_model import (
    load_model_schema,
    predict_risk,
    prepare_transaction,
    risk_level,
    transformed_feature_sources,
)


def _tree_shap_values(model: Pipeline, prepared_row) -> np.ndarray:
    """Return one positive-class TreeSHAP vector in encoded feature space."""
    try:
        import shap
    except ImportError as error:
        raise RuntimeError("TreeSHAP requires shap. Install it with: python -m pip install shap") from error

    preprocessor = model.named_steps["preprocessor"]
    lightgbm_model = model.named_steps["model"]
    encoded_row = preprocessor.transform(prepared_row)
    if hasattr(encoded_row, "toarray"):
        encoded_row = encoded_row.toarray()
    explainer = shap.TreeExplainer(lightgbm_model.booster_)
    values = explainer.shap_values(encoded_row)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values[0]


def _display_value(feature: str, value: Any) -> Any:
    if feature == "is_international":
        return "international" if int(value) else "domestic"
    if feature in {"new_device", "new_ip"}:
        return "yes" if int(value) else "no"
    if isinstance(value, float):
        return round(value, 4)
    return value


def _impact_label(absolute_value: float, largest_value: float) -> str:
    if largest_value == 0 or absolute_value < largest_value / 3:
        return "low"
    if absolute_value < (largest_value * 2 / 3):
        return "medium"
    return "high"


def explain_risk(
    model: Pipeline,
    transaction_features: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one transaction and return its top five grouped TreeSHAP signals.

    TreeSHAP is run over the same fitted one-hot representation passed to
    LightGBM. One-hot contributions are summed back into their original,
    human-readable field such as ``payment_method = card``.
    """
    schema = schema or load_model_schema()
    prepared_row = prepare_transaction(transaction_features, schema)
    score = predict_risk(model, transaction_features, schema)
    shap_values = _tree_shap_values(model, prepared_row)
    source_features = transformed_feature_sources(model, schema)
    grouped_values: dict[str, float] = defaultdict(float)
    for source, shap_value in zip(source_features, shap_values, strict=True):
        grouped_values[source] += float(shap_value)

    top_features = sorted(grouped_values, key=lambda feature: abs(grouped_values[feature]), reverse=True)[:5]
    largest_value = max((abs(grouped_values[feature]) for feature in top_features), default=0.0)
    signals = [
        {
            "feature": feature,
            "value": _display_value(feature, transaction_features[feature]),
            "impact": _impact_label(abs(grouped_values[feature]), largest_value),
            "direction": (
                "increases_risk" if grouped_values[feature] > 0
                else "reduces_risk" if grouped_values[feature] < 0
                else "neutral"
            ),
            # TreeSHAP values are in LightGBM's raw-score (log-odds) space.
            "shap_value": round(grouped_values[feature], 6),
        }
        for feature in top_features
    ]
    return {"risk_score": round(score, 6), "risk_level": risk_level(score), "risk_signals": signals}
