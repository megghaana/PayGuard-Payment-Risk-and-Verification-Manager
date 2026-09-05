"""Run TreeSHAP and the deterministic risk graph for three real dataset rows."""

from __future__ import annotations

import json

import pandas as pd

from risk_graph import build_risk_graph
from risk_model import load_model_schema, load_risk_model
from risk_state import make_risk_state
from shap_explainer import explain_risk


DATASET_PATH = "payguard_razorpay_risk_dataset.csv"


def closest_transaction(data: pd.DataFrame, scores, lower: float, upper: float, target: float) -> dict:
    candidates = data[(scores >= lower) & (scores < upper)].copy()
    if candidates.empty:
        raise RuntimeError(f"No transaction found for score range {lower} to {upper}")
    candidates["distance"] = (scores[candidates.index] - target).abs()
    return candidates.sort_values("distance").iloc[0].to_dict()


def main() -> None:
    model = load_risk_model()
    schema = load_model_schema()
    data = pd.read_csv(DATASET_PATH)
    features = schema["feature_columns"]
    scores = pd.Series(model.predict_proba(data[features])[:, 1], index=data.index)
    graph = build_risk_graph()
    examples = {
        "LOW-RISK": closest_transaction(data, scores, 0.0, 0.25, 0.12),
        "MEDIUM-RISK": closest_transaction(data, scores, 0.25, 0.60, 0.42),
        "HIGH-RISK": closest_transaction(data, scores, 0.60, 1.01, 0.75),
    }
    for label, transaction in examples.items():
        transaction_features = {feature: transaction[feature] for feature in features}
        explanation_result = explain_risk(model, transaction_features, schema)
        state = graph.invoke(make_risk_state("captured", explanation_result))
        print(f"\n{'=' * 14} {label} {'=' * 14}")
        print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
