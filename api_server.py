from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from risk_graph import build_risk_graph
from risk_model import load_model_schema, load_risk_model
from risk_state import make_risk_state
from shap_explainer import explain_risk


app = FastAPI(title="PayGuard Risk API")

# Load these ONCE when the server starts.
model = load_risk_model()
schema = load_model_schema()
graph = build_risk_graph()


class RiskRequest(BaseModel):
    payment_status: str = "captured"
    features: dict[str, Any]


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "PayGuard Risk API"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/api/risk/analyze")
def analyze_risk(request: RiskRequest):

    try:
        # Run LightGBM + TreeSHAP
        explanation_result = explain_risk(
            model,
            request.features,
            schema
        )

        # Run deterministic decision + LLM explanation
        state = make_risk_state(
            request.payment_status,
            explanation_result
        )

        result = graph.invoke(state)

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )