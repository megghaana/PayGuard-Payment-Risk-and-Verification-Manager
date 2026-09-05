"""Typed state passed between explainability and merchant-decision nodes."""

from __future__ import annotations

from typing import Any, TypedDict


class RiskState(TypedDict):
    payment_status: str
    risk_score: float
    risk_level: str
    risk_signals: list[dict[str, Any]]
    recommendation: str
    explanation: str


def make_risk_state(payment_status: str, explanation_result: dict[str, Any]) -> RiskState:
    """Create state from the fixed model score and fixed TreeSHAP output."""
    return {
        "payment_status": payment_status,
        "risk_score": explanation_result["risk_score"],
        "risk_level": explanation_result["risk_level"],
        "risk_signals": explanation_result["risk_signals"],
        "recommendation": "",
        "explanation": "",
    }
