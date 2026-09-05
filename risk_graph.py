from __future__ import annotations

import os
from typing import Any

from groq import Groq
from langgraph.graph import END, START, StateGraph

from risk_state import RiskState


def decision_node(state: RiskState) -> dict[str, Any]:
    """
    Deterministic decision maker.

    The LLM is NOT allowed to decide the action.
    """

    payment_status = state["payment_status"].lower()
    risk_level = state["risk_level"]

    if payment_status != "captured":
        recommendation = "WAIT"
    elif risk_level == "LOW":
        recommendation = "RELEASE"
    elif risk_level == "MEDIUM":
        recommendation = "VERIFY"
    else:
        recommendation = "HOLD"

    return {
        "recommendation": recommendation
    }


def llm_explanation_node(state: RiskState) -> dict[str, Any]:
    """
    Uses Llama through Groq to convert the structured
    model evidence into a merchant-friendly explanation.

    The LLM cannot modify:
        - risk_score
        - risk_level
        - recommendation
        - SHAP values
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        # Safe fallback if Groq is unavailable.
        return {
            "explanation": deterministic_explanation(state)
        }

    client = Groq(api_key=api_key)

    risk_score = state["risk_score"]
    risk_level = state["risk_level"]
    recommendation = state["recommendation"]
    payment_status = state["payment_status"]
    signals = state["risk_signals"]

    increasing = [
        signal
        for signal in signals
        if signal["direction"] == "increases_risk"
    ]

    evidence = "\n".join(
        f"- {signal['feature']} = {signal['value']} "
        f"({signal['impact']} impact, SHAP={signal['shap_value']:.4f})"
        for signal in increasing[:3]
    )

    prompt = f"""
You are a payment risk explanation assistant for a merchant.

Your job is ONLY to explain an already-made risk decision.

You MUST NOT change or question the decision.

Payment status: {payment_status}
Risk score: {risk_score:.4f}
Risk level: {risk_level}
Decision: {recommendation}

Top evidence from the explainability model:
{evidence}

Write a concise merchant-facing explanation in 1-2 sentences.

Rules:
- Do not claim this is definitely fraud.
- Do not invent information.
- Do not mention SHAP, LightGBM, Llama, Groq, or internal model details.
- Do not change the decision.
- If the decision is HOLD, clearly say the order should be held.
- If the decision is VERIFY, clearly say the order should be verified.
- If the decision is RELEASE, clearly say the order can be released.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain payment risk decisions accurately "
                        "and concisely. Never invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            max_tokens=150,
        )

        explanation = response.choices[0].message.content.strip()

        return {
            "explanation": explanation
        }

    except Exception:
        # If the LLM/API fails, the risk system still works.
        return {
            "explanation": deterministic_explanation(state)
        }


def deterministic_explanation(state: RiskState) -> str:
    """
    Safe fallback explanation when the LLM is unavailable.
    """

    recommendation = state["recommendation"]
    risk_level = state["risk_level"]
    signals = state["risk_signals"]

    increasing = [
        signal
        for signal in signals
        if signal["direction"] == "increases_risk"
    ]

    if recommendation == "WAIT":
        return "The payment has not been captured, so the order should remain on hold."

    if not increasing:
        return (
            f"The payment has a {risk_level.lower()} model risk level. "
            f"Recommended action: {recommendation}."
        )

    factors = ", ".join(
        f"{signal['feature']}={signal['value']}"
        for signal in increasing[:3]
    )

    action_text = {
        "RELEASE": "Release this order.",
        "VERIFY": "Verify this order before fulfillment.",
        "HOLD": "Hold this order for review.",
    }.get(recommendation, recommendation)

    return (
        f"{action_text} "
        f"The strongest factors increasing risk are {factors}."
    )


def build_risk_graph():
    """
    Build the LangGraph workflow.

    Decision is deterministic.
    LLM only writes the explanation.
    """

    graph = StateGraph(RiskState)

    graph.add_node("determine_recommendation", decision_node)
    graph.add_node("write_merchant_explanation", llm_explanation_node)

    graph.add_edge(START, "determine_recommendation")
    graph.add_edge(
        "determine_recommendation",
        "write_merchant_explanation",
    )
    graph.add_edge("write_merchant_explanation", END)

    return graph.compile()