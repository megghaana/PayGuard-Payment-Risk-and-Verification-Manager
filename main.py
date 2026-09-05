"""
PayGuard - Razorpay Webhook + Real-Time Risk Analysis

Flow:
Razorpay Webhook
      ↓
Signature Verification
      ↓
Payment Status Check
      ↓
Feature Extraction + Behavioral Features
      ↓
LightGBM Risk Model
      ↓
TreeSHAP Explanation
      ↓
LangGraph Decision
      ↓
Latest Risk Result
"""

import hashlib
import hmac
import json
import logging
import os
from dotenv import load_dotenv
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from feature_extraction import extract_payment_features
from risk_model import load_risk_model, load_model_schema
from shap_explainer import explain_risk
from risk_graph import build_risk_graph
from risk_state import make_risk_state

load_dotenv()


# -------------------------------------------------------------------
# APP
# -------------------------------------------------------------------

app = FastAPI(
    title="PayGuard - AI Risk Manager",
    description="Razorpay payment verification and AI risk detection",
    version="1.0.0",
)


# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("payguard")


# -------------------------------------------------------------------
# LOAD MODEL ONCE
# -------------------------------------------------------------------

try:
    risk_model = load_risk_model()
    model_schema = load_model_schema()
    risk_graph = build_risk_graph()

    logger.info("PayGuard risk model loaded successfully")

except Exception as exc:
    risk_model = None
    model_schema = None
    risk_graph = None

    logger.exception("Failed to load PayGuard risk model: %s", exc)


# -------------------------------------------------------------------
# WEBHOOK DEDUPLICATION
# -------------------------------------------------------------------

processed_event_ids: set[str] = set()


# -------------------------------------------------------------------
# SIMPLE IN-MEMORY MERCHANT HISTORY
#
# This is enough for the hackathon demo.
#
# IMPORTANT:
# Vercel serverless instances are not persistent.
# For production, replace this with Redis/Postgres/Supabase/etc.
# -------------------------------------------------------------------

merchant_transactions = defaultdict(deque)
merchant_failed_transactions = defaultdict(deque)

merchant_stats: dict[str, dict[str, Any]] = defaultdict(
    lambda: {
        "count": 0,
        "total_amount": 0.0,
        "max_amount": 0.0,
        "first_seen": None,
        "devices": set(),
        "ips": set(),
    }
)


# -------------------------------------------------------------------
# SENSITIVE FIELDS
# -------------------------------------------------------------------

SENSITIVE_FIELD_NAMES = {
    "authorization",
    "api_key",
    "api_secret",
    "card_number",
    "cvv",
    "password",
    "secret",
    "token",
    "webhook_secret",
}


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def redact_sensitive_values(value: Any) -> Any:
    """
    Redact obvious secrets before logging webhook payloads.
    """

    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if key.lower() in SENSITIVE_FIELD_NAMES
            else redact_sensitive_values(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            redact_sensitive_values(item)
            for item in value
        ]

    return value


def get_payment_entity(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Safely extract Razorpay payment.entity.
    """

    return (
        payload
        .get("payload", {})
        .get("payment", {})
        .get("entity", {})
    )


def get_payment_id(payload: dict[str, Any]) -> str | None:
    """
    Extract Razorpay payment ID.
    """

    payment = get_payment_entity(payload)

    payment_id = payment.get("id")

    if isinstance(payment_id, str):
        return payment_id

    return None


def get_merchant_key(payload: dict[str, Any]) -> str:
    """
    Determine which merchant's transaction history to use.

    For the hackathon prototype, account_id is used when available.

    If account_id isn't available in the webhook, everything falls
    under a demo merchant bucket.
    """

    payment = get_payment_entity(payload)

    merchant_id = (
        payment.get("account_id")
        or payment.get("merchant_id")
        or "demo_merchant"
    )

    return str(merchant_id)


def get_payment_status(event_type: str) -> str:
    """
    Convert Razorpay webhook event into a simple payment status.
    """

    event = event_type.lower()

    if event in {"payment.captured", "order.paid"}:
        return "captured"

    if event == "payment.authorized":
        return "authorized"

    if event == "payment.failed":
        return "failed"

    return "unknown"


def verify_razorpay_signature(
    raw_body: bytes,
    received_signature: str | None,
) -> bool:
    """
    Verify Razorpay webhook signature.

    Razorpay uses HMAC-SHA256 over the RAW request body.
    """

    webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")

    if not webhook_secret:
        logger.error(
            "RAZORPAY_WEBHOOK_SECRET is not configured"
        )
        return False

    if not received_signature:
        logger.warning(
            "Webhook did not contain X-Razorpay-Signature"
        )
        return False

    expected_signature = hmac.new(
        webhook_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(
        expected_signature,
        received_signature,
    )


def parse_created_timestamp(payment: dict[str, Any]) -> datetime:
    """
    Convert Razorpay created_at Unix timestamp into UTC datetime.

    Falls back to current UTC time if unavailable.
    """

    created_at = payment.get("created_at")

    if isinstance(created_at, (int, float)):
        return datetime.fromtimestamp(
            created_at,
            tz=timezone.utc,
        )

    return datetime.now(timezone.utc)


# -------------------------------------------------------------------
# BEHAVIORAL FEATURE GENERATION
# -------------------------------------------------------------------

def build_behavioral_features(
    payload: dict[str, Any],
    base_features: dict[str, Any],
) -> dict[str, Any]:
    """
    Add the behavioral features required by the 22-feature model.

    These are calculated using previous transactions belonging
    to the same merchant.
    """

    payment = get_payment_entity(payload)

    merchant_key = get_merchant_key(payload)

    current_time = parse_created_timestamp(payment)

    current_timestamp = current_time.timestamp()

    history = merchant_transactions[merchant_key]
    failed_history = merchant_failed_transactions[merchant_key]

    stats = merchant_stats[merchant_key]

    # ---------------------------------------------------------------
    # Current transaction information
    # ---------------------------------------------------------------

    amount_paise = float(
        base_features.get("amount_paise", 0)
    )

    # ---------------------------------------------------------------
    # Remove old transactions from rolling windows
    # ---------------------------------------------------------------

    ten_minutes = 10 * 60
    one_hour = 60 * 60

    while history and current_timestamp - history[0][0] > one_hour:
        history.popleft()

    while (
        failed_history
        and current_timestamp - failed_history[0] > one_hour
    ):
        failed_history.popleft()

    # ---------------------------------------------------------------
    # Transaction velocity
    # ---------------------------------------------------------------

    transactions_last_10min = sum(
        1
        for timestamp, _amount in history
        if current_timestamp - timestamp <= ten_minutes
    )

    transactions_last_1hr = len(history)

    failed_attempts_last_10min = sum(
        1
        for timestamp in failed_history
        if current_timestamp - timestamp <= ten_minutes
    )

    failed_attempts_last_1hr = len(failed_history)

    # ---------------------------------------------------------------
    # Merchant baseline
    # ---------------------------------------------------------------

    previous_count = stats["count"]
    previous_total = stats["total_amount"]
    previous_max = stats["max_amount"]

    if previous_count > 0:
        merchant_average = (
            previous_total / previous_count
        )
    else:
        merchant_average = amount_paise

    if previous_max > 0:
        amount_vs_merchant_max = (
            amount_paise / previous_max
        )
    else:
        amount_vs_merchant_max = 1.0

    if merchant_average > 0:
        amount_vs_merchant_avg = (
            amount_paise / merchant_average
        )
    else:
        amount_vs_merchant_avg = 1.0

    # ---------------------------------------------------------------
    # Device / IP information
    #
    # Razorpay payment payloads may not contain these directly.
    # If your frontend/backend supplies them later, these can be
    # populated here.
    # ---------------------------------------------------------------

    device_id = (
        payment.get("device_id")
        or payment.get("device")
        or None
    )

    ip_address = (
        payment.get("ip")
        or payment.get("ip_address")
        or None
    )

    if device_id:
        new_device = int(
            device_id not in stats["devices"]
        )
    else:
        # No device information available.
        new_device = 0

    if ip_address:
        new_ip = int(
            ip_address not in stats["ips"]
        )
    else:
        # No IP information available.
        new_ip = 0

    # ---------------------------------------------------------------
    # Velocity score
    #
    # This is a deterministic behavioral feature used by the model.
    # It is NOT an AI prediction.
    # ---------------------------------------------------------------

    velocity_score = (
        transactions_last_10min * 1.0
        + transactions_last_1hr * 0.1
        + failed_attempts_last_10min * 2.0
        + failed_attempts_last_1hr * 0.5
    )

    # ---------------------------------------------------------------
    # Merchant account information
    # ---------------------------------------------------------------

    if stats["first_seen"] is None:
        merchant_account_age_days = 0.0
    else:
        age_seconds = (
            current_timestamp - stats["first_seen"]
        )

        merchant_account_age_days = max(
            age_seconds / 86400,
            0.0,
        )

    merchant_transactions_prior = previous_count

    # ---------------------------------------------------------------
    # Night indicator
    # ---------------------------------------------------------------

    payment_hour = int(
        base_features.get("payment_hour_utc", 0)
    )

    is_night_utc = int(
        payment_hour < 6 or payment_hour >= 22
    )

    # ---------------------------------------------------------------
    # Build final 22-feature dictionary
    # ---------------------------------------------------------------

    behavioral_features = {
        "transactions_last_10min": transactions_last_10min,
        "transactions_last_1hr": transactions_last_1hr,
        "failed_attempts_last_10min": failed_attempts_last_10min,
        "failed_attempts_last_1hr": failed_attempts_last_1hr,
        "amount_vs_merchant_avg": amount_vs_merchant_avg,
        "amount_vs_merchant_max": amount_vs_merchant_max,
        "new_device": new_device,
        "new_ip": new_ip,
        "velocity_score": velocity_score,
        "merchant_account_age_days": merchant_account_age_days,
        "merchant_transactions_prior": merchant_transactions_prior,
        "is_night_utc": is_night_utc,
    }

    return {
        **base_features,
        **behavioral_features,
    }


def update_merchant_history(
    payload: dict[str, Any],
    features: dict[str, Any],
    payment_status: str,
) -> None:
    """
    Store the current transaction AFTER calculating its features.

    This ordering is important.

    We don't want the current transaction influencing its own
    historical features.
    """

    merchant_key = get_merchant_key(payload)

    payment = get_payment_entity(payload)

    current_time = parse_created_timestamp(payment)

    timestamp = current_time.timestamp()

    amount_paise = float(
        features.get("amount_paise", 0)
    )

    history = merchant_transactions[merchant_key]

    history.append(
        (
            timestamp,
            amount_paise,
        )
    )

    # Keep only one hour of transaction history.
    one_hour = 60 * 60

    while (
        history
        and timestamp - history[0][0] > one_hour
    ):
        history.popleft()

    # Failed transactions are tracked separately.
    if payment_status == "failed":
        merchant_failed_transactions[
            merchant_key
        ].append(timestamp)

    failed_history = merchant_failed_transactions[
        merchant_key
    ]

    while (
        failed_history
        and timestamp - failed_history[0] > one_hour
    ):
        failed_history.popleft()

    # ---------------------------------------------------------------
    # Merchant statistics
    # ---------------------------------------------------------------

    stats = merchant_stats[merchant_key]

    if stats["first_seen"] is None:
        stats["first_seen"] = timestamp

    stats["count"] += 1
    stats["total_amount"] += amount_paise
    stats["max_amount"] = max(
        stats["max_amount"],
        amount_paise,
    )

    # ---------------------------------------------------------------
    # Device / IP history
    # ---------------------------------------------------------------

    payment_device = (
        payment.get("device_id")
        or payment.get("device")
    )

    payment_ip = (
        payment.get("ip")
        or payment.get("ip_address")
    )

    if payment_device:
        stats["devices"].add(
            str(payment_device)
        )

    if payment_ip:
        stats["ips"].add(
            str(payment_ip)
        )


# -------------------------------------------------------------------
# RUN PAYGUARD RISK ANALYSIS
# -------------------------------------------------------------------

def analyze_payment(
    payload: dict[str, Any],
    payment_status: str,
) -> dict[str, Any]:
    """
    Run:

    feature extraction
        ↓
    LightGBM
        ↓
    TreeSHAP
        ↓
    LangGraph
    """

    if risk_model is None or model_schema is None:
        raise RuntimeError(
            "Risk model is not loaded"
        )

    if risk_graph is None:
        raise RuntimeError(
            "Risk graph is not loaded"
        )

    # ---------------------------------------------------------------
    # Extract base Razorpay features
    # ---------------------------------------------------------------

    base_features = extract_payment_features(
        payload
    )

    if not base_features:
        raise ValueError(
            "Could not extract payment features"
        )

    # ---------------------------------------------------------------
    # Add behavioral features
    # ---------------------------------------------------------------

    features = build_behavioral_features(
        payload,
        base_features,
    )

    logger.info(
        "Final model features: %s",
        json.dumps(
            features,
            sort_keys=True,
            default=str,
        ),
    )

    # ---------------------------------------------------------------
    # LightGBM + TreeSHAP
    # ---------------------------------------------------------------

    explanation_result = explain_risk(
        risk_model,
        features,
        model_schema,
    )

    logger.info(
        "Risk score: %.6f",
        explanation_result["risk_score"],
    )

    logger.info(
        "Risk level: %s",
        explanation_result["risk_level"],
    )

    # ---------------------------------------------------------------
    # LangGraph
    # ---------------------------------------------------------------

    state = make_risk_state(
        payment_status,
        explanation_result,
    )

    final_state = risk_graph.invoke(
        state
    )

    # ---------------------------------------------------------------
    # Update history AFTER inference
    # ---------------------------------------------------------------

    update_merchant_history(
        payload,
        features,
        payment_status,
    )

    # ---------------------------------------------------------------
    # Return merchant-facing result
    # ---------------------------------------------------------------

    return {
        "payment_status": payment_status,
        "payment_id": get_payment_id(payload),
        "risk_score": final_state["risk_score"],
        "risk_level": final_state["risk_level"],
        "risk_signals": final_state["risk_signals"],
        "recommendation": final_state["recommendation"],
        "explanation": final_state["explanation"],
        "features": features,
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# -------------------------------------------------------------------
# LATEST RESULT
#
# Simple hackathon dashboard endpoint.
# -------------------------------------------------------------------

latest_risk_result: dict[str, Any] | None = None


# -------------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------------

@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "PayGuard",
        "model_loaded": risk_model is not None,
        "graph_loaded": risk_graph is not None,
    }


# -------------------------------------------------------------------
# LATEST RISK RESULT
# -------------------------------------------------------------------

@app.get("/api/risk/latest")
async def latest_risk() -> JSONResponse:

    if latest_risk_result is None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "no_transactions",
                "message": "No risk analysis has been performed yet.",
            },
        )

    return JSONResponse(
        status_code=200,
        content=latest_risk_result,
    )


# -------------------------------------------------------------------
# MANUAL RISK ANALYSIS ENDPOINT
#
# Useful for Swagger/testing before connecting the frontend.
# -------------------------------------------------------------------

@app.post("/api/risk/analyze")
async def manual_risk_analysis(
    request: Request,
) -> JSONResponse:

    try:
        body = await request.json()

    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid_json"
            },
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid_request"
            },
        )

    payment_status = body.get(
        "payment_status",
        "captured",
    )

    features = body.get("features")

    if not isinstance(features, dict):
        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid_request",
                "message": "Request must contain a 'features' object.",
            },
        )

    try:

        if risk_model is None or model_schema is None:
            raise RuntimeError(
                "Risk model is not loaded"
            )

        explanation_result = explain_risk(
            risk_model,
            features,
            model_schema,
        )

        state = make_risk_state(
            payment_status,
            explanation_result,
        )

        final_state = risk_graph.invoke(
            state
        )

        result = {
            "payment_status": payment_status,
            "risk_score": final_state["risk_score"],
            "risk_level": final_state["risk_level"],
            "risk_signals": final_state["risk_signals"],
            "recommendation": final_state["recommendation"],
            "explanation": final_state["explanation"],
        }

        return JSONResponse(
            status_code=200,
            content=result,
        )

    except Exception as exc:

        logger.exception(
            "Manual risk analysis failed: %s",
            exc,
        )

        return JSONResponse(
            status_code=500,
            content={
                "status": "analysis_failed",
                "message": str(exc),
            },
        )


# -------------------------------------------------------------------
# RAZORPAY WEBHOOK
# -------------------------------------------------------------------

@app.post(
    "/api/webhook",
    include_in_schema=False,
)
@app.post(
    "/webhook",
    include_in_schema=False,
)
async def razorpay_webhook(
    request: Request,
) -> JSONResponse:

    global latest_risk_result

    logger.info(
        "Razorpay webhook received"
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # Read the RAW body first.
    #
    # Do NOT call request.json() before signature verification.
    # ---------------------------------------------------------------

    try:
        raw_body = await request.body()

    except Exception:
        logger.exception(
            "Could not read webhook body"
        )

        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid_body"
            },
        )

    # ---------------------------------------------------------------
    # Verify Razorpay signature
    # ---------------------------------------------------------------

    received_signature = request.headers.get(
        "X-Razorpay-Signature"
    )

    if not verify_razorpay_signature(
        raw_body,
        received_signature,
    ):
        logger.warning(
            "Invalid Razorpay webhook signature"
        )

        return JSONResponse(
            status_code=401,
            content={
                "status": "invalid_signature"
            },
        )

    # ---------------------------------------------------------------
    # Parse JSON AFTER signature verification
    # ---------------------------------------------------------------

    try:
        payload = json.loads(
            raw_body
        )

    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ):
        logger.warning(
            "Webhook contained invalid JSON"
        )

        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid_json"
            },
        )

    if not isinstance(payload, dict):
        return JSONResponse(
            status_code=400,
            content={
                "status": "invalid_json"
            },
        )

    # ---------------------------------------------------------------
    # Event ID for duplicate webhook protection
    # ---------------------------------------------------------------

    event_id = request.headers.get(
        "x-razorpay-event-id"
    )

    if event_id:
        if event_id in processed_event_ids:
            logger.info(
                "Duplicate webhook ignored: %s",
                event_id,
            )

            return JSONResponse(
                status_code=200,
                content={
                    "status": "duplicate_ignored"
                },
            )

        processed_event_ids.add(
            event_id
        )

    # ---------------------------------------------------------------
    # Identify event
    # ---------------------------------------------------------------

    event_type = payload.get(
        "event",
        "unknown",
    )

    payment_id = get_payment_id(
        payload
    )

    payment_status = get_payment_status(
        event_type
    )

    logger.info(
        "Event type: %s",
        event_type,
    )

    logger.info(
        "Payment ID: %s",
        payment_id or "not available",
    )

    logger.info(
        "Payment status: %s",
        payment_status,
    )

    # ---------------------------------------------------------------
    # Only analyze captured payments
    #
    # Authorized ≠ captured.
    # Failed ≠ fraud.
    # ---------------------------------------------------------------

    if payment_status == "authorized":

        logger.info(
            "Payment authorized but not captured. "
            "Risk decision: WAIT."
        )

        latest_risk_result = {
            "payment_status": "authorized",
            "payment_id": payment_id,
            "risk_score": None,
            "risk_level": "PENDING",
            "risk_signals": [],
            "recommendation": "WAIT",
            "explanation": (
                "Payment is authorized but has not "
                "been captured yet. Do not release "
                "the order."
            ),
            "processed_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "decision": "WAIT",
            },
        )

    if payment_status == "failed":

        logger.info(
            "Payment failed. No release."
        )

        latest_risk_result = {
            "payment_status": "failed",
            "payment_id": payment_id,
            "risk_score": None,
            "risk_level": "FAILED",
            "risk_signals": [],
            "recommendation": "DO_NOT_RELEASE",
            "explanation": (
                "The payment failed. "
                "Do not release the order."
            ),
            "processed_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        # Store failed transaction for future
        # behavioral features.
        try:
            base_features = extract_payment_features(
                payload
            )

            if base_features:
                update_merchant_history(
                    payload,
                    base_features,
                    "failed",
                )

        except Exception as exc:
            logger.warning(
                "Could not update failed-payment history: %s",
                exc,
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "decision": "DO_NOT_RELEASE",
            },
        )

    if payment_status != "captured":

        logger.info(
            "Ignoring unsupported webhook event: %s",
            event_type,
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "ignored",
                "event": event_type,
            },
        )

    # ---------------------------------------------------------------
    # CAPTURED PAYMENT
    #
    # This is where PayGuard runs the AI risk pipeline.
    # ---------------------------------------------------------------

    try:

        logger.info(
            "Captured payment detected. "
            "Running PayGuard risk analysis."
        )

        result = analyze_payment(
            payload,
            payment_status,
        )

        latest_risk_result = result

        logger.info(
            "FINAL DECISION: %s",
            result["recommendation"],
        )

        logger.info(
            "FINAL RISK LEVEL: %s",
            result["risk_level"],
        )

        logger.info(
            "FINAL RISK SCORE: %.6f",
            result["risk_score"],
        )

        # -----------------------------------------------------------
        # Don't log entire result if it contains unnecessary
        # sensitive information.
        # -----------------------------------------------------------

        safe_result = {
            "payment_status": result[
                "payment_status"
            ],
            "payment_id": result[
                "payment_id"
            ],
            "risk_score": result[
                "risk_score"
            ],
            "risk_level": result[
                "risk_level"
            ],
            "recommendation": result[
                "recommendation"
            ],
            "explanation": result[
                "explanation"
            ],
        }

        logger.info(
            "PayGuard result: %s",
            json.dumps(
                safe_result,
                default=str,
            ),
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "decision": result[
                    "recommendation"
                ],
                "risk_level": result[
                    "risk_level"
                ],
            },
        )

    except Exception as exc:

        logger.exception(
            "PayGuard risk analysis failed: %s",
            exc,
        )

        # Webhook itself was valid, so don't report
        # invalid webhook just because our ML system failed.
        return JSONResponse(
            status_code=200,
            content={
                "status": "received",
                "analysis": "failed",
                "message": "Risk analysis failed",
            },
        )