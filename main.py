"""Minimal Razorpay webhook receiver for inspecting test-mode webhook payloads."""

import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("razorpay_webhook")

app = FastAPI(title="PayGuard Razorpay Webhook Receiver")

# Keep the payload shape visible while preventing accidental secret/card logging.
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


def redact_sensitive_values(value: Any) -> Any:
    """Return a copy suitable for terminal logs, preserving JSON structure."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if key.lower() in SENSITIVE_FIELD_NAMES
            else redact_sensitive_values(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_values(item) for item in value]
    return value


def get_payment_id(payload: dict[str, Any]) -> str | None:
    """Find the Razorpay payment ID when the event includes a payment entity."""
    payment = payload.get("payload", {}).get("payment", {}).get("entity", {})
    payment_id = payment.get("id")
    return payment_id if isinstance(payment_id, str) else None


@app.post("/webhook")
async def razorpay_webhook(request: Request) -> JSONResponse:
    logger.info("Webhook received")

    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Webhook contained invalid JSON")
        return JSONResponse(status_code=400, content={"status": "invalid_json"})

    if not isinstance(payload, dict):
        logger.warning("Webhook JSON body was not an object")
        return JSONResponse(status_code=400, content={"status": "invalid_json"})

    event_type = payload.get("event", "unknown")
    payment_id = get_payment_id(payload)
    logger.info("Event type: %s", event_type)
    logger.info("Payment ID: %s", payment_id or "not available")

    # This is the real Razorpay body received by this endpoint. Sensitive values
    # are redacted, but all keys and the surrounding JSON structure are retained.
    logger.info("Webhook payload:\n%s", json.dumps(redact_sensitive_values(payload), indent=2))

    return JSONResponse(status_code=200, content={"status": "ok"})
