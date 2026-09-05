"""Convert Razorpay payment webhook data into safe model features."""

from __future__ import annotations

from datetime import UTC, datetime
from math import log1p
from typing import Any


FEATURE_COLUMNS = [
    "amount_paise",
    "amount_log1p",
    "payment_method",
    "is_international",
    "card_network",
    "card_type",
    "issuer",
    "card_sub_type",
    "payment_hour_utc",
    "payment_day_of_week",
]


def _category(value: Any) -> str:
    """Normalise missing categories without retaining personal identifiers."""
    if value is None or value == "":
        return "unknown"
    return str(value).strip().lower().replace(" ", "_")


def get_payment_entity(webhook_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Get the payment entity from a Razorpay payment webhook payload."""
    payment = webhook_payload.get("payload", {}).get("payment", {})
    entity = payment.get("entity", {}) if isinstance(payment, dict) else {}
    return entity if isinstance(entity, dict) else None


def extract_payment_features(webhook_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract only fields available at payment authorisation time.

    IDs, contact information, card last-four digits and outcome-revealing fields
    are deliberately excluded from the model feature vector.
    """
    payment = get_payment_entity(webhook_payload)
    if not payment:
        return None

    amount = int(payment.get("amount") or 0)
    card = payment.get("card") if isinstance(payment.get("card"), dict) else {}
    timestamp = payment.get("created_at") or webhook_payload.get("created_at")
    occurred_at = datetime.fromtimestamp(int(timestamp), tz=UTC) if timestamp else datetime.now(UTC)

    return {
        "amount_paise": amount,
        "amount_log1p": round(log1p(amount), 6),
        "payment_method": _category(payment.get("method")),
        "is_international": int(bool(payment.get("international"))),
        "card_network": _category(card.get("network")),
        "card_type": _category(card.get("type")),
        "issuer": _category(card.get("issuer")),
        "card_sub_type": _category(card.get("sub_type")),
        "payment_hour_utc": occurred_at.hour,
        "payment_day_of_week": occurred_at.weekday(),
    }


def outcome_label(webhook_payload: dict[str, Any]) -> int | None:
    """Return 0 for captured and 1 for failed events, otherwise no label.

    This is for creating training labels after an outcome occurs. It must never
    be included in the live model feature vector.
    """
    event_type = webhook_payload.get("event")
    if event_type == "payment.captured":
        return 0
    if event_type == "payment.failed":
        return 1
    return None
