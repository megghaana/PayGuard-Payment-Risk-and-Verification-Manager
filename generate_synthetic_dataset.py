"""Generate a reproducible, webhook-shaped dataset for pipeline prototyping.

This data is synthetic. It validates engineering and LightGBM evaluation only;
it must not be presented as a production fraud dataset.
"""

from __future__ import annotations

import csv
import random
from datetime import UTC, datetime, timedelta
from math import exp, log1p
from pathlib import Path

from feature_extraction import FEATURE_COLUMNS


ROW_COUNT = 50_000
RANDOM_SEED = 42
OUTPUT_PATH = Path("payguard_razorpay_synthetic_v2.csv")

METHODS = ["card", "upi", "netbanking", "wallet", "emi", "pay_later"]
METHOD_WEIGHTS = [0.34, 0.36, 0.12, 0.08, 0.05, 0.05]
CARD_NETWORKS = ["visa", "mastercard", "rupay", "amex"]
ISSUERS = ["hdfc", "icici", "sbi", "axis", "kotak_mahindra", "yes_bank", "rbl", "indusind"]


def sigmoid(value: float) -> float:
    return 1 / (1 + exp(-value))


def make_row(rng: random.Random, start: datetime) -> dict[str, object]:
    method = rng.choices(METHODS, weights=METHOD_WEIGHTS, k=1)[0]
    is_international = int(rng.random() < (0.12 if method == "card" else 0.015))
    amount = max(100, int(rng.lognormvariate(8.2, 1.15)))
    occurred_at = start + timedelta(seconds=rng.randint(0, 180 * 24 * 60 * 60))

    if method == "card":
        card_network = rng.choices(CARD_NETWORKS, weights=[0.36, 0.31, 0.27, 0.06], k=1)[0]
        card_type = rng.choices(["credit", "debit", "prepaid"], weights=[0.43, 0.48, 0.09], k=1)[0]
        issuer = rng.choice(ISSUERS)
        card_sub_type = rng.choices(["consumer", "business"], weights=[0.91, 0.09], k=1)[0]
    else:
        card_network = card_type = issuer = card_sub_type = "unknown"

    # The outcome rule is intentionally probabilistic, not a copied feature.
    # It represents a payment-failure prediction benchmark, not fraud truth.
    risk_logit = -4.3
    risk_logit += 1.25 * (log1p(amount) - 8.2)
    risk_logit += 1.8 * is_international
    risk_logit += {"netbanking": 1.3, "upi": -1.0, "wallet": -0.7, "emi": 1.15, "pay_later": 0.9, "card": 0.0}[method]
    risk_logit += 1.0 if occurred_at.hour < 5 else 0
    risk_logit += 1.2 if card_type == "prepaid" else 0
    risk_logit += 0.8 if issuer in {"rbl", "yes_bank"} else 0
    risk_logit += 0.35 if card_sub_type == "business" else 0
    risk_logit += rng.gauss(0, 0.25)
    outcome = int(rng.random() < sigmoid(risk_logit))

    return {
        "amount_paise": amount,
        "amount_log1p": round(log1p(amount), 6),
        "payment_method": method,
        "is_international": is_international,
        "card_network": card_network,
        "card_type": card_type,
        "issuer": issuer,
        "card_sub_type": card_sub_type,
        "payment_hour_utc": occurred_at.hour,
        "payment_day_of_week": occurred_at.weekday(),
        "outcome": outcome,
    }


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*FEATURE_COLUMNS, "outcome"])
        writer.writeheader()
        for _ in range(ROW_COUNT):
            writer.writerow(make_row(rng, start))
    print(f"Created {ROW_COUNT:,} rows at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
