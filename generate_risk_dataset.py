"""Create a chronological, synthetic payment-risk dataset from the v2 source.

All behavioural features are calculated from events that occurred before the
current event. The generated labels are synthetic payment-failure outcomes.
"""

from __future__ import annotations

import csv
import random
from collections import deque
from datetime import UTC, datetime, timedelta
from math import exp, log1p
from pathlib import Path

import pandas as pd


SOURCE_PATH = Path("payguard_razorpay_synthetic_v2.csv")
OUTPUT_PATH = Path("payguard_razorpay_risk_dataset.csv")
ROW_COUNT = 80_000
RANDOM_SEED = 20260904
MERCHANT_COUNT = 400


def sigmoid(value: float) -> float:
    return 1 / (1 + exp(-value))


def purge_before(events: deque[datetime], cutoff: datetime) -> None:
    while events and events[0] < cutoff:
        events.popleft()


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    source = pd.read_csv(SOURCE_PATH)
    source_rows = source.to_dict("records")

    # A long-tailed merchant distribution gives some merchants realistic bursts.
    merchant_weights = [1 / ((index + 1) ** 0.72) for index in range(MERCHANT_COUNT)]
    start = datetime(2026, 1, 1, tzinfo=UTC)
    merchant_state: dict[int, dict[str, object]] = {}
    for merchant_id in range(MERCHANT_COUNT):
        baseline = int(rng.lognormvariate(8.3, 0.65))
        merchant_state[merchant_id] = {
            "baseline": baseline,
            # This represents pre-window history already available to the merchant.
            "sum": float(baseline * 20),
            "count": 20,
            "max": baseline * 3,
            "transactions": deque(),
            "failures": deque(),
            "seen_devices": set(),
            "seen_ips": set(),
            "account_start": start - timedelta(days=rng.randint(30, 1_095)),
        }

    pending: list[tuple[datetime, int, dict[str, object]]] = []
    for _ in range(ROW_COUNT):
        base = dict(rng.choice(source_rows))
        merchant_id = rng.choices(range(MERCHANT_COUNT), weights=merchant_weights, k=1)[0]
        day = rng.randint(0, 179)
        event_time = start + timedelta(
            days=day,
            hours=int(base["payment_hour_utc"]),
            minutes=rng.randint(0, 59),
            seconds=rng.randint(0, 59),
        )
        pending.append((event_time, merchant_id, base))
    pending.sort(key=lambda item: item[0])

    fieldnames = [
        "event_timestamp_utc",
        "amount_paise", "amount_log1p", "payment_method", "is_international",
        "card_network", "card_type", "issuer", "card_sub_type",
        "payment_hour_utc", "payment_day_of_week",
        "transactions_last_10min", "transactions_last_1hr",
        "failed_attempts_last_10min", "failed_attempts_last_1hr",
        "amount_vs_merchant_avg", "amount_vs_merchant_max",
        "new_device", "new_ip", "velocity_score",
        "merchant_account_age_days", "merchant_transactions_prior", "is_night_utc",
        "outcome",
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for event_time, merchant_id, base in pending:
            state = merchant_state[merchant_id]
            transactions = state["transactions"]
            failures = state["failures"]
            assert isinstance(transactions, deque) and isinstance(failures, deque)
            purge_before(transactions, event_time - timedelta(hours=1))
            purge_before(failures, event_time - timedelta(hours=1))
            transactions_last_1hr = len(transactions)
            transactions_last_10min = sum(time >= event_time - timedelta(minutes=10) for time in transactions)
            failed_attempts_last_1hr = len(failures)
            failed_attempts_last_10min = sum(time >= event_time - timedelta(minutes=10) for time in failures)

            merchant_average = float(state["sum"]) / int(state["count"])
            base_amount = int(base["amount_paise"])
            amount = max(100, int(base_amount * rng.lognormvariate(0, 0.32)))
            amount_vs_merchant_avg = amount / merchant_average
            amount_vs_merchant_max = amount / float(state["max"])

            seen_devices = state["seen_devices"]
            seen_ips = state["seen_ips"]
            assert isinstance(seen_devices, set) and isinstance(seen_ips, set)
            # Flags are created before the current transaction outcome is known.
            new_device = int(rng.random() < (0.36 if len(seen_devices) < 5 else 0.075))
            new_ip = int(rng.random() < (0.30 if len(seen_ips) < 8 else 0.09))
            if new_device:
                seen_devices.add(f"device_{len(seen_devices) + 1}")
            if new_ip:
                seen_ips.add(f"ip_{len(seen_ips) + 1}")

            velocity_score = min(
                100,
                round(
                    7 * transactions_last_10min + 2.2 * transactions_last_1hr
                    + 13 * failed_attempts_last_10min + 5 * failed_attempts_last_1hr
                    + 7 * new_device + 8 * new_ip,
                    2,
                ),
            )
            method = str(base["payment_method"])
            international = int(base["is_international"])
            card_type = str(base["card_type"])
            card_sub_type = str(base["card_sub_type"])
            issuer = str(base["issuer"])
            is_night = int(event_time.hour < 5)

            # The synthetic outcome has overlapping risk profiles and random noise.
            # No current-event outcome field is used in any feature above.
            risk_logit = -6.0
            risk_logit += 1.55 * max(0.0, min(log1p(amount_vs_merchant_avg), 2.8))
            risk_logit += 1.85 * max(0.0, min(amount_vs_merchant_max - 0.55, 1.5))
            risk_logit += 0.13 * velocity_score
            risk_logit += 1.65 * min(failed_attempts_last_1hr, 3)
            risk_logit += 2.0 * new_device + 2.0 * new_ip
            risk_logit += 1.20 if new_device and new_ip else 0
            risk_logit += 2.20 * international + 0.80 * is_night
            risk_logit += {"netbanking": 1.35, "emi": 1.20, "pay_later": 1.05, "card": 0.0, "upi": -0.45, "wallet": -0.35}[method]
            risk_logit += 1.20 if card_type == "prepaid" else 0
            risk_logit += 0.50 if card_sub_type == "business" else 0
            risk_logit += 0.45 if issuer in {"rbl", "yes_bank"} else 0
            risk_logit += rng.gauss(0, 0.45)
            outcome = int(rng.random() < sigmoid(risk_logit))

            writer.writerow({
                "event_timestamp_utc": event_time.isoformat(),
                "amount_paise": amount,
                "amount_log1p": round(log1p(amount), 6),
                "payment_method": method,
                "is_international": international,
                "card_network": str(base["card_network"]),
                "card_type": card_type,
                "issuer": issuer,
                "card_sub_type": card_sub_type,
                "payment_hour_utc": event_time.hour,
                "payment_day_of_week": event_time.weekday(),
                "transactions_last_10min": transactions_last_10min,
                "transactions_last_1hr": transactions_last_1hr,
                "failed_attempts_last_10min": failed_attempts_last_10min,
                "failed_attempts_last_1hr": failed_attempts_last_1hr,
                "amount_vs_merchant_avg": round(amount_vs_merchant_avg, 6),
                "amount_vs_merchant_max": round(amount_vs_merchant_max, 6),
                "new_device": new_device,
                "new_ip": new_ip,
                "velocity_score": velocity_score,
                "merchant_account_age_days": (event_time - state["account_start"]).days,
                "merchant_transactions_prior": int(state["count"]) - 20,
                "is_night_utc": is_night,
                "outcome": outcome,
            })

            # Update state only after all current-event features and label exist.
            transactions.append(event_time)
            if outcome:
                failures.append(event_time)
            state["sum"] = float(state["sum"]) + amount
            state["count"] = int(state["count"]) + 1
            state["max"] = max(float(state["max"]), amount)

    print(f"Created {ROW_COUNT:,} chronological synthetic rows at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
