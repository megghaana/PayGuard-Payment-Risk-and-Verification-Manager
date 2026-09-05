import hashlib
import hmac
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

secret = os.getenv("RAZORPAY_WEBHOOK_SECRET")

if not secret:
    raise RuntimeError(
        "RAZORPAY_WEBHOOK_SECRET was not loaded from .env"
    )

print("Secret loaded:", bool(secret))
print("Secret length:", len(secret))
payload = {
    "event": "payment.captured",
    "payload": {
        "payment": {
            "entity": {
                "id": "pay_test_123",
                "amount": 10000,
                "currency": "INR",
                "method": "card",
                "international": False,
                "card": {
                    "network": "Mastercard",
                    "type": "credit",
                    "sub_type": "consumer",
                },
                "bank": "HDFC",
                "created_at": 1757060000,
            }
        }
    }
}

raw_body = json.dumps(
    payload,
    separators=(",", ":"),
).encode("utf-8")

signature = hmac.new(
    secret.encode("utf-8"),
    raw_body,
    hashlib.sha256,
).hexdigest()

response = requests.post(
    "http://127.0.0.1:8000/api/webhook",
    data=raw_body,
    headers={
        "Content-Type": "application/json",
        "X-Razorpay-Signature": signature,
        "x-razorpay-event-id": "test-event-001",
    },
)

print(response.status_code)
print(response.json())