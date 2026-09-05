# PayGuard: Payment Risk and Verification Manager

PayGuard is a merchant focused payment risk and verification system
built for the Razorpay AI Buildathon.

The main question it answers is:

> Can I safely release this order?

PayGuard first verifies the payment using Razorpay server side events.
Once a payment is captured, it analyzes the transaction using a LightGBM
risk model, explains the result using TreeSHAP, and uses LangGraph to
orchestrate the final decision.

The system produces three outcomes:

-   LOW → RELEASE
-   MEDIUM → VERIFY
-   HIGH → HOLD

## Problem

Merchants can lose money when they release an order based only on a
customer provided payment screenshot or an incomplete payment status.

PayGuard provides a server side verification and risk layer between the
payment and the merchant's fulfillment decision.

## Architecture

``` text
Razorpay Checkout
        |
        v
Razorpay Webhook
        |
        v
Payment Verification
        |
        v
Feature Engineering
        |
        v
LightGBM Risk Model
        |
        v
TreeSHAP Explanation
        |
        v
LangGraph Decision Flow
        |
        v
Merchant Decision
 RELEASE / VERIFY / HOLD
```

## Key Features

### Razorpay Payment Verification

PayGuard listens for Razorpay payment events and uses server side
verification instead of trusting customer supplied proof.

Supported events:

-   payment.authorized
-   payment.captured
-   payment.failed

The webhook signature is verified using HMAC SHA256, and duplicate event
IDs are handled to reduce repeated processing.

### Machine Learning Risk Scoring

The LightGBM model uses payment and behavioral signals including:

-   Transaction amount
-   Payment method
-   International payment status
-   Card information
-   Recent transaction velocity
-   Recent failed attempts
-   Amount compared with merchant history
-   New device
-   New IP address
-   Merchant account history
-   Payment time and day

The model produces a risk score that is mapped to LOW, MEDIUM, or HIGH.

### Explainable AI

TreeSHAP identifies the features contributing most to each prediction.

Example signals:

``` text
High transaction velocity → increases risk
Unusual transaction amount → increases risk
New device → increases risk
Long merchant history → reduces risk
```

This gives the merchant a reason behind the recommendation instead of
only a score.

### Decision Orchestration

LangGraph coordinates the risk analysis and decision workflow.

The language model is only used to make the explanation easier to
understand. It cannot change the risk score or final recommendation.

## Model Evaluation

The model was evaluated on a held out chronological test set.

-   80,000 synthetic transactions
-   22 input features
-   12,000 held out test transactions

  Metric                Result
  ------------------- --------
  Precision             51.82%
  Recall                50.77%
  F1 Score              51.29%
  Average Precision     53.92%

The dataset is synthetic and should not be interpreted as real Razorpay
fraud or payment failure statistics.

A chronological split was used to reduce the risk of using future
transaction behavior to predict earlier transactions.

## Tech Stack

-   Python
-   FastAPI
-   Razorpay Test Mode
-   LightGBM
-   TreeSHAP
-   LangGraph
-   Groq
-   Llama
-   Pandas
-   Scikit-learn
-   Vercel

## Project Structure

``` text
PayGuard
├── api
│   └── index.py
├── main.py
├── feature_extraction.py
├── risk_model.py
├── shap_explainer.py
├── risk_state.py
├── risk_graph.py
├── train_risk_model.py
├── test_risk_explainability.py
├── index.html
├── requirements.txt
└── vercel.json
```

## Running Locally

Install dependencies:

``` bash
pip install -r requirements.txt
```

Create a `.env` file with your Test Mode credentials:

``` env
RAZORPAY_KEY_ID=your_test_key
RAZORPAY_KEY_SECRET=your_test_secret
RAZORPAY_WEBHOOK_SECRET=your_webhook_secret
GROQ_API_KEY=your_groq_key
```

Start the application:

``` bash
uvicorn main:app --reload
```

## API Endpoints

``` text
GET  /api/health
GET  /api/risk/latest
POST /api/risk/analyze
POST /api/create-order
POST /api/webhook
```

## Razorpay Test Mode

The demo uses Razorpay Test Mode so the complete payment flow can be
demonstrated without real money.

The flow is:

``` text
Razorpay Test Payment
        ↓
Payment Captured
        ↓
Webhook Received
        ↓
Risk Score Generated
        ↓
SHAP Signals Generated
        ↓
Merchant Decision
```

The demonstrated low risk payment received a risk score of approximately
0.017 and was classified as LOW with a RELEASE recommendation.

## Security

-   Payment status is verified server side.
-   Razorpay webhook signatures are validated.
-   Webhook event IDs are used to reduce duplicate processing.
-   API credentials and webhook secrets are stored as environment
    variables.
-   Sensitive payment identifiers are not used as model features.
-   The language model cannot override the machine learning decision.

## Future Improvements

-   Calibrate the risk score into a true probability.
-   Add a larger and more realistic labeled fraud dataset.
-   Add persistent merchant transaction history.
-   Add automated verification workflows for medium risk payments.
-   Add webhook monitoring and alerting.
-   Add more behavioral signals.
-   Evaluate the model with real merchant feedback.

## Disclaimer

This project was created as a hackathon prototype.

The training data is synthetic and the model should not be treated as a
production fraud detection system without further validation on
representative real world data.
