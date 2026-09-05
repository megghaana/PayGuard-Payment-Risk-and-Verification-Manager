# PayGuard ML Data Pipeline

## Training dataset

Generate the synthetic webhook-derived training dataset:

```powershell
python generate_synthetic_dataset.py
```

This creates `payguard_razorpay_synthetic_v2.csv` with 50,000 rows and these
model features:

- `amount_paise`, `amount_log1p`
- `payment_method`, `is_international`
- `card_network`, `card_type`, `issuer`, `card_sub_type`
- `payment_hour_utc`, `payment_day_of_week`

`outcome` is the label: `0` for captured and `1` for failed. The dataset is
synthetic and suitable for exercising the pipeline, not for claiming production
fraud-detection performance.

## Evaluation

```powershell
python train_evaluate.py
```

The script uses train/validation/test splits. It selects the classification
threshold on validation data and reports precision, recall and average precision
once on the holdout test data. Results are written to
`synthetic_evaluation_metrics.json`.

## Real-time feature extraction

`feature_extraction.py` is the single source of truth for features. The webhook
receiver calls it for every payment event and logs the extracted feature vector.
Use the same function for data collection and live inference so model columns
and categories stay aligned.

The extractor deliberately excludes IDs, email, contact details, card last-four
digits, error fields and payment status. Outcome fields must only be used later
to create labels, never as live prediction inputs.
