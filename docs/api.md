# Prediction API

Flask app in `backend/pulse_api/`. Run it with `uv run pulse-api` (http://127.0.0.1:8000). Every
response is JSON unless stated otherwise; errors are `{"error": code, "detail": text, ...}`.

All endpoints share one risk policy (`backend/pulse_api/risk.py`):

| Level | Rule | Recommended action |
|---|---|---|
| `high` | probability >= top-10% cutoff (0.1698, the riskiest 10% of test-window orders) | Notify the customer proactively and review a carrier upgrade. |
| `elevated` | probability >= decision threshold (0.055) | Monitor the shipment; contact the customer if tracking stalls. |
| `normal` | otherwise | No action needed. |

High orders also get a customer-message draft (a fixed template, no LLM). Probabilities rank
orders; they over-predict on recent months (the model was trained at a 7.6% late rate and tested
at 3.8%), so the API reports levels and counts, not "expected late orders".

## GET /health

```json
{"status": "ok", "model": "logistic_regression", "threshold": 0.055, "top10_risk_cutoff": 0.169767,
 "risk_levels": {"elevated_from": 0.055, "high_from": 0.169767}, "contract_version": "1.0", ...}
```

Degraded mode (model or artifacts missing): `{"status": "degraded", "reason": ...}` with HTTP 200;
the scoring endpoints then answer 503 `model_unavailable`.

## POST /predict

Real-time scoring, all or nothing (one invalid record rejects the request).

Request: `{"records": [ {...}, ... ]}` with 1 to 1,000 records.

| Field | Required | Notes |
|---|---|---|
| `n_items`, `total_price`, `total_freight`, `max_installments` | yes | numbers within the contract bounds |
| `customer_state`, `seller_state`, `primary_category`, `payment_type` | yes | contract allowed values |
| `purchase_ts` | yes | `YYYY-MM-DD HH:MM:SS` |
| `estimated_delivery_date` or `promised_days` | one of them | `YYYY-MM-DD` / whole days |
| dimensions, weight, `payment_total`, `n_payments`, `distance_km`, zip prefixes | no | zip prefixes give the distance |
| `order_id` | no | echoed back, at most 64 characters, never a model feature |

Response 200:

```json
{"model": {"name": "logistic_regression", "threshold": 0.055, "top10_cutoff": 0.169767, "contract_version": "1.0"},
 "predictions": [{"index": 0, "order_id": "o-1", "late_probability": 0.1635, "predicted_late": true,
                  "top_10pct_risk": false, "risk_level": "elevated",
                  "recommended_action": "Monitor the shipment; contact the customer if tracking stalls."}],
 "warnings": []}
```

| Status | `error` | When |
|---|---|---|
| 400 | `invalid_request` / `validation_failed` | not a JSON object with `records`; 0 or more than 1,000 records |
| 422 | `validation_failed` | field errors: `details[] = {index, field, message}` |
| 503 | `model_unavailable` | degraded mode |

## POST /predict/batch

Batch scoring with partial success: valid rows are scored and ranked, invalid rows are reported.

Request: a CSV as the body (`Content-Type: text/csv`) or as a multipart `file`; one header row with
the fields above (plus optional `order_id`); at most 5,000 rows and 2 MB. Columns outside the
schema are ignored with one warning. Query `format=csv` returns a CSV instead of JSON.

Response 200 (JSON), predictions sorted from the highest risk:

```json
{"summary": {"rows": 3, "scored": 2, "rejected": 1, "high": 0, "elevated": 1, "normal": 1},
 "predictions": [{"row": 1, "order_id": "demo-1", "customer_state": "BA", "late_probability": 0.1635,
                  "risk_level": "elevated", "recommended_action": "...", "customer_message": ""}],
 "errors": [{"row": 3, "field": "customer_state", "message": "'XX' is not an allowed value"}],
 "warnings": [], "model": {...}}
```

`format=csv` columns: `row, order_id, customer_state, late_probability, risk_level,
recommended_action, customer_message`.

| Status | `error` | When |
|---|---|---|
| 400 | `no_file` | empty body and no `file` |
| 400 | `not_csv` / `empty` | not UTF-8 CSV; header without rows |
| 400 | `missing_columns` | `missing[]` lists the required columns that are absent |
| 413 | `too_large` | more than 2 MB or more than 5,000 rows |
| 422 | `no_valid_rows` | every row failed; `errors[]` says why |
| 503 | `model_unavailable` | degraded mode |

```bash
curl -s -X POST http://127.0.0.1:8000/predict/batch -H "Content-Type: text/csv" --data-binary @orders.csv
curl -s -X POST "http://127.0.0.1:8000/predict/batch?format=csv" -F "file=@orders.csv" -o scored_orders.csv
```

## GET /risk-list

The daily risk list, replayed from history: the delivered orders purchased on one day of the test
window (2018-05-01 to 2018-07-31), scored and ranked. In this coursework setting it stands in for
"yesterday's orders", and because the day is in the past, the real outcome can be shown.

Query: `date=YYYY-MM-DD` (required), `min_level=high|elevated|normal` (default `elevated`),
`include_outcome=true|false` (default `false`).

Response 200: the batch JSON plus `date`; with `include_outcome=true`, every prediction carries
`actual_late` and a `backtest` block reports the real late rate per level. Example for
2018-07-16: 238 orders, 42 high (11.9% actually late), 142 elevated (2.8%), 54 normal (1.9%),
against 4.2% for the whole day.

| Status | `error` | When |
|---|---|---|
| 400 | `invalid_date` / `invalid_level` | bad `date` or `min_level` |
| 404 | `out_of_window` / `no_orders` | date outside the test window; no orders that day |
| 503 | `model_unavailable` / `data_unavailable` | degraded mode; `clean_data.csv` missing |

```bash
curl -s "http://127.0.0.1:8000/risk-list?date=2018-07-16&include_outcome=true"
```
