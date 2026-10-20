# Model card: late-delivery risk at checkout

## Model details

- **Model:** `logistic_regression` (LogisticRegression): LogisticRegression on scaled numerics + one-hot categories (threshold tuned for the imbalance).
- **Selected by:** highest mean CV PR-AUC (4 decimals) among non-baseline models; the lean logistic regression wins when within 2% of the best.
- **Decision threshold:** 0.055 (tuned on out-of-fold F1).
- **Reproducibility:** seed 42, 2 threads, scikit-learn 1.7.2, Python 3.13.13.
- **Contract:** dataset_contract v1.0, sha256 `05f94f8cfd5c74843278ec9ec2c557bd357c2b63c7574a0d7e830b473b9b08e5`.

## Model purpose

Estimate, at checkout, the probability that an Olist order will be delivered after the date promised to the customer, so operations can prioritize risky orders (proactive messages, carrier upgrades, promise adjustments).

- **Intended users:** Marketplace operations and customer-experience teams.
- **Out of scope:**
  - Denying service, changing prices or penalizing customers or sellers.
  - Orders outside Brazil or from other marketplaces without revalidation.
  - Predicting the exact delivery date (this is a yes/no lateness risk).

## Training data summary

| fact | value |
|---|---|
| training rows | 70,851 (2017-01-01 .. 2018-04-30) |
| test rows | 19,001 (2018-05-01 .. 2018-07-31) |
| late rate train / test | 7.6% / 3.8% |
| features | 25 checkout-time features |
| dataset contract | v1.0 sha256 05f94f8cfd5c |
| raw data | Olist GitHub commit d9e49802f3e9 |
| privacy | pseudonymous ids; no names, cities or review text |

70,851 delivered orders purchased 2017-01-01..2018-04-30 (7.6% late); tested on 19,001 orders from 2018-05-01..2018-07-31 (3.8% late).

## Metrics

| metric | logistic_regression | baseline_prior |
|---|---|---|
| CV PR-AUC (mean ± std) | 0.180 ± 0.083 | n/a |
| test ROC-AUC | 0.733 | 0.500 |
| test PR-AUC | 0.102 | 0.038 |
| precision in riskiest 10% | 12.1% | 4.3% |
| Brier score | 0.038 | 0.038 |
| precision / recall at threshold 0.055 | 5.8% / 83.0% | n/a |

## Limitations

- Measured among delivered orders only; orders that never arrived are excluded.
- Right-censoring: August 2018 is held out because late orders may not have arrived by the 2018-10-17 snapshot.
- The late rate is non-stationary; the decision threshold must be re-tuned on recent data.
- Data from 2017-2018 on one marketplace; patterns may have changed.
- Associations, not causes: the model does not explain why an order is late.

## Ethical considerations

- Regional bias: customers in North and Northeast states have higher late rates; interventions must improve their service, never deprioritize them.
- Seller fairness: scores are for operational prioritization only, not seller penalties.
- Privacy: identifiers are pseudonymous, but zip prefix plus timestamps are quasi-identifiers; only aggregates are shown in dashboards.
- License: the Olist data is shared for non-commercial use (CC BY-NC-SA 4.0 listing); this model is coursework, not a production system.
- Portuguese review text is never used, avoiding sensitive free-text content.

## Monitoring

- Track the monthly late rate, PR-AUC and calibration, overall and by state.
- Re-tune the threshold and retrain when the late rate shifts.
- Alert when input distributions (promised days, freight, states) drift.

## How to use

Run the API (`uv run python -m api.serve`) and POST checkout records:

```json
{"records": [{"n_items": 1, "total_price": 129.9, "total_freight": 21.5, "customer_state": "BA", "seller_state": "SP", "primary_category": "furniture_decor", "payment_type": "boleto", "max_installments": 1, "purchase_ts": "2018-06-12 14:03:00", "estimated_delivery_date": "2018-07-05 00:00:00"}]}
```

---
<sub>narrative_source: template | grounding: metrics.json sha256=bb25900b6ecc, model_meta.json sha256=40c72d3d8a96</sub>
