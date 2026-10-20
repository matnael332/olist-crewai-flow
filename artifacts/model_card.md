# Model card: late-delivery risk at checkout

## Model details

- **Model:** `logistic_regression` (LogisticRegression): LogisticRegression on scaled numerics + one-hot categories (threshold tuned for the imbalance).
- **Selected by:** highest mean CV PR-AUC (4 decimals) among non-baseline models; the lean logistic regression wins when within 2% of the best.
- **Decision threshold:** 0.055 (tuned on out-of-fold F1).
- **Reproducibility:** seed 42, 2 threads, scikit-learn 1.7.2, Python 3.13.13.
- **Contract:** dataset_contract v1.0, sha256 `8fe2e5d99bbc4d0b4023f97786c8864f9ff3d7f55c6473e634ee57ecae7da0fe`.

## Model purpose

The purpose of this model is to predict late deliveries for e-commerce orders, enabling proactive operational interventions and improving customer satisfaction.

- **Intended users:** Operations teams, logistics planners, and customer service management within the e-commerce platform.
- **Out of scope:**
  - Denying service, changing prices or penalizing customers or sellers.
  - Orders outside Brazil or from other marketplaces without revalidation.
  - Predicting the exact delivery date (this is a yes/no lateness risk).
  - Determining individual employee or driver performance metrics.
  - Automated punitive actions against third-party sellers without human review.
  - Financial credit scoring or fraud detection.

## Training data summary

| fact | value |
|---|---|
| training rows | 70,851 (2017-01-01 .. 2018-04-30) |
| test rows | 19,001 (2018-05-01 .. 2018-07-31) |
| late rate train / test | 7.6% / 3.8% |
| features | 25 checkout-time features |
| dataset contract | v1.0 sha256 8fe2e5d99bbc |
| raw data | Olist GitHub commit d9e49802f3e9 |
| privacy | pseudonymous ids; no names, cities or review text |

The training set contains 70,851 rows spanning from 2017-01-01 to 2018-04-30 with a late rate prevalence of 7.6%. A test set of 19,001 rows spans from 2018-05-01 to 2018-07-31 with a late rate prevalence of 3.8%. Additionally, a censored buffer dataset contains 6,351 rows from 2018-08-01 to 2018-08-31 with a late rate prevalence of 6.2%.

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
- Modest overall precision of 5.8% at the chosen threshold due to the low base rate.
- Brier score of 0.0384 is slightly worse than alternative models.
- Performance is sensitive to distribution shift between training and testing windows.

## Ethical considerations

- Regional bias: customers in North and Northeast states have higher late rates; interventions must improve their service, never deprioritize them.
- Seller fairness: scores are for operational prioritization only, not seller penalties.
- Privacy: identifiers are pseudonymous, but zip prefix plus timestamps are quasi-identifiers; only aggregates are shown in dashboards.
- License: the Olist data is shared for non-commercial use (CC BY-NC-SA 4.0 listing); this model is coursework, not a production system.
- Portuguese review text is never used, avoiding sensitive free-text content.
- Regional bias between Brazilian states may affect prediction accuracy for less-represented geographic areas.
- Seller fairness must be maintained to ensure smaller merchants are not unfairly penalized by routing or fulfillment predictions.
- Privacy of zip prefixes and timestamps must be strictly preserved to prevent re-identification of buyers or sellers.
- Data usage must comply with non-commercial data license restrictions.

## Monitoring

- Track the monthly late rate, PR-AUC and calibration, overall and by state.
- Re-tune the threshold and retrain when the late rate shifts.
- Alert when input distributions (promised days, freight, states) drift.
- Track daily and weekly prediction distributions to catch data drift resulting from temporal changes in late rates.
- Monitor precision and recall metrics continuously on rolling test windows.
- Audit predictions across different Brazilian states regularly to detect and mitigate regional bias.

## How to use

Run the API (`uv run python -m api.serve`) and POST checkout records:

```json
{"records": [{"n_items": 1, "total_price": 129.9, "total_freight": 21.5, "customer_state": "BA", "seller_state": "SP", "primary_category": "furniture_decor", "payment_type": "boleto", "max_installments": 1, "purchase_ts": "2018-06-12 14:03:00", "estimated_delivery_date": "2018-07-05 00:00:00"}]}
```

---
<sub>narrative_source: template+llm | grounding: metrics.json sha256=bb25900b6ecc, model_meta.json sha256=a3d237ceb8e9</sub>
