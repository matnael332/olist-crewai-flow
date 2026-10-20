# Model card: late-delivery risk at checkout

## Model details

- **Model:** `logistic_regression` (LogisticRegression): LogisticRegression on scaled numerics + one-hot categories (threshold tuned for the imbalance).
- **Selected by:** highest mean CV PR-AUC (4 decimals) among non-baseline models; the lean logistic regression wins when within 2% of the best.
- **Decision threshold:** 0.055 (tuned on out-of-fold F1).
- **Reproducibility:** seed 42, 2 threads, scikit-learn 1.7.2, Python 3.13.13.
- **Contract:** dataset_contract v1.0, sha256 `e881be9f2bb0c17297a799ab0852c7015c99309fb3822ce9fa414b80c06591c7`.

## Model purpose

The purpose of this model is to predict late deliveries for logistics operations using a LogisticRegression classifier trained on scaled numeric features and one-hot encoded categories, with a decision threshold tuned specifically to handle class imbalance.

- **Intended users:** Intended users include operations teams, logistics planners, and supply chain managers who need to identify potential late orders in advance.
- **Out of scope:**
  - Denying service, changing prices or penalizing customers or sellers.
  - Orders outside Brazil or from other marketplaces without revalidation.
  - Predicting the exact delivery date (this is a yes/no lateness risk).
  - This model must never be used for automated punitive actions against sellers or third-party couriers without human review.
  - The model is out of scope for financial credit scoring or pricing decisions.
  - It should not be used as a standalone customer communication tool regarding delivery delays.

## Training data summary

| fact | value |
|---|---|
| training rows | 70,851 (2017-01-01 .. 2018-04-30) |
| test rows | 19,001 (2018-05-01 .. 2018-07-31) |
| late rate train / test | 7.6% / 3.8% |
| features | 25 checkout-time features |
| dataset contract | v1.0 sha256 e881be9f2bb0 |
| raw data | Olist GitHub commit d9e49802f3e9 |
| privacy | pseudonymous ids; no names, cities or review text |

The training dataset contains 70851 rows spanning from 2017-01-01 to 2018-04-30 with a late prevalence of 7.6%. The evaluation framework uses a TimeSeriesSplit with 3 splits on the train window, comparing against a test set of 19001 rows spanning from 2018-05-01 to 2018-07-31 with a test prevalence of 3.8%, plus a censored buffer set of 6351 rows from 2018-08-01 to 2018-08-31 with a prevalence of 6.2%.

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
- The model exhibits a modest precision of 5.8% on the test set, leading to a high volume of false alarms.
- The low F1 score of 0.1085 reflects the severe class imbalance challenge present in the data.
- The Brier score of 0.0384 indicates that probability calibration can be further improved for operational reliability.

## Ethical considerations

- Regional bias: customers in North and Northeast states have higher late rates; interventions must improve their service, never deprioritize them.
- Seller fairness: scores are for operational prioritization only, not seller penalties.
- Privacy: identifiers are pseudonymous, but zip prefix plus timestamps are quasi-identifiers; only aggregates are shown in dashboards.
- License: the Olist data is shared for non-commercial use (CC BY-NC-SA 4.0 listing); this model is coursework, not a production system.
- Portuguese review text is never used, avoiding sensitive free-text content.
- Geographic features such as customer_state and seller_state heavily influence predictions, requiring ongoing audits to ensure regional fairness and prevent systemic bias against specific locations.
- Reliance on delivery distance and state attributes may indirectly impact vulnerable regions with underfunded infrastructure, risking disparate service quality.

## Monitoring

- Track the monthly late rate, PR-AUC and calibration, overall and by state.
- Re-tune the threshold and retrain when the late rate shifts.
- Alert when input distributions (promised days, freight, states) drift.
- Monitor model performance drift continuously, given the distribution shift between the train prevalence of 7.6% and the test prevalence of 3.8%.
- Explore post-processing probability calibration techniques to improve the Brier score over time.
- Track top predictive features such as promised_days and customer_state to ensure feature stability and catch upstream data quality issues.

## How to use

Run the API (`uv run python -m api.serve`) and POST checkout records:

```json
{"records": [{"n_items": 1, "total_price": 129.9, "total_freight": 21.5, "customer_state": "BA", "seller_state": "SP", "primary_category": "furniture_decor", "payment_type": "boleto", "max_installments": 1, "purchase_ts": "2018-06-12 14:03:00", "estimated_delivery_date": "2018-07-05 00:00:00"}]}
```

---
<sub>narrative_source: template+llm | grounding: metrics.json sha256=bb25900b6ecc, model_meta.json sha256=24dcf092d98c</sub>
