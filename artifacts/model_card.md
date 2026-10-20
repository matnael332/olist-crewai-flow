# Model card: late-delivery risk at checkout

## Model details

- **Model:** `logistic_regression` (LogisticRegression): LogisticRegression on scaled numerics + one-hot categories (threshold tuned for the imbalance).
- **Selected by:** highest mean CV PR-AUC (4 decimals) among non-baseline models; the lean logistic regression wins when within 2% of the best.
- **Decision threshold:** 0.055 (tuned on out-of-fold F1).
- **Reproducibility:** seed 42, 2 threads, scikit-learn 1.7.2, Python 3.13.13.
- **Contract:** dataset_contract v1.0, sha256 `c330762f13cd68bef9031fdaf4891a3faa3aecbf854de3eb912d8181ccbdaf99`.

## Model purpose

The purpose of this model is to predict the likelihood of delivery delays for orders using a logistic regression pipeline, enabling proactive identification and intervention for high-risk shipments.

- **Intended users:** Intended users include supply chain managers, logistics coordinators, and operational risk teams responsible for monitoring order fulfillment and managing delivery exceptions.
- **Out of scope:**
  - Denying service, changing prices or penalizing customers or sellers.
  - Orders outside Brazil or from other marketplaces without revalidation.
  - Predicting the exact delivery date (this is a yes/no lateness risk).
  - This model must never be used for automated punitive actions against sellers or third-party logistics providers without human review.
  - The model is not intended for financial credit scoring or pricing decisions.
  - It must not be used outside the geographic and temporal distribution of the training data without re-validation.

## Training data summary

| fact | value |
|---|---|
| training rows | 70,851 (2017-01-01 .. 2018-04-30) |
| test rows | 19,001 (2018-05-01 .. 2018-07-31) |
| late rate train / test | 7.6% / 3.8% |
| features | 25 checkout-time features |
| dataset contract | v1.0 sha256 c330762f13cd |
| raw data | Olist GitHub commit d9e49802f3e9 |
| privacy | pseudonymous ids; no names, cities or review text |

The model was trained on 70851 rows spanning from 2017-01-01 to 2018-04-30 with a train prevalence of 7.6%, validated across a TimeSeriesSplit(n_splits=3) scheme, tested on 19001 rows from 2018-05-01 to 2018-07-31 with a test prevalence of 3.8%, and evaluated against a censored buffer of 6351 rows from 2018-08-01 to 2018-08-31 with a prevalence of 6.2%.

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
- The model exhibits a modest precision of 5.8% at the operational threshold of 0.055, leading to false positives.
- The F1 score remains low at 0.1085 due to severe class imbalance and low prevalence.
- The late rate differs significantly between the train prevalence of 7.6% and test prevalence of 3.8%, creating distribution shift challenges.

## Ethical considerations

- Regional bias: customers in North and Northeast states have higher late rates; interventions must improve their service, never deprioritize them.
- Seller fairness: scores are for operational prioritization only, not seller penalties.
- Privacy: identifiers are pseudonymous, but zip prefix plus timestamps are quasi-identifiers; only aggregates are shown in dashboards.
- License: the Olist data is shared for non-commercial use (CC BY-NC-SA 4.0 listing); this model is coursework, not a production system.
- Portuguese review text is never used, avoiding sensitive free-text content.
- Geographic features such as customer_state and seller_state are top predictors, which could potentially introduce regional biases in service monitoring.
- False positives resulting from the modest precision of 5.8% may unfairly burden specific fulfillment routes or regional sellers with unwarranted scrutiny.
- Shifting prevalences between training and test windows require careful oversight to prevent systematic performance degradation for underrepresented regions or periods.

## Monitoring

- Track the monthly late rate, PR-AUC and calibration, overall and by state.
- Re-tune the threshold and retrain when the late rate shifts.
- Alert when input distributions (promised days, freight, states) drift.
- Monitor model performance continuously to account for the shifting late rates observed between train prevalence of 7.6%, test prevalence of 3.8%, and censored buffer prevalence of 6.2%.
- Track precision, recall, and false positive rates over time at the operational threshold of 0.055 to ensure stability against distribution drift.
- Audit predictions regularly across different customer states and seller states to verify fairness and mitigate regional bias.

## How to use

Run the API (`uv run olist-api`) and POST checkout records:

```json
{"records": [{"n_items": 1, "total_price": 129.9, "total_freight": 21.5, "customer_state": "BA", "seller_state": "SP", "primary_category": "furniture_decor", "payment_type": "boleto", "max_installments": 1, "purchase_ts": "2018-06-12 14:03:00", "estimated_delivery_date": "2018-07-05 00:00:00"}]}
```

---
<sub>narrative_source: template+llm | grounding: metrics.json sha256=bb25900b6ecc, model_meta.json sha256=fb91a8fbfee5</sub>
