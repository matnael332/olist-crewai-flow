# Evaluation report: late-delivery risk at checkout

## Setup (computed)

- **Task:** binary classification of `is_late` (delivered after the promised date, calendar days), measured among delivered orders.
- **Validation:** TimeSeriesSplit(n_splits=3) on the train window; threshold tuned on out-of-fold predictions; the test window is scored once.
- **Censored buffer:** 6,351 August-2018 orders (6.2% late) are excluded from metrics (right-censoring).

| split | purchase dates | orders | late rate |
|---|---|---|---|
| censored_buffer | 2018-08-01 .. 2018-08-31 | 6,351 | 6.2% |
| test | 2018-05-01 .. 2018-07-31 | 19,001 | 3.8% |
| train | 2017-01-01 .. 2018-04-30 | 70,851 | 7.6% |

## Cross-validation on the train window (computed)

| model | ROC-AUC (mean ± std) | PR-AUC (mean ± std) |
|---|---|---|
| baseline_prior | 0.500 ± 0.000 | 0.090 ± 0.032 |
| hist_gradient_boosting | 0.643 ± 0.046 | 0.167 ± 0.072 |
| logistic_regression | 0.664 ± 0.046 | 0.180 ± 0.083 |

Folds for the selected model (logistic_regression); note the shifting late rate:

| fold | train rows | validation rows | late rate | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 1 | 17,715 | 17,712 | 5.4% | 0.603 | 0.086 |
| 2 | 35,427 | 17,712 | 8.3% | 0.678 | 0.166 |
| 3 | 53,139 | 17,712 | 13.2% | 0.712 | 0.289 |

## Test window results (computed)

| model | ROC-AUC | PR-AUC | precision top 10% | lift top 10% | Brier |
|---|---|---|---|---|---|
| baseline_prior | 0.500 | 0.038 | 4.3% | 1.12x | 0.038 |
| hist_gradient_boosting | 0.698 | 0.084 | 10.0% | 2.63x | 0.036 |
| logistic_regression | 0.733 | 0.102 | 12.1% | 3.17x | 0.038 |

## Model selection

- **Rule:** highest mean CV PR-AUC (4 decimals) among non-baseline models; the lean logistic regression wins when within 2% of the best.
- **CV PR-AUC:** hist_gradient_boosting 0.1674, logistic_regression 0.1803
- **Selected:** `logistic_regression` (margin 0.0129 PR-AUC over the runner-up).

## Operating point of the selected model (test window)

| measure | value |
|---|---|
| threshold (tuned on out-of-fold F1) | 0.055 |
| orders flagged | 54.3% |
| precision | 5.8% |
| recall | 83.0% |
| F1 | 0.109 |
| precision in riskiest 10% | 12.1% |

## Most important features (permutation importance, test window)

| rank | feature | drop in PR-AUC |
|---|---|---|
| 1 | promised_days | 0.0291 |
| 2 | customer_state | 0.0144 |
| 3 | same_state | 0.0126 |
| 4 | distance_km | 0.0097 |
| 5 | seller_state | 0.0041 |
| 6 | total_volume_cm3 | 0.0027 |
| 7 | purchase_month_num | 0.0026 |
| 8 | total_price | 0.0012 |
| 9 | total_weight_g | 0.0006 |
| 10 | n_sellers | 0.0004 |

## Figures

- `figures/roc_curves.png`
- `figures/pr_curves.png`
- `figures/calibration_selected.png`
- `figures/feature_importance_selected.png`
- `figures/late_rate_by_month.png` (from the EDA): the late rate drifts over time

## Limitations of this evaluation

- Prevalence shift: 7.6% late in train vs 3.8% in test, so absolute precision and the threshold do not transfer automatically.
- Delivered orders only; orders that never arrived are not scored.
- One test window (May-July 2018); results may differ in other seasons.

## Reviewer commentary

**Summary.** The selected model (logistic_regression) ranks late orders well above chance: the riskiest 10% of test orders are late 12.1% of the time, 3.2x the 3.8% test late rate.

**Comparison.** logistic_regression reached mean CV PR-AUC 0.180 versus 0.167 for hist_gradient_boosting; on the test window its ROC-AUC is 0.733 and PR-AUC 0.102 against a 0.038 baseline.

**Strengths**

- Evaluated on later months than it was trained on (time-aware split).
- Uses only checkout-time information; strongest signals: promised_days, customer_state, same_state.
- Lean, fast and fully reproducible pipeline (fixed seed and thread count).

**Weaknesses**

- Modest absolute precision: at threshold 0.055 precision is 5.8% with recall 83.0%.
- The late rate drifts: 7.6% in train vs 3.8% in test.
- Delivered orders only: orders that never arrived are not represented.

**Threshold recommendation.** Use the score to rank orders and act on the riskiest 10% (precision 12.1%); re-tune the 0.055 threshold on the most recent months before any automated action.

**Next steps**

- Add seller history features (past late rate, as of the purchase time).
- Retrain monthly and monitor the late rate and calibration by state.
- Model delivery days (regression) to set better delivery promises.

---
<sub>narrative_source: template | grounding: metrics.json sha256=bb25900b6ecc, model_meta.json sha256=40c72d3d8a96</sub>
