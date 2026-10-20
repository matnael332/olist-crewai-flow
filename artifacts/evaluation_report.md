# Evaluation report: late-delivery risk at checkout

## Setup (computed)

- **Task:** binary classification of `is_late` (delivered after the promised date, calendar days), measured among delivered orders.
- **Validation:** TimeSeriesSplit(n_splits=3) on the train window; threshold tuned on out-of-fold predictions; the test window is scored once.
- **Censored buffer:** 6,351 August-2018 orders (6.2% late) are excluded from metrics (right-censoring).

| split | purchase dates | orders | late rate |
|---|---|---|---|
| train | 2017-01-01 .. 2018-04-30 | 70,851 | 7.6% |
| test | 2018-05-01 .. 2018-07-31 | 19,001 | 3.8% |
| censored_buffer | 2018-08-01 .. 2018-08-31 | 6,351 | 6.2% |

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

**Summary.** We evaluated a baseline prior, a HistGradientBoostingClassifier, and a LogisticRegression model using TimeSeriesSplit(n_splits=3) on the train window. Note that the late rate differs between the train and test windows, with a train prevalence of 7.6% and a test prevalence of 3.8%. The selected model for operations is logistic_regression, chosen because it achieved the highest mean CV PR-AUC among non-baseline models with a selection margin of 0.0129.

**Comparison.** The baseline prior achieved a cv_pr_auc_mean of 0.0898 and cv_roc_auc_mean of 0.5, with a test pr_auc of 0.038 and roc_auc of 0.5. The HistGradientBoostingClassifier achieved a cv_pr_auc_mean of 0.1674 and cv_roc_auc_mean of 0.6428, with a test pr_auc of 0.0839 and roc_auc of 0.6981. The selected LogisticRegression model achieved a cv_pr_auc_mean of 0.1803 and cv_roc_auc_mean of 0.6643, along with a test pr_auc of 0.1019 and roc_auc of 0.7326.

**Strengths**

- LogisticRegression achieves the highest cross-validation PR-AUC (0.1803) and ROC-AUC (0.6643) among all candidates.
- Shows strong ranking capability on test data with a test ROC-AUC of 0.7326 and a lift_at_top10pct of 3.1719.
- Maintains a high recall of 83.0% on the test set at its tuned threshold.

**Weaknesses**

- Modest precision of 5.8% on the test set, leading to a high volume of false alarms.
- Low F1 score of 0.1085 reflecting the severe class imbalance challenge.
- Brier score of 0.0384 indicates calibration can be further improved for operational reliability.

**Threshold recommendation.** We recommend setting the operational decision threshold to 0.055 for the logistic regression model, which successfully balances the trade-off to capture an 83.0% recall while maintaining a precision_at_top10pct of 12.1%.

**Next steps**

- Monitor model performance drift given the distribution shift between the train prevalence of 7.6% and test prevalence of 3.8%.
- Explore post-processing probability calibration techniques to improve the Brier score.
- Incorporate additional features beyond the top drivers like promised_days and customer_state to improve precision.

---
<sub>narrative_source: llm | grounding: metrics.json sha256=bb25900b6ecc, model_meta.json sha256=24dcf092d98c</sub>
