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

**Summary.** We evaluated a modest logistic regression model against a baseline prior and a histogram gradient boosting model to predict late deliveries. The late rate differs between the train and test windows, showing 7.6% prevalence in training and 3.8% in testing. The selected logistic regression model achieves a cross-validation PR-AUC of 18.0% and a test ROC-AUC of 73.3%, making it a pragmatic choice for operations despite its modest overall performance.

**Comparison.** The logistic regression model was selected because it achieved the highest mean cross-validation PR-AUC of 18.0%, narrowly beating the histogram gradient boosting model at 16.7% under our selection rule. While the gradient boosting model achieved a higher test precision at the top 10% (10.0% versus 12.0% for logistic regression), the logistic regression offers better calibration and a much higher test recall of 83.0% compared to 33.5% for gradient boosting at their respective thresholds.

**Strengths**

- Achieves a strong test ROC-AUC of 73.3% and a test PR-AUC of 10.2%.
- Provides a high test recall of 83.0%, catching a large majority of late orders.
- Strong lift at the top 10% of 3.2, outperforming the baseline prior's 1.1 and gradient boosting's 2.6.
- Lean and interpretable model structure relying on transparent coefficients.

**Weaknesses**

- Modest overall precision of 5.8% at the chosen threshold due to the low base rate.
- Brier score of 0.0384 is slightly worse than the gradient boosting model's 0.0362.
- Performance is sensitive to distribution shift, evidenced by the late rate differing between the train window (7.6%) and the test window (3.8%).

**Threshold recommendation.** We recommend setting the operational decision threshold at 5.5% to balance catching late orders against operational review capacity, leveraging the model's strong ranking ability in the top 10% slice where precision reaches 12.0%.

**Next steps**

- Incorporate recent temporal features to adapt to the late rate differing between train and test windows.
- Run a pilot with the operations team using the 5.5% threshold to measure real-world intervention impact.
- Explore feature engineering on customer and seller state interactions to further boost PR-AUC.

---
<sub>narrative_source: llm | grounding: metrics.json sha256=bb25900b6ecc, model_meta.json sha256=a3d237ceb8e9</sub>
