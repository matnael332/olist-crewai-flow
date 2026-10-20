# Business insights: what happened

Two layers: **all orders** (every status, business volume) and **delivered orders** (the clean population where lateness and reviews are measured).

## Key numbers (computed)

| layer | metric | value |
|---|---|---|
| All orders | orders (every status) | 99,441 |
| All orders | never delivered | 3.0% |
| All orders | peak month | 2017-11 (7,544) |
| All orders | busiest day | 2017-11-24 (1,176) |
| Delivered | orders | 96,203 |
| Delivered | customers (repeat share) | 93,096 (3.0%) |
| Delivered | GMV / AOV | BRL 13.18M / BRL 137.00 |
| Delivered | late rate | 6.8% (6,531 orders) |
| Delivered | median delivery / promised days | 10 / 24 |
| Delivered | 1-2 star reviews: late vs on time | 62.4% vs 9.2% |

## Headline

Orders grew from 800 in January 2017 to 7,544 in 2017-11; among delivered orders 6.8% arrived late, and late orders got 1-2 star reviews 62% of the time versus 9% when on time.

## Key findings

### 1. Fast growth with a Black Friday spike

- **Evidence:** All orders: 800 in 2017-01, peak month 2017-11 with 7,544, about 6,749 per month in 2018; busiest day 2017-11-24 with 1,176 orders vs 499 on the next busiest day.
- **What it means:** Capacity and carrier contracts must scale ahead of November.

### 2. Late deliveries destroy satisfaction

- **Evidence:** 1-2 star reviews: 62.4% of late orders vs 9.2% of on-time orders; mean score 2.27 vs 4.29.
- **What it means:** Preventing or announcing a delay protects ratings and retention.

### 3. Promises are generous, yet lateness persists

- **Evidence:** Median promised 24 days vs median actual 10 days; still 6.8% late (6,531 orders).
- **What it means:** Lateness is concentrated in specific orders, so it is predictable.

### 4. Lateness depends on geography

- **Evidence:** Highest late rates: AL 21.5%, MA 17.5%, PI 13.9%; lowest: MG 4.6%, SP 4.5%, PR 4.1% (states with at least 0.4% of delivered orders).
- **What it means:** Regional logistics, not only sellers, drive delays.

### 5. Lateness is not stable over time

- **Evidence:** Monthly late rate ranged from 1.2% (2018-06) to 19.0% (2018-03).
- **What it means:** Thresholds and models need recent data and periodic re-tuning.

### 6. Revenue is concentrated in a few categories

- **Evidence:** Top category health_beauty generated BRL 1.23M of BRL 13.18M delivered GMV; AOV BRL 137.00.
- **What it means:** Delivery improvements in top categories protect most revenue.

## Recommendations

- Score every order for late-delivery risk at checkout and proactively notify or reprioritize high-risk orders.
- Prioritize logistics improvements for the states with the highest late rates.
- Plan carrier capacity ahead of the November peak.
- Track the monthly late rate among delivered orders as an operational KPI.

## Data caveats

- Late rates are measured among delivered orders only; 3.0% of all orders never reached 'delivered' status.
- The data snapshot ends 2018-10-17; the last purchase months are incomplete.
- 99.3% of delivered orders have a review.

## Implications for modeling

- The target is imbalanced (6.8% positive): prefer PR-AUC and precision at the top of the ranking over accuracy.
- The late rate drifts over time, so evaluate on a later time window, never a random split.
- Only checkout-time information may be used: delivery dates and reviews are leakage.
- Customer state, promised days and freight are promising predictors.

## Figures

- `figures/monthly_orders_all_vs_delivered.png`
- `figures/late_rate_by_month.png`
- `figures/late_rate_by_state.png`
- `figures/delivery_days_hist.png`
- `figures/review_score_vs_late.png`
- `figures/gmv_by_category_top10.png`
- `figures/promised_vs_actual_by_month.png`

---
<sub>narrative_source: template | grounding: eda_summary.json sha256=ab5021a4aeb1</sub>
