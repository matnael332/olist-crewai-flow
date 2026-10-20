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

Olist experienced significant growth culminating in a 2017-11 peak, but faced delivery and late-rate challenges concentrated in specific states.

## Key findings

### 1. Business Volume and Peak Growth

- **Evidence:** Total business volume reached 99,441 orders across all statuses, with a peak month in 2017-11 seeing 7,544 orders and a peak day on 2017-11-24 recording 1,176 orders.
- **What it means:** The platform scales successfully during promotional events, though such spikes strain operational capacity.

### 2. Delivered Population and Lateness

- **Evidence:** Among delivered orders, out of 96,203 orders, 6,531 were late, resulting in a late rate of 6.8%.
- **What it means:** A small yet impactful portion of orders misses delivery estimates, risking customer trust.

### 3. Geographic Disparities in Lateness

- **Evidence:** Among delivered orders, the highest late-rate states included AL with a late rate of 21.5% across 396 orders, and MA with 17.5% across 713 orders, whereas SP recorded a late rate of 4.5% across 40,399 orders.
- **What it means:** Logistical bottlenecks in remote or northern states heavily skew regional delivery performance.

### 4. Impact of Lateness on Reviews

- **Evidence:** Among delivered orders, the bad review rate for late orders was 62.4% with a mean score of 2.2706, compared to a bad review rate of 9.2% and a mean score of 4.2912 for on-time orders.
- **What it means:** Late deliveries are the single largest driver of customer dissatisfaction on the platform.

### 5. Payment Preferences and Installments

- **Evidence:** Among delivered orders, credit cards accounted for 75.5% of payment types, and 51.5% of transactions involved more than one installment with a median of 2.0 installments.
- **What it means:** Customers rely heavily on credit and installment plans to finance their purchases.

## Recommendations

- Investigate and remediate logistical bottlenecks in high late-rate states like AL and MA.
- Improve delivery time estimations during high-volume periods such as November to prevent spikes in late rates.
- Implement proactive customer communication when an order is flagged as at-risk of becoming late to mitigate bad reviews.
- Optimize carrier allocation for top GMV categories like health_beauty and bed_bath_table to maintain stable late rates.

## Data caveats

- Non-delivered orders account for 3.0% of all orders and are excluded from delivery performance and review metrics.
- Repeat customer share is low at 3.0%, meaning the dataset is heavily skewed toward one-time buyers.

## Implications for modeling

- Models predicting late delivery should account for strong seasonal spikes, particularly in months like 2017-11, 2018-02, and 2018-03.
- Geographic features (state-level variables) are crucial given the severe variance in late rates between states like SP and AL.
- Review score prediction models must heavily weight delivery status, as lateness massively increases the probability of a bad review.

## Figures

- `figures/monthly_orders_all_vs_delivered.png`
- `figures/late_rate_by_month.png`
- `figures/late_rate_by_state.png`
- `figures/delivery_days_hist.png`
- `figures/review_score_vs_late.png`
- `figures/gmv_by_category_top10.png`
- `figures/promised_vs_actual_by_month.png`

---
<sub>narrative_source: llm | grounding: eda_summary.json sha256=ab5021a4aeb1</sub>
