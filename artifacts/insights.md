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

Olist achieved a total business volume of 99,441 orders, peaking in November 2017 with 7,544 orders, while among delivered orders the overall late rate stood at 6.8% with a bad review rate of 12.8%.

## Key findings

### 1. Business Growth and Volume Peak

- **Evidence:** Total orders reached 99,441 across the dataset, with 2017-11 recording the peak month at 7,544 orders and 2017-11-24 marking the peak day at 1,176 orders.
- **What it means:** The platform experienced massive seasonal demand spikes, particularly during late 2017 shopping events.

### 2. Delivery Performance Among Delivered Orders

- **Evidence:** Among delivered orders (96,203 total), the overall late rate was 6.8% across 6,531 late orders, with a mean delivery time of 12.4768 days and a median of 10.0 days.
- **What it means:** The majority of shipments arrive on time or ahead of the median promised days of 24.0 days, though operational bottlenecks occur during peak periods.

### 3. Geographic Disparities in Late Rates

- **Evidence:** State-level performance among delivered orders showed high late rates in AL at 21.5% (396 orders), MA at 17.5% (713 orders), and PI at 13.9% (475 orders), compared to lower late rates in SP at 4.5% (40,399 orders) and MG at 4.6% (11,319 orders).
- **What it means:** Logistical infrastructure heavily favors southeastern states like São Paulo and Minas Gerais, while northern and northeastern regions suffer from severe delivery delays.

### 4. Customer Satisfaction and Review Impact

- **Evidence:** Among delivered orders, the bad review rate was 12.8%, which spiked to 62.4% for late orders compared to 9.2% for on-time orders, alongside a mean score of 2.2706 for late deliveries versus 4.2912 for on-time deliveries.
- **What it means:** Late deliveries are the single most damaging driver of customer dissatisfaction, drastically dragging down review scores.

### 5. Payment Preferences and GMV Distribution

- **Evidence:** Credit cards dominated payment types at 75.5% with 51.5% of orders utilizing more than one installment, generating a total GMV of BRL 13,179,777.95 across 93,096 customers and 2,936 sellers.
- **What it means:** Installment payment flexibility is crucial for customer purchasing behavior, supporting an average order value of BRL 136.9997.

## Recommendations

- Investigate logistics and carrier networks in high late-rate states such as Alagoas and Maranhão to reduce delivery friction.
- Improve inventory and warehouse planning ahead of seasonal peaks like November to mitigate spikes in delivery delays.
- Enhance predictive delivery estimates for customers in remote regions to better manage expectations and protect review scores.
- Encourage seller performance standards to minimize non-delivered and late orders which severely degrade customer satisfaction.

## Data caveats

- The dataset contains 99,441 total orders, but delivery metrics and reviews are strictly evaluated on the 96,203 delivered orders population.
- Non-delivered orders account for 3.0% of total volume and are excluded from late-rate calculations.

## Implications for modeling

- Machine learning models predicting delivery lateness must account for heavy train split imbalances, where the train split late rate is 7.6% across 70,851 orders versus the test split late rate of 3.8% across 19,001 orders.
- Target variables should separate general order volume forecasts from delivered order classification tasks like lateness and bad reviews.

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
