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

Olist achieved a total business volume of 99,441 orders generating 13,179,777.95 BRL in delivered GMV, though logistics bottlenecks caused late deliveries in certain regions and months.

## Key findings

### 1. Business Volume and Growth

- **Evidence:** Total all_orders volume reached 99,441 across the dataset, starting at 800 orders in 2017-01 and averaging 6,748.9 monthly orders between 2018-01 and 2018-08.
- **What it means:** The platform experienced substantial early-stage growth before stabilizing into a steady high-volume monthly cadence through 2018.

### 2. Peak Seasonality

- **Evidence:** The peak month for business volume was 2017-11 with 7,544 orders, and the busiest single day was 2017-11-24 with 1,176 orders, followed by 2017-11-25 with 499 orders.
- **What it means:** Black Friday heavily concentrates demand, causing massive transaction spikes that test logistics and fulfillment capacity.

### 3. Overall Delivery Performance and Lateness

- **Evidence:** Among delivered orders (96,203 total), the overall late rate was 6.8%, representing 6,531 late orders out of 96,203, with a median delivery time of 10.0 days and median promised days of 24.0.
- **What it means:** While median delivery times are well within generous customer promises, operational hiccups still create noticeable pockets of delayed shipments.

### 4. Geographic Disparities in Lateness

- **Evidence:** Highest late rate states among delivered orders included AL at 21.5% (396 orders, median 22.0 days), MA at 17.5% (713 orders, median 19.0 days), PI at 13.9% (475 orders, median 16.0 days), CE at 13.8% (1,273 orders, median 18.0 days), and BA at 12.2% (3,253 orders, median 17.0 days). Lowest late rate states included MG at 4.6% (11,319 orders, median 10.0 days), SP at 4.5% (40,399 orders, median 7.0 days), and PR at 4.1% (4,903 orders, median 10.0 days).
- **What it means:** Fulfillment efficiency is highly geographically skewed, struggling significantly with remote northern and northeastern states compared to the southeast.

### 5. Customer Satisfaction Impact

- **Evidence:** Among delivered orders, the bad review rate was 12.8%, which spiked to 62.4% for late orders compared to just 9.3% for on-time orders, with mean scores of 2.3 for late deliveries versus 4.3 for on-time deliveries.
- **What it means:** Delivery delays severely damage customer sentiment, heavily driving down review scores and increasing customer dissatisfaction.

## Recommendations

- Investigate and renegotiate carrier performance in high-lateness states like AL, MA, PI, CE, and BA to reduce delivery times.
- Improve inventory and warehouse staffing ahead of November to safely handle Black Friday spikes without service degradation.
- Refine delivery estimations for regions prone to transit delays to set realistic customer expectations.
- Implement proactive customer communication protocols when shipments show early signs of falling behind schedule to mitigate bad reviews.

## Data caveats

- Non-delivered orders account for a 3.0% share of all orders and are excluded from delivery and review metrics.
- Review coverage sits at 99.3%, meaning a very small fraction of delivered orders lack review scores.
- Data splits vary in late rates, with train at 7.6%, test at 3.8%, and censored_buffer at 6.2%.

## Implications for modeling

- Models predicting delivery lateness must account for severe class imbalance and geographic feature importance, as state-level variance is extreme.
- Review score prediction models should heavily weight delivery status and actual delivery days as core predictive features given their strong correlation with bad reviews.

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
