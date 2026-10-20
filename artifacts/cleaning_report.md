# Cleaning report

This data cleaning process prepared the orders dataset by filtering for successfully delivered orders within a specific purchase window and validating timestamps, payments, items, reviews, and geography. As a result, the dataset was refined from 99,441 input rows down to 96,203 output rows for reliable business analysis.

## Rules applied (computed)

| rule | description | rows before | rows after | removed |
|---|---|---|---|---|
| R1_keep_delivered | keep orders with status 'delivered' (lateness is only defined for delivered orders) | 99,441 | 96,478 | 2,963 |
| R2_require_delivery_date | drop 'delivered' orders that have no customer delivery date | 96,478 | 96,470 | 8 |
| R3_purchase_window | keep purchases from 2017-01-01 to 2018-08-31 (2016 and Sep/Oct 2018 are sparse edges) | 96,470 | 96,203 | 267 |
| R4_timestamp_sequences | drop orders delivered before purchase; null carrier timestamps that are out of sequence (post-purchase fields, never used as features) | 96,203 | 96,203 | 0 |
| R5_items | aggregate items to one row per order (counts, price, freight, weight, volume, primary item); drop orders without items | 96,203 | 96,203 | 0 |
| R6_payments | drop payment rows typed 'not_defined' or with value <= 0, aggregate per order (type of the largest payment); drop orders left without a valid payment | 96,203 | 96,203 | 0 |
| R7_reviews | keep the latest review per order (score only; free text is never loaded into clean data) | 96,203 | 96,203 | 0 |
| R8_geography | join customer and primary-seller geography; distance between zip-prefix centroids (null when a prefix is unknown); city names dropped | 96,203 | 96,203 | 0 |
| R9_derive | derive purchase time parts, promised_days and delivery_days (calendar days) and is_late = delivery_days > promised_days | 96,203 | 96,203 | 0 |
| R10_finalize | fixed column order and dtypes, stable sort by (purchase_ts, order_id) | 96,203 | 96,203 | 0 |

## Issues found

- 2,963 rows removed by status consisting of 2 approved, 625 canceled, 5 created, 314 invoiced, 301 processing, 1,107 shipped, and 609 unavailable orders
- 8 rows removed due to missing delivery dates
- 267 rows removed falling outside the purchase window of 2017-01-01 to 2018-08-31
- 165 timestamps with carrier dates occurring before purchase dates
- 19 timestamps with carrier dates occurring after delivery dates
- 1,357 orders with unknown category
- 16 orders with unknown dimensions
- 2 untranslated categories kept: pc_gamer and portateis_cozinha_e_preparadores_de_alimentos
- 4 invalid payment rows dropped
- 523 orders with multiple reviews
- 643 orders without review
- 481 instances of unknown distance in geography
- 6,531 late orders identified representing a late rate of 6.8%
- Payment reconciliation showed 145 orders paid less than price plus freight, 288 orders paid more than price plus freight, and 95,770 orders within 1 cent

## Decisions

- R1_keep_delivered applied to retain only delivered orders, removing 2,963 rows because non-delivered statuses lack complete fulfillment metrics
- R2_require_delivery_date applied to drop 8 rows missing delivery dates to ensure delivery duration metrics can be calculated
- R3_purchase_window applied to remove 267 rows outside the 2017-01-01 to 2018-08-31 window to focus analysis on the stable period
- R4_timestamp_sequences applied to null out 165 carrier dates occurring before purchase and 19 carrier dates occurring after delivery without dropping rows
- R5_items applied to keep 1,357 orders with unknown category, 16 orders with unknown dimensions, and 2 untranslated categories without removing rows
- R6_payments applied to drop 4 invalid payment rows while preserving the rest
- R7_reviews retained 523 orders with multiple reviews and 643 orders without review without dropping rows
- R8_geography retained 481 rows with unknown distance without dropping rows
- R9_derive calculated 6,531 late orders and a late rate of 6.8%
- R10_finalize completed the dataset structure with 38 columns

## Payment reconciliation (computed)

| check | orders |
|---|---|
| payment_total within 1 cent of price + freight | 95,770 |
| paid more (installment interest) | 288 |
| paid less (vouchers) | 145 |

## Definitions

- **aov**: GMV divided by the number of orders
- **gmv**: sum of item prices in BRL, excluding freight
- **is_late**: delivery_days > promised_days, both counted in calendar days from the purchase date
- **payment_total**: final charged amount; differs from price + freight by vouchers and interest

## Caveats

- Some orders still contain unknown categories or unknown dimensions
- Certain categories like pc_gamer and portateis_cozinha_e_preparadores_de_alimentos remain untranslated
- Orders with multiple reviews or missing reviews are still present in the dataset
- Geography data retains 481 instances of unknown distance
- Payment totals may differ from price plus freight due to vouchers and interest, with 145 orders paid less and 288 orders paid more

---
<sub>narrative_source: llm | grounding: cleaning_report.json sha256=4cebaa7c7148</sub>
