# Cleaning report

This data cleaning process standardized the dataset from 99,441 rows down to 96,203 rows by focusing strictly on successfully delivered orders within the specified purchase window and removing incomplete records. These steps ensure that subsequent business analyses, revenue calculations, and delivery performance metrics are computed on reliable and verified transactions.

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

- 2963 rows removed due to non-delivered order statuses, broken down as: 2 approved, 625 canceled, 5 created, 314 invoiced, 301 processing, 1107 shipped, and 609 unavailable.
- 8 rows removed due to missing delivery dates.
- 267 rows removed due to falling outside the purchase window of 2017-01-01 to 2018-08-31.
- 184 timestamp sequence issues identified, including 165 carrier timestamps before purchase and 19 carrier timestamps after delivery.
- 1357 orders with unknown category and 16 orders with unknown dimensions.
- 4 invalid payment rows dropped.
- 1166 review anomalies noted, including 523 orders with multiple reviews and 643 orders without a review.
- 481 orders with unknown geographic distance.
- 6531 late orders identified, resulting in a late rate of 6.8%.

## Decisions

- Applied rule R1_keep_delivered to retain only delivered orders, removing 2963 rows with non-delivered statuses to ensure consistency in performance metrics.
- Applied rule R2_require_delivery_date to remove 8 orders lacking a delivery date, ensuring complete delivery lifecycle data.
- Applied rule R3_purchase_window to exclude 267 rows outside the 2017-01-01 to 2018-08-31 window, focusing the analysis on the target timeframe.
- Applied rule R4_timestamp_sequences to null out 165 carrier timestamps occurring before purchase and 19 carrier timestamps occurring after delivery without dropping rows.
- Applied rule R5_items to handle item-level data by retaining 1357 orders with unknown category, 16 orders with unknown dimensions, and keeping untranslated categories pc_gamer and portateis_cozinha_e_preparadores_de_alimentos.
- Applied rule R6_payments to drop 4 invalid payment rows while keeping 96203 reconciled payment records.
- Applied rule R7_reviews to retain review data despite 523 orders having multiple reviews and 643 orders lacking reviews.
- Applied rule R8_geography to retain 481 records where the distance was unknown.
- Applied rule R9_derive to calculate late orders, identifying 6531 late orders and a late rate of 6.8%.
- Applied rule R10_finalize to produce the final dataset containing 38 columns.

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

- 1357 orders still contain unknown categories and 16 orders have unknown dimensions.
- Some categories remain untranslated, specifically pc_gamer and portateis_cozinha_e_preparadores_de_alimentos.
- 523 orders contain multiple reviews and 643 orders have no review at all.
- 481 orders have an unknown geographic distance.
- Payment reconciliation shows 145 orders paid less than price plus freight, 288 orders paid more than price plus freight, and 95770 orders within 1 cent.

---
<sub>narrative_source: llm | grounding: cleaning_report.json sha256=4cebaa7c7148</sub>
