# Cleaning report

This data cleaning process standardized the orders dataset by filtering out non-delivered statuses, missing delivery dates, and purchases outside the specified window, reducing total rows from 99,441 to 96,203. It also handled timestamp anomalies, product category and dimension gaps, invalid payments, multiple or missing reviews, and unknown geographic distances. Finally, derived metrics such as late deliveries were calculated to finalize a clean analytical table with 38 columns.

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

- 2,963 rows removed by status, consisting of 2 approved, 625 canceled, 5 created, 314 invoiced, 301 processing, 1,107 shipped, and 609 unavailable orders
- 8 rows removed due to missing delivery dates
- 267 rows removed falling outside the purchase window of 2017-01-01 to 2018-08-31
- Timestamp anomalies including 19 carrier dates after delivery and 165 carrier dates before purchase
- 1,357 orders with unknown category and 16 orders with unknown dimensions
- 2 untranslated categories kept: pc_gamer and portateis_cozinha_e_preparadores_de_alimentos
- 4 invalid payment rows dropped
- Review anomalies including 523 orders with multiple reviews and 643 orders without review
- 481 instances of unknown geographic distance
- 6,531 late orders representing a late rate of 6.8%

## Decisions

- R1_keep_delivered applied to retain only successfully delivered orders, removing 2,963 rows
- R2_require_delivery_date applied to ensure completeness by removing 8 rows lacking delivery dates
- R3_purchase_window applied to restrict data between 2017-01-01 and 2018-08-31, removing 267 rows
- R4_timestamp_sequences applied to null out 19 carrier dates occurring after delivery and 165 carrier dates occurring before purchase without dropping rows
- R5_items applied to flag 1,357 orders with unknown category and 16 with unknown dimensions while keeping untranslated categories pc_gamer and portateis_cozinha_e_preparadores_de_alimentos
- R6_payments applied to drop 4 invalid payment rows
- R7_reviews applied to account for 523 orders with multiple reviews and 643 orders without review
- R8_geography applied to mark 481 geographic distances as unknown
- R9_derive applied to calculate 6,531 late orders resulting in a late rate of 6.8%
- R10_finalize applied to produce the final dataset containing 38 columns

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

- 2,963 orders were excluded due to not being delivered
- 8 orders were removed due to missing delivery dates
- 267 orders were excluded for falling outside the 2017-01-01 to 2018-08-31 purchase window
- 19 carrier timestamps after delivery and 165 carrier timestamps before purchase were nulled out
- 1,357 orders contain unknown categories and 16 orders have unknown dimensions
- Two categories remain untranslated: pc_gamer and portateis_cozinha_e_preparadores_de_alimentos
- 4 invalid payment rows were dropped
- 523 orders contain multiple reviews and 643 orders have no review
- 481 geographic distances are unknown
- 6,531 orders are classified as late, resulting in a late rate of 6.8%

---
<sub>narrative_source: llm | grounding: cleaning_report.json sha256=4cebaa7c7148</sub>
