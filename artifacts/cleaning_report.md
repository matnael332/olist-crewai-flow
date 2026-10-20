# Cleaning report

The nine raw Olist tables (99,441 orders) were reduced to 96,203 delivered orders, one row per order, purchased between 2017-01-01 and 2018-08-31. Items, payments and reviews were aggregated per order before joining so revenue is never double-counted.

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

- 2,963 orders were not delivered (approved 2, canceled 625, created 5, invoiced 314, processing 301, shipped 1,107, unavailable 609).
- 8 orders were marked delivered but had no delivery date.
- 267 delivered orders fell outside the 2017-01..2018-08 window.
- 165 carrier hand-overs were dated before purchase and 19 after delivery.
- 1,357 orders have a product without a category; untranslated categories: pc_gamer, portateis_cozinha_e_preparadores_de_alimentos.
- 4 payment rows were typed 'not_defined' or had a non-positive value.
- 523 orders had more than one review.
- 481 orders have a zip prefix without coordinates.

## Decisions

- Kept only delivered orders: lateness is undefined for orders that never arrived.
- Nulled out-of-sequence carrier timestamps instead of dropping orders; these post-purchase fields are never used as features.
- Took category and seller from the first order item; a missing category becomes 'unknown' and untranslated categories keep their Portuguese name.
- Used the type of the largest payment as the order's payment type and summed payments.
- Kept the latest review per order; review text is never loaded.
- Approximated customer-seller distance from median zip-prefix coordinates.

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

- The population is delivered orders only, so delivery failures are not represented.
- payment_total differs from price + freight for 433 orders (vouchers and installment interest).
- Distances use zip-prefix centroids, an approximation of the real addresses.

---
<sub>narrative_source: template | grounding: cleaning_report.json sha256=4cebaa7c7148</sub>
