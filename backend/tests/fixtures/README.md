# Test fixture: `olist_mini`

A deterministic **real subset** of the Olist dataset with a short list of injected defects, used by
the unit tests, the no-LLM Flow tests and CI. Rebuild it from the full data with:

```bash
uv run python scripts/download_data.py
uv run python backend/tests/fixtures/make_fixture.py
```

**Selection.** For every purchase month, the first 120 order ids ordered by `sha1(order_id)`, plus
every row that references them (items, payments, reviews, customers, products, sellers, and at most 3
geolocation rows per referenced zip prefix). Raw strings are copied unchanged.

**Injected defects** (targets are recorded in `olist_mini/injections.json`):

| Defect | Expected handling |
|---|---|
| Delivery date set one day before purchase | dropped by R4 |
| Delivered order with an empty delivery date | dropped by R2 |
| Carrier hand-over two hours before purchase | carrier timestamp nulled by R4 |
| All items of an order removed | dropped by R5 |
| The order's only payment typed `not_defined` | dropped by R6 |
| Extra payment row with value 0 | row dropped by R6, order kept |
| Second, later review with score 1 | R7 keeps the later review (score 1) |
| Product category blanked | primary category becomes `unknown` |
| Product category set to `pc_gamer` (no translation) | Portuguese slug kept |
| Exact duplicate geolocation row | dropped by the centroid build |
| Geolocation point in Madrid | dropped as outside Brazil |

**License.** Derived from the Olist dataset (GitHub `olist/work-at-olist-data`, MIT; Kaggle listing
CC BY-NC-SA 4.0). See `data/NOTICE.md`.
