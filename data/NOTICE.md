# Data notice

This project uses the **Brazilian E-Commerce Public Dataset by Olist**.

- Source used by the pipeline: GitHub repository `olist/work-at-olist-data`, pinned commit
  `d9e49802f3e92d09ee94ab9ccc5e457f207a8959`, folder `datasets/`. The per-file sha256 hashes are
  recorded in `data/raw/manifest.json`.
- The same data is listed on Kaggle as `olistbr/brazilian-ecommerce` under **CC BY-NC-SA 4.0**
  (attribution, non-commercial, share-alike). This project is non-commercial coursework and
  attributes Olist; derived data in this repository (`artifacts/`, `data/lookups/`,
  `tests/fixtures/olist_mini/`) is shared under the same terms.
- Raw files are not committed; `scripts/download_data.py` re-downloads and verifies them.

The GitHub repository is distributed under the following license, reproduced as required:

```
MIT License

Copyright (c) 2019 @olist

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Privacy.** Identifiers (`order_id`, `customer_unique_id`, product and seller ids) are Olist's
public pseudonymous hashes. City names and review text are dropped. Zip prefixes plus timestamps are
quasi-identifiers, so the dashboard only shows aggregates.
