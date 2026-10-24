# pulse

A factory for trustworthy prediction models. A **CrewAI Flow** hands off from a **Data Analyst
crew** to a **Data Scientist crew** with deterministic validation gates in between, graceful
failure and byte-reproducible artifacts. The dataset is not part of the core: a `DatasetSpec`
supplies the raw schema, the cleaning rules, the column catalogue, the features, the target and
the split windows, and the pipeline turns it into a validated clean table, a dataset contract, a
model, metrics, a model card and an evaluation report.

Repository: https://github.com/matnael332/pulse

**Status.** The core is dataset-agnostic. `backend/tests/toy_spec.py` is a small seeded spec that exercises every gate, fault, renderer, the
API and the ops copilot end to end. The first real dataset spec is `workforce_saas` (a
time-attendance SaaS; ten data groups, one row per customer account and month, target: cancelled
within 60 days): `backend/pulse_specs/workforce_saas/` holds the schema (`SCHEMA.md`), a seeded
synthetic generator (`uv run synth`, `SYNTH.md`) and the spec (`spec.py`), and the pipeline runs
on it end to end in deterministic mode. The product is a hosted service: customers upload an
export and run the pipeline on our server, one folder per customer (`--data-dir`, `--runs-dir`,
`--artifacts-dir`, `--run-id`). One Docker image runs the pipeline, the dashboard and the API on
that server (`Dockerfile`, `compose.yaml`, `DEPLOY.md`). Next step: the site layer (login, upload,
run, dashboard).

## How it runs

```mermaid
flowchart LR
  raw[raw tables<br/>manifest + schema] --> A[run_analyst_stage<br/>spec.clean, spec.eda_summary, contract]
  A --> G1{gate_contract}
  G1 -- contract_ok --> H[HANDOFF<br/>contract digest]
  H --> B[run_feature_stage<br/>spec.derive_features]
  B --> G2{gate_required_features}
  G2 -- features_ok --> C[run_modeling_stage<br/>baseline, logistic regression, gradient boosting]
  C --> G3{gate_artifacts}
  G3 -- artifacts_ok --> P[promote_and_finish<br/>directory swap to artifacts/]
  G1 & G2 & G3 -- failed --> F[fail_gracefully<br/>status + report, exit 1, nothing promoted]
```

**Run-then-narrate.** Each stage first runs seeded Python that writes every artifact with template
narratives. Only then (in LLM mode) does a crew run: tool-bearing agents read the computed results
and tool-less narrator agents write structured narratives grounded in those numbers. Gates never
depend on LLM behaviour; `--no-llm` makes every artifact spec-complete without a model call.

**The seam.** Producers (`stages`, the contract builder, the feature builder, the Flow, the crew
runner) receive the `DatasetSpec`. Gates, training, fault injection and the raw loader read
everything they need from the artifacts: `feature_spec.json` carries the id, time and split
columns and the decision stage; `dataset_contract.json` carries the feature rules and the split
windows. See `backend/pulse_core/spec.py` for the full interface and the conventions a spec must
honour.

## Verify

```bash
uv sync                                                            # Python 3.13 + locked deps
uv run ruff check . && uv run ruff format --check .
uv run pytest                                                      # 96 tests: toy spec, generator, workforce_saas spec
uv run synth --data-dir data                                       # workforce_saas raw data: 200 accounts, ~145 MB, < 1 min
uv run python -m pulse_flow.run --no-llm --spec pulse_specs.workforce_saas.spec:SPEC --data-dir data   # full run, promotes to artifacts/
uv run python -m pulse_flow.run --no-llm --spec tests.toy_spec:SPEC --data-dir <toy data> --no-promote
uv run python -m pulse_flow.run --no-llm --no-promote               # no spec: graceful failure, exit 1
uv run streamlit run frontend/streamlit_app.py                     # dashboard (empty until a run is promoted)
uv run pulse-api                                                   # prediction API on http://127.0.0.1:8000
```

The same inside Docker, on the server volume (`DEPLOY.md` has the full guide; CI runs these):

```bash
docker compose build
docker compose run --rm job synth --data-dir /srv/pulse/customers/demo/data
docker compose run --rm job python -m pulse_flow.run --no-llm --spec pulse_specs.workforce_saas.spec:SPEC \
  --data-dir /srv/pulse/customers/demo/data --runs-dir /srv/pulse/customers/demo/runs \
  --artifacts-dir /srv/pulse/customers/demo/artifacts
docker compose up -d                                               # dashboard :8501, API :8000 (localhost only)
```

The toy data folder is written by `tests.toy_spec.write_raw(<dir>/raw)`; the test suite does this
in a temporary directory. `synth` writes `data/raw/` (13 tables, `raw_schema.json`,
`manifest.json`), `data/truth/` (the planted story, never read by the pipeline) and an empty
`data/lookups/`; `data/` is git-ignored. `--accounts`, `--seed`, `--start-month`, `--snapshot`
and `--dirt` change the dataset, `--force` overwrites it.

| Flag | Meaning |
|---|---|
| `--spec pkg.module:ATTR` | the dataset spec (default: the `PIPELINE_DATASET_SPEC` variable): a `DatasetSpec` or a factory |
| `--spec-params FILE` | JSON keyword arguments for a spec factory, one file per customer (e.g. `make_spec`'s `snapshot_date`) |
| `--no-llm` / `--mode` | deterministic or LLM mode (default: LLM when `GEMINI_API_KEY` is set) |
| `--data-dir` | folder with `raw/` (tables, `raw_schema.json`, `manifest.json`) and `lookups/` |
| `--inject-fault NAME` | `bad_input_hash`, `crew_error`, `drop_column`, `contract_mismatch`, `missing_feature`, `leaky_feature`, `missing_artifact` |
| `--no-promote` | keep the run in `runs/<id>/` only |
| `--from-run ID --start-at features\|modeling` | resume from a previous run's artifacts |
| `--compare-to DIR` | print the reproducibility table against another artifacts folder |

**Exit codes:** 0 success; 1 graceful failure (see `runs/<id>/run_status.json`); 2 unexpected
error (redacted traceback in `runs/<id>/traceback.txt`).

## Repository layout

```
backend/
  pulse_core/        generic pipeline library (no CrewAI): spec, splits, columns, raw, contract,
                     validators (3 gates), features, modeling, reporting, promotion, manifest
  pulse_flow/        CrewAI layer: flow.py, crews/ (YAML + classes), tools/, run.py (CLI)
  pulse_api/         Flask prediction API: record validation from feature_spec.json, risk policy,
                     batch scoring
  pulse_copilot/     Ops copilot: read-only tools, the Gemini loop, instant answers
  tests/             tests + toy_spec.py (the synthetic DatasetSpec they run on)
frontend/            Streamlit dashboard (reads artifacts/ only): copilot, model, contract, runs
scripts/             repository hygiene check
Dockerfile           the server image; compose.yaml runs it, DEPLOY.md explains the server
```

`artifacts/` is the contract between the backend and the frontend: it holds the last promoted
run. It is git-ignored, like `data/`: on the server both live in the customer's own folder. Code: MIT (`LICENSE`).
