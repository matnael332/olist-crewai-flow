# Frontend

Streamlit dashboard for the Olist late-delivery workflow, live at
https://olist-late-delivery-1.streamlit.app. It reads the committed `artifacts/` only and never
imports CrewAI; order scoring reuses the backend's `olist_api.model` in-process (set `API_URL` to
call a deployed API instead).

| Section | Page | What it answers |
|---|---|---|
| Business | Overview | the question, four headline numbers, how the Flow works, what the data says |
| Business | Insights | what happened: interactive charts with one-line takeaways (Data Analyst crew) |
| Business | Risk scoring | score an order at checkout, with presets and what-if scenarios |
| Data science | Model | how good the model is: comparison, ROC/PR curves, calibration, drivers, threshold |
| Data science | Data contract | the handoff: cleaning funnel, schema explorer, allowed vs forbidden features |
| Engineering | Pipeline runs | stages, the three gates, LLM usage and the log of every run |

| Path | What it is |
|---|---|
| `streamlit_app.py` | entrypoint: page config, logo, sectioned navigation (Community Cloud main file) |
| `views/` | one module per page |
| `components/` | the UI kit: `layout` (headers, chart cards), `cards` (KPIs, badges), `charts` (Altair, one palette), `pipeline` (the stepper), `format` (numbers) |
| `services/` | `artifacts` (cached, mtime-keyed loaders and derived data), `scoring` (presets, risk levels, what-if) |
| `assets/` | logo and icon |
| `_bootstrap.py` | puts `backend/` on `sys.path` so the scoring code can be imported |
| `tests/` | AppTest pages, scoring service and chart-spec tests, against the committed artifacts |
| `requirements.txt` | slim deploy dependencies, exported from the `frontend` group of `uv.lock` |

The theme (Inter, slate and blue palette, dark sidebar) lives in `.streamlit/config.toml` at the
repository root, where both `streamlit run` and Community Cloud read it.

```bash
uv run streamlit run frontend/streamlit_app.py   # http://localhost:8501
uv run pytest frontend/tests
```

Deployment: see `docs/deploy.md`.
