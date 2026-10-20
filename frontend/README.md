# Frontend

Streamlit dashboard for the Olist late-delivery workflow. It reads the committed `artifacts/` only
and never imports CrewAI; order scoring reuses the backend's `olist_api.model` in-process (set
`API_URL` to call a deployed API instead).

| Path | What it is |
|---|---|
| `streamlit_app.py` | entrypoint: page config, sidebar and navigation (Community Cloud main file) |
| `views/` | one module per page |
| `services/` | cached, mtime-keyed loaders for the artifacts |
| `_bootstrap.py` | puts `backend/` on `sys.path` so the scoring code can be imported |
| `tests/` | AppTest suite run against the committed artifacts |
| `requirements.txt` | slim deploy dependencies, exported from the `frontend` group of `uv.lock` |

```bash
uv run streamlit run frontend/streamlit_app.py   # http://localhost:8501
uv run pytest frontend/tests
```

Deployment: see `docs/deploy.md`.
