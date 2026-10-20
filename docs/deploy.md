# Deployment

## Streamlit Community Cloud (dashboard)

Live app: https://olist-late-delivery-1.streamlit.app, deployed from `main`. Community Cloud
redeploys it on every push to `main`.

The dashboard reads the committed `artifacts/` only, so the cloud app needs no API key and never
installs CrewAI. `app/requirements.txt` sits next to the entrypoint, so Community Cloud uses it
instead of the root `uv.lock`.

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repository.
2. Create app, then "Deploy a public app from GitHub".
3. Repository `matnael332/olist-crewai-flow`, branch `main`, main file path `app/streamlit_app.py`.
4. Advanced settings: Python version **3.13** (the default may differ; 3.14 is not supported by the
   pinned pyarrow). No secrets are needed. Optional: `API_URL` if the Flask API is deployed.
5. Deploy and open every page. The app sleeps after 12 hours without visits; open it before a demo.

## Flask API (optional)

Locally: `uv run python -m api.serve` (http://127.0.0.1:8000). On a host that sets `PORT` (e.g.
Railway), the server binds `0.0.0.0:$PORT`. Build with `pip install -r app/requirements.txt flask
waitress`, start with `python -m api.serve`, health check `/health`.

## Regenerating the artifacts

```bash
uv run python scripts/download_data.py
uv run kickoff                      # LLM mode (GEMINI_API_KEY in .env); promotes to artifacts/
uv run python -m olist_core.validate artifacts
```
