# Deployment

## Streamlit Community Cloud (frontend)

Live app: https://olist-late-delivery-1.streamlit.app, deployed from `main`. Community Cloud
redeploys it on every push to `main`.

The dashboard reads the committed `artifacts/` only, so the cloud app needs no API key and never
installs CrewAI. `frontend/requirements.txt` sits next to the entrypoint, so Community Cloud uses it
instead of the root `uv.lock`. The theme comes from `.streamlit/config.toml` at the repository root.

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repository.
2. Create app, then "Deploy a public app from GitHub".
3. Repository `matnael332/pulse`, branch `main`, main file path
   `frontend/streamlit_app.py`, custom subdomain `olist-late-delivery-1`.
4. Advanced settings: Python version **3.13** (the default may differ; 3.14 is not supported by the
   pinned pyarrow). Optional: `API_URL` to score through a deployed API.
5. Secrets (app menu, Settings, Secrets) for the **Ops copilot**. Root-level secrets reach the app
   as environment variables:

   ```toml
   GEMINI_API_KEY = "your Google AI Studio key"
   COPILOT_PASSCODE = "a passcode you share with the people allowed to chat"
   # optional
   COPILOT_MAX_MESSAGES = "25"
   COPILOT_MODEL = "gemini-3.5-flash-lite"
   ```

   Without `GEMINI_API_KEY` the copilot runs in offline mode (quick replies only); without
   `COPILOT_PASSCODE` the chat is open to every visitor, so always set it on a public app.
6. Deploy and open every page. The app sleeps after 12 hours without visits; open it before a demo.

**Moving the entrypoint.** Community Cloud cannot change the main file path of a deployed app. Delete
the app first (app menu, Settings, Delete), then push the change, then redeploy with the new path.
The subdomain is released on deletion and can be reused.

## Flask API (backend, optional)

Locally: `uv run pulse-api` serves http://127.0.0.1:8000 (`/health`, `POST /predict`); example
request bodies are in `backend/pulse_api/examples/`. On a host that sets `PORT` (e.g. Railway or
Render) the server binds `0.0.0.0:$PORT`:

```bash
pip install -r frontend/requirements.txt flask waitress    # build
PYTHONPATH=backend python -m pulse_api.serve               # start; health check /health
```

Set `API_URL` on the Streamlit app to make the dashboard score through the deployed API instead of
in-process.

## Regenerating the artifacts

```bash
uv run python scripts/download_data.py
uv run kickoff                      # LLM mode (GEMINI_API_KEY in .env); promotes to artifacts/
uv run python -m pulse_core.validate artifacts
```
