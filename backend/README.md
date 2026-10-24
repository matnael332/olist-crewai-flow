# Backend

Python backend of the Olist late-delivery workflow: the data and ML library, the CrewAI Flow with its
two crews, and the Flask prediction service. It writes everything the frontend shows into the
repository's `artifacts/` folder.

| Package | What it is |
|---|---|
| `pulse_core/` | pure library, no CrewAI: download, cleaning (R1-R10), two-layer EDA, dataset contract, the three validation gates, checkout-time features, model training and comparison, reports, promotion, manifest, `paths.repo_root()` |
| `pulse_flow/` | the CrewAI layer: `flow.py` (Flow with routers, gates and graceful failure), `crews/` (Data Analyst crew, 6 agents; Data Scientist crew, 5 agents; YAML config), `tools/`, `events.py` (LLM call ledger), `run.py` (CLI) |
| `pulse_api/` | Flask app (`/health`, `POST /predict`, `POST /predict/batch`, `GET /risk-list`; contract in `docs/api.md`): validation against the dataset contract, the risk policy, batch scoring and the daily replay; `examples/` holds request bodies |
| `pulse_copilot/` | the Ops copilot's brain: read-only tools over the artifacts, the Gemini loop (`agent.py`), the grounding check and instant answers without an LLM (`quick.py`) |
| `tests/` | unit, gate, Flow, crew (fake LLM), API and reproducibility tests, plus the `olist_mini` fixture |

Run everything from the repository root:

```bash
uv run kickoff                                    # full pipeline, LLM mode (needs GEMINI_API_KEY)
uv run python -m pulse_flow.run --no-llm          # deterministic mode
uv run python -m pulse_core.validate artifacts    # the three gates on the committed artifacts
uv run pulse-api                                  # prediction API on http://127.0.0.1:8000
uv run pytest backend/tests                       # backend tests, no LLM calls
```
