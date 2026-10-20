# Olist Late-Delivery AI Workflow (CrewAI Flow)

An industry-style AI product workflow built for the "Industry-Simulated AI Product Workflow" final project.
A **CrewAI Flow** hands off from a **Data Analyst crew** (ingest, clean, EDA, insights, dataset contract)
to a **Data Scientist crew** (features, model comparison, evaluation, model card), with deterministic
validation gates, graceful failure and reproducible artifacts.

**Business question:** which orders on the Olist marketplace will be delivered late, predicted at checkout?

> Status: under construction. The full README (spec compliance map, verify-in-10-minutes guide,
> architecture and data license) lands with milestone M4b.

## Quickstart (Windows PowerShell or bash)

```bash
uv sync                      # creates .venv with Python 3.13 and all dependencies
cp .env.example .env         # then set GEMINI_API_KEY
uv run pytest                # unit tests (no LLM calls)
```
