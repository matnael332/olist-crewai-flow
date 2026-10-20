## What

## Why

## How verified
<!-- Commands you ran and the key lines of their output. -->

## Checklist
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
- [ ] `uv run pytest` is green locally
- [ ] Spec file names unchanged: clean_data.csv, eda_report.html, insights.md, dataset_contract.json, features.csv, model.joblib, evaluation_report.md, model_card.md
- [ ] Artifacts were produced by the pipeline (never hand-edited) and committed only at a milestone end
- [ ] No secrets: `uv run python scripts/check_repo_hygiene.py`
- [ ] Docs updated where behaviour changed
