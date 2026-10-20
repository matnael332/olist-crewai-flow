# Decision records

**ADR-1. Dataset: Olist instead of UCI Online Retail II.** Two independent dataset reviews ranked
Olist first. It covers sales, customers, products, sellers, payments and reviews in nine linked
tables, downloads without credentials from Olist's own GitHub at a pinned commit, and gives the
analyst crew real but bounded cleaning work. Online Retail II would have produced a 45-71 MB clean
file and a three-minute xlsx parse, with no product categories.

**ADR-2. Run, then narrate.** Deterministic Python produces every artifact; agents inspect and
explain. LLM output is not reproducible (Gemini 3.6+ ignores temperature; CrewAI's Gemini provider
drops `seed`), so reproducibility must come from code, and gates must not depend on whether an agent
decided to call a tool.

**ADR-3. Crew 2 is one crew run in two phases.** The spec requires the Flow to confirm that the
required features exist before modeling. Splitting the crew's tasks into `feature_phase()` and
`modeling_phase()` lets a Flow router gate sit between them while the crew stays one CrewAI class
with five agents; `crew()` defines the full crew.

**ADR-4. Tool-less narrator agents.** In CrewAI 1.14.5 a task inherits its agent's tools, and with
tools present Gemini receives a `structured_output` pseudo-tool from the first turn, which can
pre-empt or discard the real tool call. Every `output_pydantic` task therefore runs on an agent
without tools; tool tasks return their JSON verbatim (`result_as_answer`).

**ADR-5. Model: gemini-3.5-flash-lite.** Stable, free tier, native tool calling (the `-latest`
aliases lose it; 2.0 models are shut down). One LLM instance per agent, HTTP retries for
408/429/5xx, `max_rpm=5`. About 12 calls and 20k tokens per full run.

**ADR-6. Target and split.** `is_late` = delivered later than the promised date in calendar days,
among delivered orders (6.8%). Time split: train 2017-01..2018-04, test 2018-05..2018-07; August 2018
is a censored buffer (the snapshot ends 2018-10-17, so very late August orders may be missing).
Only checkout-time columns may become features; the contract lists the forbidden ones.

**ADR-7. Model selection.** Candidates: prior baseline, logistic regression, histogram gradient
boosting. Selection by cross-validated PR-AUC (time-series folds) with a lean-model rule. Logistic
regression won outright (CV PR-AUC 0.180 vs 0.167). `class_weight="balanced"` was removed because it
inflated probabilities (an average order scored about 0.55); the imbalance is handled by tuning the
threshold on out-of-fold predictions.

**ADR-8. Promotion by directory swap.** Windows refuses to replace a file another process holds
open (e.g. the dashboard). Staging the new set and swapping folders with retries and rollback
guarantees `artifacts/` is never a mix of two runs.

**ADR-9. Deployment.** Streamlit Community Cloud hosts a read-only dashboard over the committed
artifacts (Python 3.13, a crewai-free `app/requirements.txt`). Running CrewAI there is unrealistic
(memory, build time, API key exposure); the Flow runs locally and its artifacts are committed.

**ADR-10. Deferred.** Delivery-days regression, seller-history features, and a Railway deployment of
the API are documented as future work to protect the core scope.
