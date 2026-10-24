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
artifacts (Python 3.13, a crewai-free `frontend/requirements.txt`). Running CrewAI there is unrealistic
(memory, build time, API key exposure); the Flow runs locally and its artifacts are committed.

**ADR-10. Deferred.** Delivery-days regression, seller-history features, and a Railway deployment of
the API are documented as future work to protect the core scope.

**ADR-11. Narrators copy numbers, they never convert them.** A review of the LLM narratives found
one rounding slip (9.3% written for 0.092474), an invented threshold and an invented "BRL 0.00"
budget. Tool outputs now format every rate-like field as a percentage string before the narrators
see it, the prompts forbid new numbers, thresholds and costs, and the model card prompt asks only
for model-specific points; `merge_items` drops additions shorter than five words. After the change
every percentage in the committed narratives matches a computed value.

**ADR-12. Monorepo with `backend/` and `frontend/`.** The backend (`pulse_core`, `pulse_flow`,
`pulse_api`) produces the artifacts; the frontend (Streamlit) only reads them, so `artifacts/` is the
contract between the two. One uv project at the root keeps every command `uv run ...` from the root,
and `frontend/requirements.txt` (exported from the `frontend` dependency group) keeps CrewAI out of
the cloud deployment. The dashboard scores orders in-process by importing `pulse_api.model`, the
same code the Flask API serves, so there is no second service to host; setting `API_URL` switches it
to HTTP. Modules locate the repository with `pulse_core.paths.repo_root()` instead of counting
parent folders, so files can move without breaking paths. The `api` package was renamed
`pulse_api` to avoid a generic top-level import name.

**ADR-13. The product is an Ops copilot for the agent operator.** The user is the data team that
operates the agents. They work in a WhatsApp-style chat page inside the existing dashboard (no new
account or channel), backed by Gemini with eleven read-only tools (`backend/pulse_copilot`). The
copilot reads runs, gates, metrics, insights, documents and the contract, and scores orders; it
never runs the Flow (CrewAI is not installed in the cloud app, and a run changes artifacts).
Numbers must come from tools: rates reach the model pre-formatted (ADR-11), the Sources line is
built from the recorded tool calls, and unmatched percentages are flagged. Quick replies answer
from templates without any LLM, which is cheaper, instant, exact and keeps the chat useful when
Gemini is unavailable. The chat sits behind a passcode with 25 messages per session, because it
spends the owner's Gemini quota; the key and passcode come from environment variables (Streamlit
Cloud secrets). Rejected for the MVP: a Telegram or WhatsApp bot (new accounts, a process to host;
the same `pulse_copilot` core can serve one later) and running the Flow from chat.

**ADR-14. One risk policy for every channel.** `backend/pulse_api/risk.py` turns a probability into
high (the riskiest 10%, p >= 0.1698), elevated (p >= the 0.055 decision threshold) or normal, with
one recommended action per level and a template customer message for high-risk orders. The API,
the batch scorer, the copilot and the dashboard all call it, so the same order never gets two
labels. Batches report counts per level, not a sum of probabilities: the model over-predicts on
recent months (trained at a 7.6% late rate, tested at 3.8%).
