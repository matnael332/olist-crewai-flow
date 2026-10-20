# Demo video script

Target length 4:30, hard limit 5:00. Every pipeline run in the video is recorded in advance and
sped up in editing, so nothing on screen waits for an LLM call. Slide numbers refer to
`docs/presentation.pdf`. The video is narrated in Hebrew from
[video_narration_he.md](video_narration_he.md); the **Say** lines below are the English version.

## Before recording

1. Run `uv sync` and `uv run python scripts/download_data.py` once, so the real data is present.
2. Record the long terminal takes first (segments 4, 7 and 8) and keep the raw clips.
3. Use `--no-promote` for every run you record, so the committed `artifacts/` stay unchanged.
4. Open the live dashboard at https://olist-late-delivery-1.streamlit.app a few minutes before
   the dashboard take (a sleeping app takes a minute to wake) and click through every page once.
   Start the API (`uv run python -m api.serve`) for segment 6.
5. Never show `.env`, environment variables or the Google AI Studio page. The logs redact keys,
   but the screen does not.
6. Terminal: dark theme, font size 18 or larger, window maximised. Turn off notifications.
7. Recorder: OBS Studio or the Xbox Game Bar (`Win+Alt+R`), 1920x1080. Edit and speed up in
   Clipchamp or any editor.

## Segments

### 1. The problem (0:00-0:25)

**Show:** slide 2.

**Say:** "Olist is a Brazilian marketplace. When an order arrives late, 62 percent of customers leave
a one or two star review, against 9 percent when it arrives on time. Our product predicts, at
checkout, which orders are likely to be late, so operations can act before the customer is let
down."

### 2. The architecture (0:25-0:55)

**Show:** slide 4.

**Say:** "A CrewAI Flow runs two crews. The Data Analyst crew, six agents, cleans the data, explores
it and publishes a dataset contract. The Flow validates the contract against the cleaned data and
hands it to the Data Scientist crew, five agents, which builds features, trains and compares
models, and writes the model card. Three gates sit between the stages. If any gate fails, the run
stops and nothing is published."

### 3. The repository (0:55-1:25)

**Show:** the GitHub repository page, then the README spec compliance map, then the closed pull
requests list with green checks, then `src/olist_flow/crews/data_analyst/config/agents.yaml`.

**Say:** "Everything lives in one GitHub repository. Each milestone was one pull request, and CI runs
the tests, validates the committed artifacts and checks reproducibility on every pull request. The
agents and tasks are defined in YAML, and the README maps every requirement of the brief to the
file that implements it."

### 4. A full run (1:25-2:15)

**Show:** the pre-recorded terminal take of

```bash
uv run python -m olist_flow.run --run-id demo-video --no-promote
```

sped up about 4x. Pause or zoom on these lines as they appear:

- `STAGE run_analyst_stage START` and `CREW data_analyst ...`
- the `GATE gate_contract ... PASS` lines and `GATE gate_contract RESULT PASS (21/21 checks)`
- `HANDOFF analyst->scientist contract_version=1.0 contract_sha256=... rows=96203 required=14 forbidden=7`
- `GATE gate_required_features RESULT PASS (13/13 checks)`
- `GATE gate_artifacts RESULT PASS (9/9 checks)` and the final `run demo-video: success (exit 0)`

**Say:** "This is one run on the full dataset, sped up. First the analyst stage runs tested Python,
then the crew inspects the results and writes the narrative. The contract gate runs 21 checks. Here
is the handoff: the contract version and hash travel to the scientist crew. Then the feature gate
confirms every required feature exists before any model is trained. About twelve Gemini calls in
total, and every artifact is written to the run folder."

### 5. The dashboard (2:15-3:05)

**Show:** the live app at https://olist-late-delivery-1.streamlit.app. Overview (KPIs and the gate
table), Data and contract (scroll the contract columns), EDA and insights (the late rate by
state), Model (metrics table, then the precision-recall chart).

**Say:** "The dashboard reads only the committed artifacts. The overview shows the run status and
all three gates. Here is the dataset contract the scientists had to follow. The analysis shows
lateness concentrated in the North and Northeast and drifting between months, which is why we
test on later months. Logistic regression beat gradient boosting: on the test window the riskiest
ten percent of orders are late 12.1 percent of the time, about three times the base rate."

### 6. Scoring an order (3:05-3:30)

**Show:** the Predict page with its default order (a Bahia customer, a Sao Paulo seller,
furniture), then the same order with customer state SP. Then two API calls in a terminal (use
`curl` instead of `curl.exe` outside Windows):

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@api/examples/order.json"
curl.exe -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@api/examples/invalid_order.json"
```

The first call returns late probabilities of about 0.164 and 0.038. The second returns HTTP 422
with `'XX' is not an allowed value`.

**Say:** "The same order is about four times as risky when it ships across the country: 16 percent
for a customer in Bahia against 4 percent within Sao Paulo, so the page flags it as elevated risk.
The Flask API uses the same feature
code as training, and it validates every request against the dataset contract, so an unknown state
is rejected with a clear error."

### 7. Graceful failure (3:30-4:05)

**Show:** the pre-recorded take of

```bash
uv run python -m olist_flow.run --no-llm --data-dir tests/fixtures/olist_mini --no-promote --inject-fault missing_feature
git status --short artifacts/
```

Zoom on `FAULT INJECTED missing_feature`, the two `FAIL` lines, `RESULT FAIL (11/13 checks)`,
`RUN ... FAILED at gate_required_features`, `failed (exit 1)` and the empty `git status` output.

**Say:** "Now we break it on purpose. This run deletes a required feature after feature
engineering. The feature gate catches it before any model is trained, the Flow routes to the
failure step, writes a status file and exits with code 1. Nothing is promoted: the published
artifacts are untouched."

### 8. Reproducibility (4:05-4:25)

**Show:** the pre-recorded take of

```bash
uv run python -m olist_flow.run verify-repro --data-dir tests/fixtures/olist_mini
```

ending on the table and `ALL DETERMINISTIC ARTIFACTS IDENTICAL`.

**Say:** "Two fresh runs with different hash seeds produce byte-identical data, features, metrics and
figures. CI repeats this check on every pull request."

### 9. Wrap-up (4:25-4:40)

**Show:** slide 11, then slide 12.

**Say:** "The model card states the purpose and the limits: it prioritises orders for proactive
help and is never used to deny service, and regional bias is monitored. Next steps are seller
history features and a delivery-time regression. Thank you."

## After recording

1. Export at 1080p and check the length is under 5:00.
2. Upload as an unlisted YouTube video or a shared Google Drive file.
3. Add the link to the "Presentation and video" row of the README.
