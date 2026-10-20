# Demo video script

Target length 4:45, hard limit 5:00. Every pipeline run in the video is recorded in advance and
sped up in editing, and the few seconds the copilot waits for Gemini are cut, so nothing on screen
waits for an LLM call. Slide numbers refer to `docs/presentation.pdf`. The video is narrated in
Hebrew from [video_narration_he.md](video_narration_he.md); the **Say** lines below are the English
version.

## Before recording

1. Run `uv sync` and `uv run python scripts/download_data.py` once, so the real data is present.
2. Record the long terminal takes first (segments 4, 8 and 9) and keep the raw clips.
3. Use `--no-promote` for every run you record, so the committed `artifacts/` stay unchanged.
4. Open the live dashboard at https://olist-late-delivery-1.streamlit.app a few minutes before
   the dashboard take (a sleeping app takes a minute to wake) and click through every page once.
5. The copilot answers free questions only after the `GEMINI_API_KEY` and `COPILOT_PASSCODE`
   secrets are set ([deploy.md](deploy.md), step 5). Open **Ops copilot** and enter the passcode
   before the take, so the chat is already unlocked. Each session allows 25 messages and the take
   uses three; reload the page between rehearsals to start a new session.
6. Start the API (`uv run olist-api`) for segment 7.
7. Never show `.env`, environment variables, the Streamlit secrets page, the passcode or the
   Google AI Studio page. The logs redact keys, but the screen does not.
8. Terminal: dark theme, font size 18 or larger, window maximised. Turn off notifications.
9. Recorder: OBS Studio or the Xbox Game Bar (`Win+Alt+R`), 1920x1080. Edit and speed up in
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

**Say:** "A CrewAI Flow runs two crews. The Data Analyst crew, six agents, cleans and explores the
data and publishes a dataset contract. The Data Scientist crew, five agents, builds features,
trains and compares models, and writes the model card. Three gates sit between the stages; if one
fails, nothing is published. The product reads only what was published: an ops copilot, the
dashboard and an API."

### 3. The repository (0:55-1:15)

**Show:** the GitHub repository page, then the README spec compliance map, then the closed pull
requests list with green checks, then `backend/olist_flow/crews/data_analyst/config/agents.yaml`.

**Say:** "Everything lives in one GitHub repository. Each milestone was a pull request, and CI runs
the tests, validates the artifacts and checks reproducibility on each one. Agents and tasks are
defined in YAML, and the README maps every requirement to its file."

### 4. A full run (1:15-1:55)

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

**Say:** "This is one run on the full dataset, sped up. The analyst stage runs tested Python, then
the crew inspects the results and writes the narrative. The contract gate runs 21 checks. Here is
the handoff: the contract version and hash travel to the scientist crew. The feature gate confirms
every required feature exists before any model is trained. About twelve Gemini calls in total."

### 5. The dashboard (1:55-2:30)

**Show:** the live app at https://olist-late-delivery-1.streamlit.app. Overview (the KPI cards and
the four pipeline steps with their gate badges), Insights (hover the late-rate-by-state chart),
Model (the comparison, then the precision-recall curves).

**Say:** "The dashboard reads only the committed artifacts. The overview shows the run status and
all three gates. The analysis shows lateness concentrated in the North and Northeast and drifting
between months, which is why we test on later months. Logistic regression won: on the test window
the riskiest ten percent of orders are late 12.1 percent of the time, about three times the base
rate."

### 6. The Ops copilot (2:30-3:25)

**Show:** the **Ops copilot** page of the live app (first in the sidebar, under Operate), already
unlocked.

1. Click the **How did the last run go?** quick reply. The instant answer reads "Run final-llm-6
   finished with status success in 2 min 49 s (llm mode). Gates: gate_contract PASS 21/21,
   gate_required_features PASS 13/13, gate_artifacts PASS 9/9. The crews made 12 LLM calls (19,964
   tokens)." Point at the Sources line and "instant answer, no LLM".
2. Click **Replay 2018-07-16**: "238 orders were purchased on 2018-07-16: 42 high, 142 elevated and
   54 normal risk. In reality 5 of 42 (11.9%) high-risk orders arrived late, against 10 of 238
   (4.2%) for the whole day." Scroll over the card and the ranked table with its "Actually late"
   column.
3. Click **Action list with message drafts (CSV)** and show the downloaded file for two seconds
   (the `recommended_action` and `customer_message` columns).
4. Type "Why are orders to Bahia riskier than orders within Sao Paulo?" and press Enter; cut the
   "Thinking..." wait. Gemini's answer quotes the state late rates (Bahia 12.2% of 3,253 delivered
   orders, Sao Paulo 4.5%) and ends with a Sources line. The wording varies between runs; the
   numbers come from the tools.

**Say:** "The product is for the team that operates these agents. Instead of digging through logs,
they ask the Ops copilot. 'How did the last run go?' is answered instantly from the run files:
success, all three gates passed. Now a morning check on a real past day: the copilot scores all 238
orders of July 16, 2018. Forty-two are high risk, and almost twelve percent of them really arrived
late, against four percent for the whole day. The action list adds a message draft for every
high-risk customer. Free questions go to Gemini, which answers only from read-only tools, so every
number shows its source."

### 7. The API (3:25-3:40)

**Show:** two API calls in a terminal (use `curl` instead of `curl.exe` outside Windows):

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@backend/olist_api/examples/order.json"
curl.exe -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@backend/olist_api/examples/invalid_order.json"
```

The first call returns late probabilities 0.1635 (`"risk_level": "elevated"`, with its recommended
action) and 0.0378 (`"normal"`). The second returns HTTP 422 with `'XX' is not an allowed value`.
[docs/api.md](api.md) also documents `/predict/batch` and `/risk-list`.

**Say:** "Other systems get the same answers from the Flask API: the same features and risk policy,
and every request is checked against the dataset contract, so an unknown state gets a clear error."

### 8. Graceful failure (3:40-4:10)

**Show:** the pre-recorded take of

```bash
uv run python -m olist_flow.run --no-llm --data-dir backend/tests/fixtures/olist_mini --no-promote --inject-fault missing_feature
git status --short artifacts/
```

Zoom on `FAULT INJECTED missing_feature`, the two `FAIL` lines, `RESULT FAIL (11/13 checks)`,
`RUN ... FAILED at gate_required_features`, `failed (exit 1)` and the empty `git status` output.

**Say:** "Now we break it on purpose. This run deletes a required feature after feature
engineering. The feature gate catches it before any model is trained, the Flow routes to the
failure step, writes a status file and exits with code 1. Nothing is promoted: the published
artifacts are untouched."

### 9. Reproducibility (4:10-4:25)

**Show:** the pre-recorded take of

```bash
uv run python -m olist_flow.run verify-repro --data-dir backend/tests/fixtures/olist_mini
```

ending on the table and `ALL DETERMINISTIC ARTIFACTS IDENTICAL`.

**Say:** "Two fresh runs with different hash seeds produce byte-identical data, features, metrics and
figures. CI repeats this check on every pull request."

### 10. Wrap-up (4:25-4:45)

**Show:** slide 11, then slide 12.

**Say:** "The model card states the purpose and the limits: it prioritises orders for proactive
help and is never used to deny service, and regional bias is monitored. Next, the same copilot
core can run a Telegram bot and a daily digest. Thank you."

## After recording

1. Export at 1080p and check the length is under 5:00.
2. Upload as an unlisted YouTube video or a shared Google Drive file.
3. Add the link to the "Presentation and video" row of the README.
