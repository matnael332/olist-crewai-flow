"""Quick replies answered without any LLM: the copilot's offline mode.

The same tools the LLM agent uses, with fixed sentence templates. They keep the chat useful
without a key or when the quota runs out, and they make the main scenario testable.
"""

from __future__ import annotations

from pulse_copilot.answer import Answer, Conversation, Recorder
from pulse_copilot.tools import build_tools
from pulse_copilot.workspace import Workspace

ACTIONS = ("last_run", "gates", "model", "contract", "attachment")


def _tools(ws: Workspace, conversation: Conversation, recorder: Recorder) -> dict:
    return {f.__name__: recorder.wrap(f) for f in build_tools(ws, conversation)}


def quick_answer(action: str, ws: Workspace, conversation: Conversation) -> Answer:
    recorder = Recorder()
    tools = _tools(ws, conversation, recorder)
    mark = len(conversation.outputs)
    if action == "last_run":
        r = tools["get_run_summary"]()
        if "error" in r:
            text = f"No run found: {r['error']}."
        else:
            gates = ", ".join(f"{name} {result}" for name, result in r.get("gates", {}).items())
            text = (
                f"Run {r['run_id']} finished with status {r['status']} in {r['duration']} "
                f"({r['mode']} mode). Gates: {gates}. "
            )
            if r.get("llm_calls"):
                text += f"The crews made {r['llm_calls']} LLM calls ({r['total_tokens']} tokens)."
    elif action == "gates":
        r = tools["get_run_summary"]()
        gates = r.get("gates", {})
        text = (
            "; ".join(f"{name}: {result}" for name, result in gates.items()) + "."
            if gates
            else "No gate results are available."
        )
    elif action == "model":
        r = tools["get_model_metrics"]()
        if "error" in r:
            text = f"No model is available: {r['error']}."
        else:
            chosen = r["candidates"][r["selected_model"]]
            others = {
                k: v
                for k, v in r["candidates"].items()
                if k not in {r["selected_model"], "baseline"}
            }
            other_name, other = next(iter(others.items()))
            text = (
                f"{r['selected_model'].capitalize()} was selected: cross-validated PR-AUC "
                f"{chosen['cv_pr_auc']} against {other['cv_pr_auc']} for {other_name}. On the test "
                f"window its ROC-AUC is {chosen['test_roc_auc']}, and the riskiest 10% of rows "
                f"are positive {chosen['positive_rate_in_riskiest_10pct']} of the time "
                f"({chosen['lift_in_riskiest_10pct']} the base rate)."
            )
    elif action == "contract":
        r = tools["get_contract"]()
        if "error" in r:
            text = f"No contract is available: {r['error']}."
        else:
            text = (
                f"Contract v{r['contract_version']}: {r['rows']} rows and {r['columns']} columns. "
                f"Target: {r['target']} Required features: {', '.join(r['required_features'])}. "
                f"Forbidden: {', '.join(r['forbidden_features'])}."
            )
    elif action == "attachment":
        r = tools["score_attached_file"]()
        if "error" in r:
            text = f"I could not score a file: {r['error']}."
        else:
            s = r["summary"]
            text = (
                f"{r['file']}: {s['scored']} of {s['rows']} rows scored, {s['high']} high, "
                f"{s['elevated']} elevated and {s['normal']} normal risk. "
                f"{s['rejected']} rows failed validation."
            )
    else:
        raise ValueError(f"unknown quick action {action!r}")
    return Answer(
        text=text,
        sources=recorder.sources(),
        tool_calls=recorder.calls,
        outputs=conversation.outputs[mark:],
        offline=True,
    )
