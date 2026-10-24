"""Quick replies answered without any LLM: the copilot's offline mode.

The same tools the Gemini agent uses, with fixed sentence templates. They keep the chat useful
without a key or when the free quota runs out, and they make the main scenario testable.
"""

from __future__ import annotations

from pulse_copilot.answer import Answer, Conversation, Recorder
from pulse_copilot.tools import REPLAY_DAY, build_tools
from pulse_copilot.workspace import Workspace

ACTIONS = ("last_run", "gates", "replay", "model", "attachment")


def _tools(ws: Workspace, conversation: Conversation, recorder: Recorder) -> dict:
    return {f.__name__: recorder.wrap(f) for f in build_tools(ws, conversation)}


def quick_answer(action: str, ws: Workspace, conversation: Conversation) -> Answer:
    recorder = Recorder()
    tools = _tools(ws, conversation, recorder)
    mark = len(conversation.outputs)
    if action == "last_run":
        r = tools["get_run_summary"]()
        gates = ", ".join(f"{name} {result}" for name, result in r.get("gates", {}).items())
        text = (
            f"Run {r['run_id']} finished with status {r['status']} in {r['duration']} "
            f"({r['mode']} mode). Gates: {gates}. "
        )
        if r.get("llm_calls"):
            text += f"The crews made {r['llm_calls']} LLM calls ({r['total_tokens']} tokens)."
    elif action == "gates":
        r = tools["get_run_summary"]()
        text = "; ".join(f"{name}: {result}" for name, result in r.get("gates", {}).items()) + "."
    elif action == "replay":
        r = tools["replay_day"](day=REPLAY_DAY)
        if "error" in r:
            text = f"Replay failed: {r.get('detail', r['error'])}."
        else:
            levels, real = r["orders_by_risk_level"], r["what_really_happened"]
            text = (
                f"{r['orders_scored']} orders were purchased on {r['date']}: {levels['high']} "
                f"high, {levels['elevated']} elevated and {levels['normal']} normal risk. In "
                f"reality {real['high_risk_orders_actually_late']} high-risk orders arrived late, "
                f"against {real['all_orders_actually_late']} for the whole day. The ranked list "
                "and the action list are below."
            )
    elif action == "model":
        r = tools["get_model_metrics"]()
        chosen = r["candidates"][r["selected_model"]]
        others = {
            k: v for k, v in r["candidates"].items() if k not in {r["selected_model"], "baseline"}
        }
        other_name, other = next(iter(others.items()))
        text = (
            f"{r['selected_model'].capitalize()} was selected: cross-validated PR-AUC "
            f"{chosen['cv_pr_auc']} against {other['cv_pr_auc']} for {other_name}. On the test "
            f"window its ROC-AUC is {chosen['test_roc_auc']}, and the riskiest 10% of orders are "
            f"late {chosen['late_rate_in_riskiest_10pct']} of the time "
            f"({chosen['lift_in_riskiest_10pct']} the base rate)."
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
