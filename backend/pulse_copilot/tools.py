"""The copilot's tools: read what the agents produced, and score records. Read-only by design.

Each tool returns a JSON-friendly dict with a ``source`` list. Rates are pre-formatted as
percentage strings (ADR-11) so the model copies numbers instead of computing them. No
`from __future__ import annotations` here: google-genai reads the type hints at runtime.
"""

import re
from collections.abc import Callable

from pulse_api.batch import score_records
from pulse_copilot.answer import Conversation
from pulse_copilot.workspace import Workspace, duration_seconds, gate_summary, stage_durations
from pulse_core.reporting.fmt import as_percentages, pct

MAX_TEXT = 3000
TOP_N = 10
DOCUMENTS = {
    "model_card": "model_card.md",
    "evaluation_report": "evaluation_report.md",
    "insights": "insights.md",
    "cleaning_report": "cleaning_report.md",
}
GATES = {
    "contract": "gate_contract",
    "features": "gate_required_features",
    "required_features": "gate_required_features",
    "artifacts": "gate_artifacts",
}
MODEL_LABELS = {
    "logistic_regression": "logistic regression",
    "hist_gradient_boosting": "gradient boosting",
    "baseline_prior": "baseline",
}
GATE_ORDER = ("gate_contract", "gate_required_features", "gate_artifacts")  # pipeline order


def _top(predictions: list[dict]) -> list[dict]:
    return [
        {
            "id": p["id"] or f"row {p['row']}",
            "probability": pct(p["probability"]),
            "risk_level": p["risk_level"],
        }
        for p in predictions[:TOP_N]
    ]


def _sections(markdown: str) -> list[tuple[str, str]]:
    """(heading, body) for every '#'-heading section of a markdown document."""
    parts = re.split(r"(?m)^(#{1,4} .+)$", markdown)
    out = [("", parts[0])] if parts[0].strip() else []
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i].lstrip("#").strip(), parts[i + 1]))
    return out


def build_tools(ws: Workspace, conversation: Conversation) -> list[Callable]:
    """The tool functions for one chat, bound to the workspace and its conversation state."""

    def get_run_summary(run_id: str = "") -> dict:
        """Summarize one run of the CrewAI Flow: status, stage durations, the three quality gates
        (checks passed / total), LLM calls and tokens, errors and warnings. Leave run_id empty
        for the promoted run, the one whose artifacts are published."""
        folder = ws.run_folder(run_id)
        if folder is None:
            return {
                "error": f"no run named {run_id!r}",
                "available_runs": [r.get("run_id") for r in ws.list_runs()],
                "source": ["runs/"],
            }
        status = ws.json("run_status.json", folder) or {}
        manifest = ws.json("run_manifest.json", folder) or {}
        gates = gate_summary(ws.json("validation_report.json", ws.report_dir(folder)))
        usage = manifest.get("token_usage") or {}
        seconds = duration_seconds(status.get("started_at"), status.get("finished_at"))
        return {
            "run_id": status.get("run_id"),
            "status": status.get("status"),
            "failed_at_stage": status.get("stage") if status.get("status") == "failed" else None,
            "mode": status.get("mode"),
            "llm_model": manifest.get("run", {}).get("llm_model"),
            "published": folder == ws.artifacts,
            "duration": f"{int(seconds // 60)} min {int(seconds % 60)} s" if seconds else None,
            "stages": [
                {"stage": s["stage"], "seconds": round(s["seconds"], 1)}
                for s in stage_durations(ws.run_log(folder))
            ],
            "gates": {
                name: f"{'PASS' if g['passed'] else 'FAIL'} {g['n_passed']}/{g['n_total']}"
                for name, g in sorted(
                    gates.items(),
                    key=lambda item: GATE_ORDER.index(item[0]) if item[0] in GATE_ORDER else 9,
                )
            },
            "llm_calls": usage.get("successful_requests"),
            "total_tokens": f"{usage['total_tokens']:,}" if usage.get("total_tokens") else None,
            "errors": status.get("errors", []),
            "warnings": status.get("warnings", []),
            "narrative_sources": manifest.get("narrative_sources"),
            "source": [
                "run_status.json",
                "run_manifest.json",
                "validation_report.json",
                "pipeline.log",
            ],
        }

    def get_gate_checks(gate: str) -> dict:
        """List every check of one quality gate with its result and detail. gate is one of
        contract, features or artifacts (or the full names gate_contract,
        gate_required_features, gate_artifacts)."""
        name = GATES.get(gate.strip().lower().removeprefix("gate_"), gate.strip().lower())
        gates = gate_summary(ws.json("validation_report.json"))
        if name not in gates:
            return {"error": f"unknown gate {gate!r}", "gates": list(gates), "source": []}
        g = gates[name]
        return {
            "gate": name,
            "result": "PASS" if g["passed"] else "FAIL",
            "checks_passed": f"{g['n_passed']}/{g['n_total']}",
            "checks": [
                {"check": c.get("name"), "passed": c.get("passed"), "detail": c.get("detail")}
                for c in g["checks"]
            ],
            "source": ["validation_report.json"],
        }

    def get_model_metrics() -> dict:
        """The model comparison: the selected model and why, test-window metrics of every
        candidate, the decision threshold, the high-risk cutoff, the time split and the
        strongest features."""
        metrics, meta = ws.json("metrics.json") or {}, ws.json("model_meta.json") or {}
        if not metrics:
            return {"error": "metrics.json not found", "source": []}
        chosen = metrics["selected_model"]
        bundle = ws.bundle()
        candidates = {
            MODEL_LABELS.get(name, name): {
                "cv_pr_auc": f"{c['cv']['pr_auc_mean']:.3f}",
                "test_roc_auc": f"{c['test']['roc_auc']:.3f}",
                "test_pr_auc": f"{c['test']['pr_auc']:.3f}",
                "positive_rate_in_riskiest_10pct": pct(c["test"]["precision_at_top10pct"]),
                "lift_in_riskiest_10pct": (
                    f"{c['test']['lift_at_top10pct']:.1f}x"
                    if c["test"]["lift_at_top10pct"] is not None
                    else "n/a"
                ),
                "brier": f"{c['test']['brier']:.4f}",
            }
            for name, c in metrics["candidates"].items()
        }
        split = {
            name: {
                "dates": " to ".join(s["dates"]),
                "rows": f"{s['rows']:,}",
                "positive_rate": s["prevalence"],
            }
            for name, s in metrics["split"].items()
        }
        return as_percentages(
            {
                "selected_model": MODEL_LABELS.get(chosen, chosen),
                "selection_rule": metrics["selection"]["rule"],
                "selection_margin": f"{metrics['selection']['selection_margin']:.3f}",
                "candidates": candidates,
                "decision_threshold": pct(meta.get("threshold")),
                "high_risk_from": pct(bundle.top10_cutoff) if bundle.ok else None,
                "riskiest_share": "10%",
                "split": split,
                "top_features": [
                    f["feature"] for f in metrics["permutation_importance"]["features"][:5]
                ],
                "source": ["metrics.json", "model_meta.json"],
            }
        )

    def read_document(name: str, section: str = "") -> dict:
        """Read a document written by the pipeline: model_card, evaluation_report, insights or
        cleaning_report. Give a section heading (for example 'Limitations' or 'Ethical
        considerations') to read only that part."""
        file = DOCUMENTS.get(name.strip().lower().replace(" ", "_"))
        text = ws.text(file) if file else None
        if not text:
            return {
                "error": f"unknown document {name!r}",
                "documents": list(DOCUMENTS),
                "source": [],
            }
        sections = _sections(text)
        body = text
        if section:
            match = next((b for h, b in sections if section.lower() in h.lower()), None)
            if match is None:
                return {
                    "error": f"no section {section!r}",
                    "sections": [h for h, _ in sections if h],
                    "source": [file],
                }
            body = match
        return {
            "document": file,
            "section": section or None,
            "text": body.strip()[:MAX_TEXT],
            "sections": [h for h, _ in sections if h],
            "source": [file],
        }

    def get_contract(column: str = "") -> dict:
        """The dataset contract the Data Analyst crew handed to the Data Scientist crew: version,
        rows, required and forbidden features, binding constraints. Give a column name to read
        its type, nullability, allowed values or bounds."""
        contract = ws.json("dataset_contract.json")
        if not contract:
            return {"error": "dataset_contract.json not found", "source": []}
        if column:
            spec = next((c for c in contract["columns"] if c["name"] == column.strip()), None)
            if spec is None:
                return {
                    "error": f"unknown column {column!r}",
                    "columns": [c["name"] for c in contract["columns"]],
                    "source": ["dataset_contract.json"],
                }
            allowed = spec.get("allowed_values") or []
            return {
                "column": spec["name"],
                "type": spec["dtype"],
                "known_at": spec["available_at"],
                "nullable": spec["nullable"],
                "allowed_values": allowed[:30],
                "allowed_values_total": len(allowed),
                "bounds": spec.get("bounds"),
                "description": spec.get("description"),
                "source": ["dataset_contract.json"],
            }
        return {
            "contract_version": contract["contract_version"],
            "rows": f"{contract['row_count']:,}",
            "columns": len(contract["columns"]),
            "target": contract["target"]["definition"],
            "required_features": [f["name"] for f in contract["required_features"]],
            "forbidden_features": contract["forbidden_features"],
            "constraints": [
                {"id": c["id"], "rule": c["description"]} for c in contract["constraints"]
            ],
            "source": ["dataset_contract.json"],
        }

    def list_runs() -> dict:
        """List the pipeline runs available: the promoted run and any local runs."""
        return {
            "runs": [
                {
                    "run_id": r.get("run_id"),
                    "status": r.get("status"),
                    "mode": r.get("mode"),
                    "started_at": r.get("started_at"),
                    "where": r["_source"],
                }
                for r in ws.list_runs()
            ],
            "source": ["run_status.json", "runs/"],
        }

    def search_log(pattern: str, run_id: str = "") -> dict:
        """Find lines in a run's pipeline.log that contain a text, for example GATE, HANDOFF,
        CREW, ERROR or 'LLM call'. Returns the last 20 matches."""
        folder = ws.run_folder(run_id)
        log = ws.run_log(folder) if folder else None
        if not log:
            return {"error": "no pipeline.log for that run", "source": []}
        needle = pattern.strip().lower()
        lines = [line for line in log.splitlines() if needle in line.lower()]
        return {
            "pattern": pattern,
            "matches": len(lines),
            "lines": lines[-20:],
            "source": ["pipeline.log"],
        }

    def score_attached_file() -> dict:
        """Score the CSV of records the user attached to this chat: counts per risk level, the
        10 riskiest records and rows that failed validation. The full ranked list and the action
        list are offered for download in the chat."""
        attachment = conversation.attachment
        if not attachment:
            return {"error": "no file is attached; attach a CSV of records first", "source": []}
        if attachment.get("error"):
            return {"error": attachment["error"], "file": attachment.get("name"), "source": []}
        bundle = ws.bundle()
        if not bundle.ok:
            return {"error": f"model unavailable: {bundle.error}", "source": []}
        result = score_records(bundle, attachment["records"], attachment.get("ignored"))
        conversation.outputs.append(
            {
                "kind": "batch",
                "title": f"Scored {attachment['name']}",
                "result": result,
                "backtest": None,
            }
        )
        return {
            "file": attachment["name"],
            "summary": result.summary,
            "riskiest_records": _top(result.predictions),
            "rejected_rows": result.errors[:TOP_N],
            "rejected_rows_total": len(result.errors),
            "source": ["model.joblib", "attached file"],
        }

    return [
        get_run_summary,
        get_gate_checks,
        get_model_metrics,
        read_document,
        get_contract,
        list_runs,
        search_log,
        score_attached_file,
    ]
