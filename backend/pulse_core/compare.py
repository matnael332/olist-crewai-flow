"""Compare two artifact folders to demonstrate reproducibility.

Policies: data artifacts must be byte-identical; JSON documents that embed narrative are compared
with the narrative keys removed; narrative files are identical only when both runs used templates;
figures are same-machine identical; model.joblib is never byte-compared (predictions are).
"""

from __future__ import annotations

import json
from pathlib import Path

from pulse_core.io import read_json, sha256_file

BYTE_IDENTICAL = (
    "clean_data.csv",
    "cleaning_report.json",
    "raw_profile.json",
    "eda_summary.json",
    "features.csv",
    "metrics.json",
    "predictions_test.csv",
)
# byte-identical when the spec scores the snapshot; absent from both folders otherwise
BYTE_IDENTICAL_IF_SCORED = ("score_rows.csv", "scores_current.csv", "reasons.json")
JSON_WITHOUT = {
    "dataset_contract.json": ("assumptions", "business_rules", "column_notes", "known_limitations"),
    "feature_spec.json": ("narrative", "contract_sha256", "input_validation"),
    "model_meta.json": ("contract_sha256", "feature_spec_sha256"),
}
NARRATIVE = (
    "insights.md",
    "cleaning_report.md",
    "evaluation_report.md",
    "model_card.md",
    "eda_report.html",
)


def _json_digest(path: Path, drop: tuple[str, ...]) -> str:
    doc = read_json(path)
    for key in drop:
        doc.pop(key, None)
    return json.dumps(doc, sort_keys=True)


def _narrative_source(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    marker = "narrative_source: "
    return text.split(marker, 1)[1].split(" ", 1)[0] if marker in text else "template"


def compare_dirs(a: str | Path, b: str | Path) -> dict:
    a, b = Path(a), Path(b)
    rows = []

    def add(name: str, policy: str, result: str) -> None:
        rows.append({"artifact": name, "policy": policy, "result": result})

    scored = [n for n in BYTE_IDENTICAL_IF_SCORED if (a / n).exists() or (b / n).exists()]
    for name in (*BYTE_IDENTICAL, *scored):
        if not (a / name).exists() or not (b / name).exists():
            add(name, "byte-identical", "missing")
            continue
        same = sha256_file(a / name) == sha256_file(b / name)
        add(name, "byte-identical", "identical" if same else "DIFFERS")
    for name, drop in JSON_WITHOUT.items():
        if not (a / name).exists() or not (b / name).exists():
            add(name, "json (narrative removed)", "missing")
            continue
        same = _json_digest(a / name, drop) == _json_digest(b / name, drop)
        add(name, "json (narrative removed)", "identical" if same else "DIFFERS")
    for name in NARRATIVE:
        if not (a / name).exists() or not (b / name).exists():
            add(name, "narrative", "missing")
            continue
        sources = (
            {_narrative_source(a / name), _narrative_source(b / name)}
            if name.endswith(".md")
            else set()
        )
        same = sha256_file(a / name) == sha256_file(b / name)
        if same:
            add(name, "narrative", "identical")
        elif sources - {"template"}:
            add(name, "narrative", "differs (LLM narrative, expected)")
        else:
            add(name, "narrative", "differs")
    figures_a = sorted((a / "figures").glob("*.png")) if (a / "figures").exists() else []
    same_figs = sum(
        1
        for p in figures_a
        if (b / "figures" / p.name).exists()
        and sha256_file(p) == sha256_file(b / "figures" / p.name)
    )
    add("figures/*.png", "same machine", f"{same_figs}/{len(figures_a)} identical")
    add("model.joblib", "never byte-compared", "see predictions_test.csv")

    deterministic_ok = all(
        r["result"] == "identical"
        for r in rows
        if r["policy"] in ("byte-identical", "json (narrative removed)")
    )
    return {"deterministic_identical": deterministic_ok, "rows": rows}


def format_table(result: dict) -> str:
    width = max(len(r["artifact"]) for r in result["rows"])
    lines = [f"{'artifact':<{width}}  {'policy':<26}  result", "-" * (width + 50)]
    lines += [f"{r['artifact']:<{width}}  {r['policy']:<26}  {r['result']}" for r in result["rows"]]
    verdict = (
        "ALL DETERMINISTIC ARTIFACTS IDENTICAL"
        if result["deterministic_identical"]
        else ("DETERMINISTIC ARTIFACTS DIFFER")
    )
    return "\n".join(lines + ["", verdict])
