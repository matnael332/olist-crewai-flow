"""Gate 3 (before promotion): every spec-required artifact exists and is usable. Never raises."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from pulse_core.io import read_json, sha256_file
from pulse_core.risk import LEVELS
from pulse_core.validators.report import GateReport

GATE = "gate_artifacts"
SPEC_FILES = (
    "clean_data.csv",
    "eda_report.html",
    "insights.md",
    "dataset_contract.json",
    "features.csv",
    "model.joblib",
    "evaluation_report.md",
    "model_card.md",
)
SUPPORTING_FILES = (
    "raw_profile.json",
    "cleaning_report.json",
    "cleaning_report.md",
    "eda_summary.json",
    "feature_spec.json",
    "model_meta.json",
    "metrics.json",
    "predictions_test.csv",
)
MODEL_CARD_HEADERS = (
    "## Model purpose",
    "## Training data summary",
    "## Metrics",
    "## Limitations",
    "## Ethical considerations",
)
TEXT_SUFFIXES = (".md", ".json", ".html", ".log", ".csv")
BAD_TEXT = re.compile(
    r"np\.float64|np\.int64|Traceback \(most recent call last\)|AIza[0-9A-Za-z_\-]{35}"
)


def _present(art: Path, names) -> list[str]:
    return [n for n in names if not (art / n).exists() or (art / n).stat().st_size == 0]


def _check_scores(art: Path) -> tuple[str, bool, str]:
    """A run with score rows has one ranked, policy-levelled score per entity, and reasons."""
    name = "current_scores_usable"
    try:
        scores = pd.read_csv(art / "scores_current.csv")
        reasons = read_json(art / "reasons.json")
        entity = reasons["entity_column"]
        problems = []
        if scores[entity].duplicated().any():
            problems.append("an entity is scored twice")
        if not scores["probability"].between(0, 1).all():
            problems.append("a probability is outside [0, 1]")
        if not set(scores["risk_level"]) <= set(LEVELS):
            problems.append(
                f"unknown risk levels {sorted(set(scores['risk_level']) - set(LEVELS))}"
            )
        if scores["rank"].tolist() != list(range(1, len(scores) + 1)):
            problems.append("ranks are not 1..n in file order")
        if [e[entity] for e in reasons["entities"]] != scores[entity].tolist():
            problems.append("reasons.json does not follow scores_current.csv")
        if problems:
            return name, False, "; ".join(problems)
        return name, True, f"{len(scores)} entities scored at the snapshot, with reasons"
    except Exception as exc:  # noqa: BLE001 - any failure here is a failed check
        return name, False, f"scores_current.csv or reasons.json unusable: {exc}"


def validate_artifacts(artifacts_dir: str | Path) -> GateReport:
    art = Path(artifacts_dir)
    report = GateReport(GATE)
    missing = _present(art, SPEC_FILES)
    report.record(
        "spec_files_present",
        not missing,
        "all required files present" if not missing else f"missing {', '.join(missing)}",
    )
    missing_support = _present(art, SUPPORTING_FILES)
    report.record(
        "supporting_files_present",
        not missing_support,
        "supporting files present" if not missing_support else f"missing {missing_support}",
    )
    try:
        declared = list((read_json(art / "eda_summary.json") or {}).get("figures", []))
        declared += list((read_json(art / "metrics.json") or {}).get("figures", []))
        absent = [n for n in declared if not (art / "figures" / n).exists()]
        report.record(
            "figures_present",
            not absent,
            f"all {len(declared)} declared figures exist" if not absent else f"missing {absent}",
        )
    except (OSError, ValueError) as exc:
        report.record("figures_present", False, f"cannot read the declared figures: {exc}")

    try:
        spec = read_json(art / "feature_spec.json")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(art / "model.joblib")
        sample = pd.read_csv(art / "features.csv", nrows=5, encoding="utf-8")
        kinds = {f["name"]: f["kind"] for f in spec["features"]}
        for name, kind in kinds.items():
            sample[name] = sample[name].astype(str if kind == "categorical" else "float64")
        proba = model.predict_proba(sample[spec["model_features"]])
        ok = proba.shape == (len(sample), 2) and ((proba >= 0) & (proba <= 1)).all()
        report.record("model_loads_and_predicts", bool(ok), "model.joblib loads and scores 5 rows")
    except Exception as exc:  # noqa: BLE001 - any failure here is a failed check
        report.record("model_loads_and_predicts", False, f"model.joblib unusable: {exc}")

    try:
        metrics = read_json(art / "metrics.json")
        names = list(metrics["candidates"])
        real = [n for n in names if n != "baseline_prior"]
        ok = "baseline_prior" in names and len(real) >= 2
        report.record(
            "models_compared", ok, f"{len(real)} model variations compared against the baseline"
        )
    except Exception as exc:  # noqa: BLE001
        report.record("models_compared", False, f"metrics.json unusable: {exc}")

    if (art / "score_rows.csv").exists():
        report.record(*_check_scores(art))

    card = (
        (art / "model_card.md").read_text(encoding="utf-8")
        if (art / "model_card.md").exists()
        else ""
    )
    absent = [h for h in MODEL_CARD_HEADERS if h not in card]
    report.record(
        "model_card_sections",
        not absent,
        "model card has purpose, training data, metrics, limitations, ethics"
        if not absent
        else f"model card missing sections {absent}",
    )

    try:
        digest = sha256_file(art / "dataset_contract.json")
        spec_digest = read_json(art / "feature_spec.json").get("contract_sha256")
        meta_digest = read_json(art / "model_meta.json").get("contract_sha256")
        ok = digest == spec_digest == meta_digest and digest[:12] in card
        report.record(
            "contract_hash_consistent",
            ok,
            f"feature spec, model meta and model card follow contract {digest[:12]}"
            if ok
            else "contract sha256 differs between dataset_contract, feature_spec, model_meta, card",
        )
    except Exception as exc:  # noqa: BLE001
        report.record("contract_hash_consistent", False, f"cannot compare contract hashes: {exc}")

    html = (
        (art / "eda_report.html").read_text(encoding="utf-8")
        if (art / "eda_report.html").exists()
        else ""
    )
    images = html.count('src="data:image/png;base64,')
    try:
        expected_images = len((read_json(art / "eda_summary.json") or {}).get("figures", []))
    except (OSError, ValueError):
        expected_images = 0
    report.record(
        "eda_report_figures",
        bool(html) and images >= expected_images,
        f"eda_report.html embeds {images} of {expected_images} EDA figures",
    )

    dirty = []
    for path in sorted(art.rglob("*")):
        if path.is_file() and path.suffix in TEXT_SUFFIXES and path.stat().st_size < 60_000_000:
            raw = path.read_bytes()
            if b"\r" in raw and path.suffix != ".csv":
                dirty.append(f"{path.name}: CRLF")
            if path.suffix != ".csv" and BAD_TEXT.search(raw.decode("utf-8", errors="ignore")):
                dirty.append(f"{path.name}: numpy repr, traceback or key-like string")
    report.record(
        "text_hygiene", not dirty, "text artifacts are clean" if not dirty else f"issues: {dirty}"
    )
    return report
