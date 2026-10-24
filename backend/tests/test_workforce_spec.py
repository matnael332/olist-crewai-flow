"""The real workforce_saas spec runs the generic pipeline end to end on the synthetic data: the
three gates pass, the clean table matches the planted story, and the model learns the planted
signals rather than noise."""

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pulse_core.columns import read_clean_data
from pulse_core.io import read_json, write_json_lf
from pulse_core.raw import load_raw_tables
from pulse_core.spec import SpecNotConfigured, load_spec
from pulse_core.stages import run_analyst_stage, run_feature_stage, run_modeling_stage
from pulse_core.validators.artifacts_gate import validate_artifacts
from pulse_core.validators.contract_gate import validate_clean_data_against_contract
from pulse_core.validators.features_gate import validate_features
from pulse_specs.workforce_saas import columns as wf
from pulse_specs.workforce_saas.clean import normalise_raw
from pulse_specs.workforce_saas.spec import SPEC, make_spec, windows
from pulse_specs.workforce_saas.synth import SynthConfig, generate, write_dataset

CONFIG = SynthConfig(accounts=80, dirt=0.02)  # default dates, so SPEC applies unchanged
SNAPSHOT = pd.Timestamp(SPEC.snapshot_date)
FACTORY_REF = "pulse_specs.workforce_saas.spec:make_spec"
STAGES = [
    "check_inputs",
    "run_analyst_stage",
    "gate_contract",
    "run_feature_phase",
    "gate_required_features",
    "run_modeling_phase",
    "gate_artifacts",
    "promote_and_finish",
]


@pytest.fixture(scope="module")
def wf80(tmp_path_factory):
    data = tmp_path_factory.mktemp("wf_data")
    raw = write_dataset(data, CONFIG)
    out = tmp_path_factory.mktemp("wf_out") / "artifacts"
    run_analyst_stage(SPEC, raw, data / "lookups", out)
    run_feature_stage(SPEC, out)
    run_modeling_stage(SPEC, out, 42)
    monthly = pd.read_csv(data / "truth" / "monthly_health.csv")
    monthly["month_end"] = pd.PeriodIndex(monthly["month"], freq="M").end_time.floor("s")
    return SimpleNamespace(
        data=data,
        raw=raw,
        out=out,
        tables=load_raw_tables(raw),
        accounts_truth=pd.read_csv(data / "truth" / "accounts_truth.csv", keep_default_na=False),
        monthly_truth=monthly,
        dirt_counts=read_json(data / "truth" / "dirt_counts.json"),
        clean=read_clean_data(out / "clean_data.csv", SPEC.dtypes),
        report=read_json(out / "cleaning_report.json"),
        metrics=read_json(out / "metrics.json"),
        feature_spec=read_json(out / "feature_spec.json"),
    )


# ----------------------------------------------------------------------------- 1. the gates


def test_all_three_gates_pass(wf80):
    contract = validate_clean_data_against_contract(
        wf80.out / "clean_data.csv", wf80.out / "dataset_contract.json"
    )
    assert contract.passed, contract.errors
    constraint_checks = [c for c in contract.checks if c["name"].startswith("constraint_")]
    assert {c["name"] for c in constraint_checks} == {
        f"constraint_{c['id']}" for c in wf.CONSTRAINTS
    }
    features = validate_features(
        wf80.out / "features.csv",
        wf80.out / "feature_spec.json",
        wf80.out / "dataset_contract.json",
    )
    assert features.passed, features.errors
    artifacts = validate_artifacts(wf80.out)
    assert artifacts.passed, artifacts.errors


# ----------------------------------------------------------------------------- 2. rows and labels


def test_population_matches_the_planted_story(wf80):
    truth = wf80.monthly_truth
    known = truth[truth["month_end"] + pd.Timedelta(days=wf.HORIZON_DAYS) <= SNAPSHOT]
    cancelled = wf80.accounts_truth.set_index("account_id")["cancelled_at"]
    cancel_month = pd.to_datetime(cancelled.replace("", pd.NaT)).dt.to_period("M").astype(str)
    expected = known[known["month"] != known["account_id"].map(cancel_month).fillna("")]
    ids = set(expected["account_id"] + ":" + expected["month"])
    assert set(wf80.clean[wf.ID_COLUMN]) == ids

    report = wf80.report
    assert report["rows_in"] == len(truth) == len(wf80.tables["billing_monthly"])
    assert report["rows_out"] == len(wf80.clean)
    rules = report["rules"]
    assert [r["rule"] for r in rules][:2] == [
        "R1_drop_censored_months",
        "R2_drop_cancellation_month",
    ]
    assert all(re.match(r"^R[0-9]+_", r["rule"]) for r in rules)
    assert all(r["rows_after"] <= r["rows_before"] for r in rules)
    assert rules[0]["rows_before"] == report["rows_in"]
    assert sum(r["rows_removed"] for r in rules) == report["rows_in"] - report["rows_out"]
    assert rules[0]["rows_removed"] > 0 and rules[1]["rows_removed"] > 0


def test_labels_match_the_truth(wf80):
    clean = wf80.clean
    joined = clean.merge(
        wf80.monthly_truth[["account_id", "month", "churned_60d"]],
        on=["account_id", "month"],
        suffixes=("", "_truth"),
    )
    assert len(joined) == len(clean)
    assert (joined[wf.TARGET].astype(bool) == joined["churned_60d_truth"].astype(bool)).all()
    positives = clean[clean[wf.TARGET].astype(bool)]
    assert len(positives) >= 40
    assert positives["days_to_cancel"].between(1, wf.HORIZON_DAYS).all()
    assert (clean["cancelled_at"].isna() | (clean["cancelled_at"] > clean["month_end"])).all()
    assert clean["month"].max() == "2025-06"
    assert clean["tenure_months"].min() == 1
    assert (clean["active_employees"] >= clean["active_sites"]).all()
    month_end = clean["month_end"]
    assert (month_end.dt.hour == 23).all() and (month_end.dt.second == 59).all()
    assert list(clean.columns) == [c.name for c in wf.COLUMNS]
    assert clean.equals(clean.sort_values([wf.TIME_COLUMN, "account_id"], kind="mergesort"))


# ----------------------------------------------------------------------------- 3. splits and features


def test_splits_and_cv_folds_are_well_formed(wf80):
    rows = wf80.feature_spec["split"]["rows"]
    assert set(rows) == {"train", "horizon_gap", "test"} and all(v > 0 for v in rows.values())
    features = pd.read_csv(wf80.out / "features.csv", dtype={wf.ID_COLUMN: str})
    assert features.loc[features["split"] == "test", wf.TARGET].sum() >= 8
    assert features.loc[features["split"] == "train", "month_end"].is_monotonic_increasing
    selected = wf80.metrics["selected_model"]
    folds = wf80.metrics["candidates"][selected]["cv"]["folds"]
    assert len(folds) == 3 and all(f["val_prevalence"] > 0 for f in folds)


def test_feature_table_layout_and_future_invariance(wf80):
    features = pd.read_csv(wf80.out / "features.csv", dtype={wf.ID_COLUMN: str})
    assert list(features.columns) == [
        wf.ID_COLUMN,
        wf.TIME_COLUMN,
        "split",
        *SPEC.model_features,
        wf.TARGET,
    ]
    assert len(SPEC.model_features) == 57
    required = {f["name"] for f in SPEC.required_features}
    assert required <= set(SPEC.model_features) and not required & set(wf.FORBIDDEN)
    for definition in SPEC.feature_definitions:
        if not definition["nullable"]:
            assert features[definition["name"]].notna().all(), definition["name"]
    first = SPEC.derive_features(wf80.clean)
    perturbed = wf80.clean.copy()
    perturbed["cancelled_at"] = pd.Timestamp("2030-01-01")
    perturbed["days_to_cancel"] = 5
    perturbed[wf.TARGET] = True
    pd.testing.assert_frame_equal(SPEC.derive_features(perturbed), first)
    streak = first["months_without_payroll_export"]
    assert streak.max() >= 2 and (streak[first["payroll_exported"] == 1] == 0).all()


# ----------------------------------------------------------------------------- 4. the planted signals


def test_model_learns_the_planted_signals(wf80):
    selected = wf80.metrics["selected_model"]
    assert selected != "baseline_prior"
    predictions = pd.read_csv(wf80.out / "predictions_test.csv")
    assert roc_auc_score(predictions["y_true"], predictions[f"p_{selected}"]) >= 0.65
    test = wf80.metrics["candidates"][selected]["test"]
    baseline = wf80.metrics["candidates"]["baseline_prior"]["test"]
    assert test["pr_auc"] > 2 * baseline["pr_auc"]

    features = pd.read_csv(wf80.out / "features.csv", dtype={wf.ID_COLUMN: str})
    kinds = {f["name"]: f["kind"] for f in SPEC.feature_definitions}
    matrix = features[SPEC.model_features].copy()
    for name in matrix:
        matrix[name] = matrix[name].astype(object if kinds[name] == "categorical" else "float64")
    scores = features[[wf.ID_COLUMN]].copy()
    scores["p"] = joblib.load(wf80.out / "model.joblib").predict_proba(matrix)[:, 1]
    scores[["account_id", "month"]] = scores[wf.ID_COLUMN].str.split(":", expand=True)
    scores = scores.merge(
        wf80.accounts_truth[["account_id", "archetype", "decline_start"]], on="account_id"
    )
    fading = scores[(scores["archetype"] == "slow_fade") & (scores["decline_start"] != "")]
    fading = fading[fading["month"] >= fading["decline_start"]]
    steady = scores[scores["archetype"] == "steady"]
    assert len(fading) >= 10 and len(steady) >= 100
    assert fading["p"].mean() > 3 * steady["p"].mean()


# ----------------------------------------------------------------------------- 5. determinism and dirt


def test_clean_is_deterministic(wf80):
    first, report_a = SPEC.clean(wf80.tables, wf80.data / "lookups")
    second, report_b = SPEC.clean(wf80.tables, wf80.data / "lookups")
    pd.testing.assert_frame_equal(first, second)
    assert report_a == report_b
    assert len(first) == len(wf80.clean)


def test_raw_fixes_are_counted(wf80):
    fixes = wf80.report["raw_fixes"]
    planted = wf80.dirt_counts
    assert fixes["total"] > 0
    assert fixes["duplicate_rows"]["hours_daily"] == planted["duplicate_rows"]["hours_daily"] > 0
    assert fixes["bad_enum"]["shifts"] == planted["bad_enum"]["shifts"] > 0
    assert fixes["swapped_range"]["shifts"] == planted["swapped_range"]["shifts"] > 0
    assert (
        fixes["bad_enum"]["punches"]
        <= planted["bad_enum"]["punches"] + planted["late_edit"]["punches"]
    )
    summary = read_json(wf80.out / "eda_summary.json")
    assert summary["raw_fixes"]["total"] == fixes["total"]
    assert any(
        k["label"] == "raw rows fixed" and k["value"] == fixes["total"]
        for k in summary["key_numbers"]
    )


def test_clean_data_needs_no_fixes():
    tables, _truth, _counts = generate(SynthConfig(accounts=12, dirt=0.0))
    _fixed, fixes = normalise_raw(tables)
    assert fixes["total"] == 0
    assert all(
        n == 0
        for kind in ("duplicate_rows", "bad_enum", "swapped_range")
        for n in fixes[kind].values()
    )


# ----------------------------------------------------------------------------- 6. the factory and the Flow


def test_windows_follow_the_snapshot():
    assert SPEC.splits == {
        "train": ("2022-01-01", "2024-10-31"),
        "horizon_gap": ("2024-11-01", "2024-12-31"),
        "test": ("2025-01-01", "2025-06-30"),
    }
    assert windows("2024-07-01") == {
        "train": ("2022-01-01", "2023-08-31"),
        "horizon_gap": ("2023-09-01", "2023-10-31"),
        "test": ("2023-11-01", "2024-04-30"),
    }
    other = make_spec("2024-07-01", train_start="2022-06-01", test_months=3, gap_months=1)
    assert other.splits["test"] == ("2024-02-01", "2024-04-30")
    assert other.splits["horizon_gap"] == ("2024-01-01", "2024-01-31")
    assert other.splits["train"] == ("2022-06-01", "2023-12-31")
    assert other.snapshot_date == "2024-07-01" and other.name == SPEC.name
    with pytest.raises(ValueError):
        make_spec("2024-07-01", train_start="2024-01-01")


def test_load_spec_builds_one_spec_per_customer():
    built = load_spec(FACTORY_REF, {"snapshot_date": "2024-07-01", "test_months": 3})
    assert built.snapshot_date == "2024-07-01"
    assert built.splits["test"] == ("2024-02-01", "2024-04-30")
    assert load_spec(FACTORY_REF).splits == SPEC.splits  # no params: the factory defaults
    with pytest.raises(SpecNotConfigured, match="takes no spec params"):
        load_spec("pulse_specs.workforce_saas.spec:SPEC", {"snapshot_date": "2024-07-01"})
    with pytest.raises(SpecNotConfigured, match="cannot build"):
        load_spec(FACTORY_REF, {"no_such_param": 1})
    with pytest.raises(SpecNotConfigured, match="cannot build"):
        load_spec(FACTORY_REF, {"snapshot_date": "2024-07-01", "train_start": "2024-01-01"})


def test_flow_runs_end_to_end(tmp_path, wf80):
    """The hosted path: the factory plus a --spec-params file (the same snapshot as SPEC)."""
    from pulse_flow.run import main

    params = tmp_path / "spec_params.json"
    params.write_text(json.dumps({"snapshot_date": SPEC.snapshot_date}), encoding="utf-8")
    code = main(
        [
            "--no-llm",
            "--spec",
            FACTORY_REF,
            "--spec-params",
            str(params),
            "--data-dir",
            str(wf80.data),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--run-id",
            "wf",
        ]
    )
    assert code == 0
    status = json.loads((tmp_path / "runs" / "wf" / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "success" and status["stages_done"] == STAGES
    report = read_json(tmp_path / "artifacts" / "validation_report.json")
    assert all(
        report["gates"][g] for g in ("gate_contract", "gate_required_features", "gate_artifacts")
    )
    manifest = read_json(tmp_path / "artifacts" / "run_manifest.json")
    assert manifest["run"]["run_id"] == "wf" and "punches.csv" in manifest["inputs"]
    assert manifest["run"]["spec_ref"] == FACTORY_REF
    assert manifest["run"]["spec_params"] == {"snapshot_date": SPEC.snapshot_date}
    contract = read_json(tmp_path / "artifacts" / "dataset_contract.json")
    assert contract["dataset"]["name"] == "workforce_saas"
    assert "pseudonymous" in contract["privacy"]["note"].lower()
    assert contract["target"]["split_policy"]["horizon_gap"] == ["2024-11-01", "2024-12-31"]
    card = (tmp_path / "artifacts" / "model_card.md").read_text(encoding="utf-8")
    assert "Retrain whenever the snapshot moves" in card
    assert np.isfinite(
        read_json(tmp_path / "artifacts" / "metrics.json")["candidates"]["logistic_regression"][
            "test"
        ]["roc_auc"]
    )


# ----------------------------------------------------------------------------- 7. optional data groups


def _trimmed_raw(raw: Path, out: Path, groups: tuple[str, ...]) -> Path:
    """A copy of ``raw`` holding only the tables of ``groups``, as the hosted ingestion writes it:
    the schema and the manifest list the present tables only."""
    keep = {t for g in groups for t in wf.GROUP_TABLES[g]}
    schema = read_json(raw / "raw_schema.json")
    schema["tables"] = {t: s for t, s in schema["tables"].items() if t in keep}
    manifest = read_json(raw / "manifest.json")
    files = {s["file"] for s in schema["tables"].values()}
    manifest["files"] = {f: e for f, e in manifest["files"].items() if f in files}
    (out / "raw").mkdir(parents=True)
    (out / "lookups").mkdir()
    for name in files:
        shutil.copyfile(raw / name, out / "raw" / name)
    write_json_lf(out / "raw" / "raw_schema.json", schema, ndigits=None)
    write_json_lf(out / "raw" / "manifest.json", manifest, ndigits=None)
    return out


def _run_groups(tmp_path: Path, data: Path, groups: tuple[str, ...], run_id: str) -> int:
    from pulse_flow.run import main

    params = tmp_path / f"{run_id}_params.json"
    params.write_text(
        json.dumps({"snapshot_date": SPEC.snapshot_date, "groups": list(groups)}),
        encoding="utf-8",
    )
    return main(
        [
            "--no-llm",
            "--spec",
            FACTORY_REF,
            "--spec-params",
            str(params),
            "--data-dir",
            str(data),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--run-id",
            run_id,
        ]
    )


def test_groups_are_checked():
    with pytest.raises(ValueError, match="core data groups missing"):
        make_spec(groups=("commercial", "punches", "usage"))
    with pytest.raises(ValueError, match="unknown data groups"):
        make_spec(groups=(*wf.CORE_GROUPS, "weather"))
    names = {c.name for c in wf.columns_for((*wf.CORE_GROUPS, "hours"))}
    assert "overtime_share" in names and "labor_cost" not in names  # needs the wages as well
    assert wf.columns_for(wf.ALL_GROUPS) == wf.COLUMNS
    assert make_spec().data_groups["missing"] == []


def test_hours_without_wages_drop_labor_cost_only(wf80):
    spec = make_spec(groups=(*wf.CORE_GROUPS, "hours"))
    clean, _report = spec.clean(wf80.tables, wf80.data / "lookups")
    assert list(clean.columns) == [c.name for c in spec.columns]
    assert "overtime_share" in clean and "labor_cost" not in clean
    # same values as the full run (wf80.clean comes back from the CSV, hence no dtype check)
    pd.testing.assert_frame_equal(clean, wf80.clean[list(clean.columns)], check_dtype=False)


def test_core_groups_alone_run_end_to_end(tmp_path, wf80):
    data = _trimmed_raw(wf80.raw, tmp_path / "core_data", wf.CORE_GROUPS)
    assert _run_groups(tmp_path, data, wf.CORE_GROUPS, "core") == 0
    art = tmp_path / "artifacts"
    contract = read_json(art / "dataset_contract.json")
    missing = [g for g in wf.ALL_GROUPS if g not in wf.CORE_GROUPS]
    assert contract["data_groups"]["missing"] == missing
    columns = {c["name"] for c in contract["columns"]}
    assert "labor_cost" not in columns and "punches" in columns
    features = read_json(art / "feature_spec.json")
    names = {f["name"] for f in features["features"]}
    assert "overtime_share_delta_1m" not in names and "punches_delta_1m" in names
    card = (art / "model_card.md").read_text(encoding="utf-8")
    assert "No device health data (devices)" in card and "data_gap" not in card


def test_a_missing_core_group_fails_gracefully(tmp_path, wf80):
    groups = ("commercial", "punches", "usage")
    data = _trimmed_raw(wf80.raw, tmp_path / "no_payroll", groups)
    assert _run_groups(tmp_path, data, groups, "nopayroll") == 1
    status = json.loads(
        (tmp_path / "runs" / "nopayroll" / "run_status.json").read_text(encoding="utf-8")
    )
    assert status["stage"] == "check_inputs"
    assert "core data groups missing: ['payroll']" in status["errors"][0]


def test_a_group_the_spec_reads_must_be_in_the_raw_schema(tmp_path, wf80):
    data = _trimmed_raw(wf80.raw, tmp_path / "core_only", wf.CORE_GROUPS)
    assert _run_groups(tmp_path, data, (*wf.CORE_GROUPS, "shifts"), "noshifts") == 1
    status = json.loads(
        (tmp_path / "runs" / "noshifts" / "run_status.json").read_text(encoding="utf-8")
    )
    assert "the spec reads shifts" in status["errors"][0]


# ----------------------------------------------------------------------------- 8. scoring at the snapshot


def test_live_accounts_are_scored_at_the_last_complete_month(wf80):
    scores = pd.read_csv(wf80.out / "scores_current.csv")
    cancelled = pd.to_datetime(
        wf80.accounts_truth.set_index("account_id")["cancelled_at"].replace("", pd.NaT)
    )
    live = set(
        cancelled[cancelled.isna() | (cancelled > pd.Timestamp("2025-08-31 23:59:59"))].index
    )
    billed = wf80.tables["billing_monthly"]
    billed = set(billed.loc[billed["period"].dt.to_period("M") == "2025-08", "account_id"])
    assert set(scores["account_id"]) == live & billed
    assert (scores["month_end"] == "2025-08-31 23:59:59").all()
    assert scores["rank"].tolist() == list(range(1, len(scores) + 1))
    assert scores["probability"].is_monotonic_decreasing


def test_scores_follow_the_planted_hazard(wf80):
    scores = pd.read_csv(wf80.out / "scores_current.csv")
    truth = wf80.monthly_truth[wf80.monthly_truth["month"] == "2025-08"]
    merged = scores.merge(truth[["account_id", "hazard"]], on="account_id")
    assert len(merged) == len(scores)
    assert merged["probability"].corr(merged["hazard"], method="spearman") > 0.3


def test_reasons_explain_the_scores(wf80):
    from pulse_specs.workforce_saas.features import REASON_CODES

    scores = pd.read_csv(wf80.out / "scores_current.csv")
    reasons = read_json(wf80.out / "reasons.json")
    assert [e["account_id"] for e in reasons["entities"]] == scores["account_id"].tolist()
    top = reasons["entities"][0]["reasons"]
    assert top, "the riskiest account has at least one reason"
    for entity in reasons["entities"]:
        contributions = [r["contribution"] for r in entity["reasons"]]
        assert all(c > 0 for c in contributions) and contributions == sorted(
            contributions, reverse=True
        )
        for r in entity["reasons"]:
            assert r["feature"] in reasons["features_tried"]
            assert r["code"] == REASON_CODES.get(r["feature"], r["feature"])
