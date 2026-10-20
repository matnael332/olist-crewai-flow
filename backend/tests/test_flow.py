"""End-to-end Flow tests in deterministic mode (no LLM) on the fixture."""

import json
from pathlib import Path

import pytest

from olist_core.faults import FAULTS
from olist_core.validators.artifacts_gate import SPEC_FILES

FIXTURE = Path(__file__).parent / "fixtures" / "olist_mini"


def _run(tmp_path: Path, run_id: str, *extra: str) -> int:
    from olist_flow.run import main

    return main(
        [
            "--no-llm",
            "--data-dir",
            str(FIXTURE),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--run-id",
            run_id,
            *extra,
        ]
    )


def _status(tmp_path: Path, run_id: str) -> dict:
    return json.loads((tmp_path / "runs" / run_id / "run_status.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def successful_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("flow")
    code = _run(tmp, "ok")
    return tmp, code


def test_success_run_promotes_spec_artifacts(successful_run):
    tmp, code = successful_run
    assert code == 0
    status = _status(tmp, "ok")
    assert status["status"] == "success"
    assert status["stages_done"] == [
        "check_inputs",
        "run_analyst_stage",
        "gate_contract",
        "run_feature_phase",
        "gate_required_features",
        "run_modeling_phase",
        "gate_artifacts",
        "promote_and_finish",
    ]
    promoted = tmp / "artifacts"
    for name in SPEC_FILES:
        assert (promoted / name).exists(), name
    report = json.loads((promoted / "validation_report.json").read_text(encoding="utf-8"))
    assert {g: r["passed"] for g, r in report["gates"].items()} == {
        "gate_contract": True,
        "gate_required_features": True,
        "gate_artifacts": True,
    }
    manifest = json.loads((promoted / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run"]["status"] == "success"
    assert "clean_data.csv" in manifest["artifacts"]
    log = (promoted / "logs" / "pipeline.log").read_text(encoding="utf-8")
    assert "HANDOFF analyst->scientist" in log and "GATE gate_contract RESULT PASS" in log
    assert b"\r" not in (promoted / "logs" / "pipeline.log").read_bytes()
    from olist_core.validate import main as validate_main

    assert validate_main([str(promoted)]) == 0


def test_resume_from_previous_run(successful_run):
    tmp, _ = successful_run
    code = _run(tmp, "resume", "--from-run", "ok", "--start-at", "modeling", "--no-promote")
    assert code == 0
    assert _status(tmp, "resume")["stages_done"][:2] == ["gate_required_features", "check_inputs"]


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_every_fault_fails_gracefully(tmp_path, capfd, fault):
    existing = tmp_path / "artifacts"
    existing.mkdir()
    (existing / "sentinel.txt").write_text("previous run", encoding="utf-8")
    code = _run(tmp_path, f"fault-{fault}", "--inject-fault", fault)
    out, err = capfd.readouterr()
    assert code == 1
    status = _status(tmp_path, f"fault-{fault}")
    assert status["status"] == "failed"
    assert status["stage"] == FAULTS[fault]
    assert status["errors"][0].startswith(f"{FAULTS[fault]}: ")
    assert "Traceback" not in out and "Traceback" not in err
    assert sorted(p.name for p in existing.iterdir()) == ["sentinel.txt"]
    assert (tmp_path / "runs" / f"fault-{fault}" / "artifacts" / "validation_report.json").exists()
