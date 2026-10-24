import pytest

from pulse_core.validators.artifacts_gate import MODEL_CARD_HEADERS, validate_artifacts


def test_artifacts_gate_passes(pipeline_out):
    report = validate_artifacts(pipeline_out)
    assert report.passed, report.errors


def test_reports_have_required_sections(pipeline_out):
    card = (pipeline_out / "model_card.md").read_text(encoding="utf-8")
    for header in MODEL_CARD_HEADERS:
        assert header in card
    evaluation = (pipeline_out / "evaluation_report.md").read_text(encoding="utf-8")
    for header in (
        "## Cross-validation on the train window",
        "## Test window results",
        "## Model selection",
        "## Reviewer commentary",
    ):
        assert header in evaluation
    assert "narrative_source: template" in card and "narrative_source: template" in evaluation


@pytest.mark.parametrize(
    ("case", "failed_check"),
    [
        ("missing_card", "spec_files_present"),
        ("corrupt_model", "model_loads_and_predicts"),
        ("card_section", "model_card_sections"),
        ("contract_edit", "contract_hash_consistent"),
    ],
)
def test_artifacts_gate_failure_modes(pipeline_copy, case, failed_check):
    if case == "missing_card":
        (pipeline_copy / "model_card.md").unlink()
    elif case == "corrupt_model":
        (pipeline_copy / "model.joblib").write_bytes(b"not a model")
    elif case == "card_section":
        card = pipeline_copy / "model_card.md"
        text = card.read_text(encoding="utf-8").replace("## Ethical considerations", "## Ethics?")
        card.write_text(text, encoding="utf-8", newline="\n")
    elif case == "contract_edit":
        contract = pipeline_copy / "dataset_contract.json"
        contract.write_text(contract.read_text(encoding="utf-8") + " ", encoding="utf-8")
    report = validate_artifacts(pipeline_copy)
    failed = {c["name"] for c in report.checks if not c["passed"]}
    assert not report.passed and failed_check in failed, report.errors
    assert all(e.startswith("gate_artifacts: ") for e in report.errors)
