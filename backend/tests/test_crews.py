"""Crew structure, YAML placeholders, tools and the LLM path with a fake kickoff (no network)."""

import json
import re
import shutil
from pathlib import Path

import pytest

from pulse_core.narratives import (
    CleaningNarrative,
    ColumnNote,
    ContractNarrative,
    EvaluationNarrative,
    FeatureNarrative,
    Finding,
    Insights,
    ModelCardNarrative,
)
from tests.conftest import SPEC_REF

CREWS = Path(__file__).resolve().parents[1] / "pulse_flow" / "crews"
COMMON = {"run_id", "dataset_name", "entity", "target_label"}
DIGEST_KEYS = {
    "contract_version",
    "contract_sha256",
    "row_count",
    "target_definition",
    "required_features",
    "forbidden_features",
    "split_policy",
}

FAKES = {
    CleaningNarrative: CleaningNarrative(
        summary="LLM cleaning summary",
        issues_found=["issue"],
        decisions=["decision"],
        caveats=["caveat"],
    ),
    Insights: Insights(
        headline="LLM headline",
        key_findings=[Finding(title="t", evidence="e", business_meaning="m")],
        recommendations=["r"],
        data_caveats=["c"],
        modeling_implications=["i"],
    ),
    ContractNarrative: ContractNarrative(
        assumptions=["LLM assumption"],
        business_rules=["LLM rule"],
        column_notes=[
            ColumnNote(name="x_num", note="LLM note"),
            ColumnNote(name="not_a_column", note="dropped"),
        ],
        known_limitations=["LLM limitation"],
    ),
    FeatureNarrative: FeatureNarrative(
        contract_compliance="LLM compliance",
        leakage_statement="LLM leakage",
        feature_groups=["g"],
        concerns=["c"],
    ),
    EvaluationNarrative: EvaluationNarrative(
        executive_summary="LLM summary",
        comparison_summary="LLM comparison",
        strengths=["s"],
        weaknesses=["w"],
        threshold_recommendation="LLM threshold",
        next_steps=["n"],
    ),
    ModelCardNarrative: ModelCardNarrative(
        purpose="LLM purpose",
        intended_users="ops",
        out_of_scope=["x"],
        training_data_summary="LLM data",
        limitations=["LLM limitation about the calibration of this model."],
        ethical_considerations=["LLM ethics point about the errors of this model."],
        monitoring_recommendations=["LLM monitoring of the threshold of this model."],
    ),
}


class FakeTaskOutput:
    def __init__(self, name, pydantic=None, raw=""):
        self.name, self.pydantic, self.raw = name, pydantic, raw


class FakeOutput:
    def __init__(self, tasks_output):
        self.tasks_output = tasks_output
        self.token_usage = None


def fake_kickoff(crew, inputs):
    """Structured outputs for every narrator task; tool tasks return their tool's text."""
    outputs = []
    for task in crew.tasks:
        model = task.output_pydantic
        outputs.append(FakeTaskOutput(task.name, FAKES[model] if model else None, raw="{}"))
    return FakeOutput(outputs)


@pytest.fixture()
def context(tmp_path, pipeline_out):
    from pulse_flow.runctx import RunContext, set_context

    art = tmp_path / "run" / "artifacts"
    shutil.copytree(pipeline_out, art)
    set_context(RunContext(art, art.parent, "gemini-3.5-flash-lite"))
    return art


def test_both_crews_have_at_least_three_agents_and_tool_less_narrators(context):
    from pulse_flow.crews.data_analyst.crew import DataAnalystCrew
    from pulse_flow.crews.data_scientist.crew import DataScientistCrew

    analyst = DataAnalystCrew()
    crew1 = analyst.crew()
    scientist = DataScientistCrew()
    crew2 = scientist.crew()
    assert len(crew1.agents) == 6 and len(crew2.agents) == 5
    for task in list(crew1.tasks) + list(crew2.tasks):
        if task.output_pydantic is not None:
            assert not task.tools, f"narrator task {task.name} must not have tools"
        else:
            assert task.tools, f"tool task {task.name} has no tool"
    phase1 = DataScientistCrew().feature_phase()
    phase2 = DataScientistCrew().modeling_phase()
    assert [t.name for t in phase1.tasks] == ["feature_task", "feature_narrative_task"]
    assert [t.name for t in phase2.tasks] == ["training_task", "evaluation_task", "model_card_task"]


def test_yaml_placeholders_are_all_kickoff_inputs():
    allowed = {"data_analyst": COMMON, "data_scientist": COMMON | DIGEST_KEYS}
    for crew, names in allowed.items():
        for name in ("agents.yaml", "tasks.yaml"):
            text = (CREWS / crew / "config" / name).read_text(encoding="utf-8")
            used = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_\-]*)\}", text))
            assert used <= names, f"{crew}/{name}: unknown placeholders {used - names}"


def test_tools_are_read_through_and_never_raise(context):
    from pulse_flow.tools.analyst_tools import draft_contract_schema, ingest_and_clean, run_eda
    from pulse_flow.tools.scientist_tools import build_features, train_models, validate_inputs

    for tool in (
        ingest_and_clean,
        run_eda,
        draft_contract_schema,
        validate_inputs,
        build_features,
        train_models,
    ):
        payload = json.loads(tool.run())
        assert payload["ok"] is True, tool.name
    assert ingest_and_clean.result_as_answer and not validate_inputs.result_as_answer
    (context / "metrics.json").unlink()
    broken = json.loads(train_models.run())
    assert broken["ok"] is False and "metrics.json" in broken["error"]


def test_tool_outputs_give_narrators_formatted_percentages():
    from pulse_flow.tools.analyst_tools import as_percentages

    out = as_percentages(
        {
            "positive_rate": 0.068,
            "bad_rate_on_time": 0.092474,
            "positive_rate_by_group": [{"group": "AL", "positive_rate": 0.214646, "rows": 396}],
            "mix": {"type_share": {"card": 0.198954}, "median_count": 2.0},
            "test": {"roc_auc": 0.7326, "precision_at_top10pct": 0.1213, "lift": 3.17},
            "threshold": 0.055,
        }
    )
    assert out["positive_rate"] == "6.8%" and out["bad_rate_on_time"] == "9.2%"
    assert out["positive_rate_by_group"] == [{"group": "AL", "positive_rate": "21.5%", "rows": 396}]
    assert out["mix"] == {"type_share": {"card": "19.9%"}, "median_count": 2.0}
    assert out["test"] == {"roc_auc": 0.7326, "precision_at_top10pct": "12.1%", "lift": 3.17}
    assert out["threshold"] == 0.055


def test_merge_items_keeps_statements_and_drops_labels_and_repeats():
    from pulse_core.reporting.markdown_ds import merge_items

    base = ["Group bias: rows in the North score higher."]
    extra = [
        "Fairness",
        "group bias: rows in the north score higher.",
        "Recall at the 0.055 threshold is 83.0%, so most positives are flagged.",
    ]
    assert merge_items(base, extra) == [*base, extra[2]]


def test_narrate_uses_llm_outputs_and_falls_back_to_templates(context, monkeypatch, spec):
    from pulse_flow.crews import runner

    monkeypatch.setattr(runner, "kickoff", fake_kickoff)
    ctx = {
        "run_id": "t",
        "llm_model": "gemini-3.5-flash-lite",
        "contract_digest": dict.fromkeys(DIGEST_KEYS, "x"),
    }
    result = runner.narrate("analyst", context, ctx, spec)
    assert result["sources"]["insights.md"] == "llm" and not result["warnings"]
    contract = json.loads((context / "dataset_contract.json").read_text(encoding="utf-8"))
    assert "LLM assumption" in contract["assumptions"]["items"]
    assert contract["assumptions"]["items"][0].startswith("Grain")  # template items kept first
    assert all(n["name"] != "not_a_column" for n in contract["column_notes"]["items"])

    def broken_kickoff(crew, inputs):
        return FakeOutput([FakeTaskOutput(t.name, None, raw="not json") for t in crew.tasks])

    monkeypatch.setattr(runner, "kickoff", broken_kickoff)
    result = runner.narrate("modeling", context, ctx, spec)
    assert result["sources"]["evaluation_report.md"] == "template"
    assert any("evaluation_task" in w for w in result["warnings"])
    card = (context / "model_card.md").read_text(encoding="utf-8")
    assert "## Ethical considerations" in card and "narrative_source: template" in card


def _run_llm(tmp_path, data_dir, monkeypatch, kickoff, run_id):
    from pulse_flow.crews import runner
    from pulse_flow.run import main

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(runner, "kickoff", kickoff)
    code = main(
        [
            "--mode",
            "llm",
            "--spec",
            SPEC_REF,
            "--data-dir",
            str(data_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--run-id",
            run_id,
        ]
    )
    status = json.loads(
        (tmp_path / "runs" / run_id / "run_status.json").read_text(encoding="utf-8")
    )
    return code, status


def test_flow_llm_mode_end_to_end_with_fake_crews(tmp_path, toy_data_dir, monkeypatch):
    code, status = _run_llm(tmp_path, toy_data_dir, monkeypatch, fake_kickoff, "llm-ok")
    assert code == 0 and status["status"] == "success"
    assert status["narrative_sources"]["insights.md"] == "llm"
    assert status["narrative_sources"]["model_card.md"] == "template+llm"
    card = (tmp_path / "artifacts" / "model_card.md").read_text(encoding="utf-8")
    assert "LLM ethics" in card and "Toy data carries no personal information" in card


def test_quota_error_fails_gracefully(tmp_path, toy_data_dir, monkeypatch):
    def quota(crew, inputs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded for model")

    code, status = _run_llm(tmp_path, toy_data_dir, monkeypatch, quota, "llm-quota")
    assert code == 1 and status["stage"] == "run_analyst_stage"
    assert status["errors"][0].startswith("run_analyst_stage: quota exhausted:")
    assert not (tmp_path / "artifacts").exists()


def test_other_crew_errors_degrade_to_templates(tmp_path, toy_data_dir, monkeypatch):
    def flaky(crew, inputs):
        raise RuntimeError("model returned an empty response")

    code, status = _run_llm(tmp_path, toy_data_dir, monkeypatch, flaky, "llm-degraded")
    assert code == 0 and status["status"] == "success"
    assert any(w.startswith("narrative_degraded (analyst)") for w in status["warnings"])
