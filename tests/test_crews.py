"""Crew structure, YAML placeholders, tools and the LLM path with a fake kickoff (no network)."""

import json
import re
import shutil
from pathlib import Path

import pytest

from olist_core.narratives import (
    CleaningNarrative,
    ColumnNote,
    ContractNarrative,
    EvaluationNarrative,
    FeatureNarrative,
    Finding,
    Insights,
    ModelCardNarrative,
)

CREWS = Path(__file__).parents[1] / "src" / "olist_flow" / "crews"
FIXTURE = Path(__file__).parent / "fixtures" / "olist_mini"
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
            ColumnNote(name="promised_days", note="LLM note"),
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
        limitations=["LLM limitation"],
        ethical_considerations=["LLM ethics"],
        monitoring_recommendations=["LLM monitoring"],
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
    from olist_flow.runctx import RunContext, set_context

    art = tmp_path / "run" / "artifacts"
    shutil.copytree(pipeline_out, art)
    set_context(RunContext(art, art.parent, "gemini-3.5-flash-lite"))
    return art


def test_both_crews_have_at_least_three_agents_and_tool_less_narrators(context):
    from olist_flow.crews.data_analyst.crew import DataAnalystCrew
    from olist_flow.crews.data_scientist.crew import DataScientistCrew

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
    allowed = {
        "data_analyst": {"run_id", "dataset_name"},
        "data_scientist": {"run_id"} | DIGEST_KEYS,
    }
    for crew, names in allowed.items():
        for name in ("agents.yaml", "tasks.yaml"):
            text = (CREWS / crew / "config" / name).read_text(encoding="utf-8")
            used = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_\-]*)\}", text))
            assert used <= names, f"{crew}/{name}: unknown placeholders {used - names}"


def test_tools_are_read_through_and_never_raise(context):
    from olist_flow.tools.analyst_tools import draft_contract_schema, ingest_and_clean, run_eda
    from olist_flow.tools.scientist_tools import build_features, train_models, validate_inputs

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


def test_narrate_uses_llm_outputs_and_falls_back_to_templates(context, monkeypatch):
    from olist_flow.crews import runner

    monkeypatch.setattr(runner, "kickoff", fake_kickoff)
    ctx = {
        "run_id": "t",
        "llm_model": "gemini-3.5-flash-lite",
        "contract_digest": dict.fromkeys(DIGEST_KEYS, "x"),
    }
    result = runner.narrate("analyst", context, ctx)
    assert result["sources"]["insights.md"] == "llm" and not result["warnings"]
    contract = json.loads((context / "dataset_contract.json").read_text(encoding="utf-8"))
    assert "LLM assumption" in contract["assumptions"]["items"]
    assert contract["assumptions"]["items"][0].startswith("Grain")  # template items kept first
    assert all(n["name"] != "not_a_column" for n in contract["column_notes"]["items"])

    def broken_kickoff(crew, inputs):
        return FakeOutput([FakeTaskOutput(t.name, None, raw="not json") for t in crew.tasks])

    monkeypatch.setattr(runner, "kickoff", broken_kickoff)
    result = runner.narrate("modeling", context, ctx)
    assert result["sources"]["evaluation_report.md"] == "template"
    assert any("evaluation_task" in w for w in result["warnings"])
    card = (context / "model_card.md").read_text(encoding="utf-8")
    assert "## Ethical considerations" in card and "narrative_source: template" in card


def _run_llm(tmp_path, monkeypatch, kickoff, run_id):
    from olist_flow.crews import runner
    from olist_flow.run import main

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(runner, "kickoff", kickoff)
    code = main(
        [
            "--mode",
            "llm",
            "--data-dir",
            str(FIXTURE),
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


def test_flow_llm_mode_end_to_end_with_fake_crews(tmp_path, monkeypatch):
    code, status = _run_llm(tmp_path, monkeypatch, fake_kickoff, "llm-ok")
    assert code == 0 and status["status"] == "success"
    assert status["narrative_sources"]["insights.md"] == "llm"
    assert status["narrative_sources"]["model_card.md"] == "template+llm"
    card = (tmp_path / "artifacts" / "model_card.md").read_text(encoding="utf-8")
    assert "LLM ethics" in card and "Regional bias" in card


def test_quota_error_fails_gracefully(tmp_path, monkeypatch):
    def quota(crew, inputs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded for model")

    code, status = _run_llm(tmp_path, monkeypatch, quota, "llm-quota")
    assert code == 1 and status["stage"] == "run_analyst_stage"
    assert status["errors"][0].startswith("run_analyst_stage: quota exhausted:")
    assert not (tmp_path / "artifacts").exists()


def test_other_crew_errors_degrade_to_templates(tmp_path, monkeypatch):
    def flaky(crew, inputs):
        raise RuntimeError("model returned an empty response")

    code, status = _run_llm(tmp_path, monkeypatch, flaky, "llm-degraded")
    assert code == 0 and status["status"] == "success"
    assert any(w.startswith("narrative_degraded (analyst)") for w in status["warnings"])
