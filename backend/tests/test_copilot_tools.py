"""Copilot tools against the committed artifacts: exact numbers, formatted rates, read-only."""

import pytest

from olist_api.replay import orders_for_day, parse_day
from olist_api.risk import LEVELS
from olist_copilot.answer import Conversation
from olist_copilot.tools import build_tools
from olist_copilot.workspace import Workspace, gate_summary


@pytest.fixture(scope="module")
def ws() -> Workspace:
    return Workspace.from_env()


def _tools(ws: Workspace, conversation: Conversation | None = None) -> dict:
    return {f.__name__: f for f in build_tools(ws, conversation or Conversation())}


def test_every_tool_has_a_docstring_for_the_model(ws):
    for name, tool in _tools(ws).items():
        assert tool.__doc__ and len(tool.__doc__) > 40, name


def test_run_summary_matches_the_promoted_run(ws):
    result = _tools(ws)["get_run_summary"]()
    status, manifest = ws.json("run_status.json"), ws.json("run_manifest.json")
    assert result["run_id"] == status["run_id"] and result["status"] == status["status"]
    assert result["published"] is True
    assert result["llm_calls"] == manifest["token_usage"]["successful_requests"]
    gates = gate_summary(ws.json("validation_report.json"))
    assert list(result["gates"]) == ["gate_contract", "gate_required_features", "gate_artifacts"]
    for name, text in result["gates"].items():
        assert text == f"PASS {gates[name]['n_passed']}/{gates[name]['n_total']}"
    assert "validation_report.json" in result["source"]


def test_unknown_run_lists_the_available_runs(ws):
    result = _tools(ws)["get_run_summary"](run_id="no-such-run")
    assert "error" in result and ws.promoted_run_id() in result["available_runs"]


def test_gate_checks_accept_short_names(ws):
    result = _tools(ws)["get_gate_checks"](gate="contract")
    report = ws.json("validation_report.json")["gates"]["gate_contract"]
    assert result["gate"] == "gate_contract" and len(result["checks"]) == len(report["checks"])
    assert "error" in _tools(ws)["get_gate_checks"](gate="quality")


def test_model_metrics_come_from_metrics_json(ws):
    result = _tools(ws)["get_model_metrics"]()
    metrics = ws.json("metrics.json")
    chosen = metrics["candidates"][metrics["selected_model"]]["test"]
    selected = result["candidates"][result["selected_model"]]
    assert selected["test_roc_auc"] == f"{chosen['roc_auc']:.3f}"
    assert (
        selected["late_rate_in_riskiest_10pct"] == f"{chosen['precision_at_top10pct'] * 100:.1f}%"
    )
    assert result["decision_threshold"] == "5.5%"
    assert all(s["late_rate"].endswith("%") for s in result["split"].values())


def test_insights_are_formatted_as_percentages(ws):
    tools = _tools(ws)
    states = tools["get_insight"](topic="states")
    assert all(s["late_rate"].endswith("%") for s in states["late_rate_by_state"])
    assert tools["get_insight"](topic="headline")["headline"]
    assert "topics" in tools["get_insight"](topic="weather")


def test_documents_can_be_read_by_section(ws):
    tools = _tools(ws)
    limits = tools["read_document"](name="model_card", section="Limitations")
    assert "delivered orders" in limits["text"].lower()
    assert "sections" in tools["read_document"](name="model_card", section="No such part")
    assert "documents" in tools["read_document"](name="secret_plans")


def test_contract_summary_and_one_column(ws):
    tools = _tools(ws)
    summary = tools["get_contract"]()
    contract = ws.json("dataset_contract.json")
    assert summary["forbidden_features"] == contract["forbidden_features"]
    column = tools["get_contract"](column="customer_state")
    assert column["allowed_values_total"] == 27


def test_score_order_fills_defaults_and_says_so(ws):
    result = _tools(ws)["score_order"](customer_state="ba", seller_state="SP", total_price=129.9)
    assert result["risk_level"] in LEVELS and result["late_probability"].endswith("%")
    assert len(result["assumed"]) == 4
    rejected = _tools(ws)["score_order"](customer_state="XX", seller_state="SP", total_price=10)
    assert rejected["error"] == "the order does not pass the dataset contract"


def test_replay_day_returns_counts_outcomes_and_a_download(ws):
    conversation = Conversation()
    result = _tools(ws, conversation)["replay_day"](day="2018-07-16")
    assert result["orders_scored"] == sum(result["orders_by_risk_level"].values())
    assert result["what_really_happened"]["all_orders_actually_late"].endswith("%)")
    output = conversation.outputs[0]
    assert output["backtest"]["all"]["orders"] == result["orders_scored"]
    assert len(output["result"].predictions) == result["orders_scored"]
    outside = _tools(ws)["replay_day"](day="2018-01-05")
    assert outside["error"] == "out_of_window"


def test_attached_file_is_scored_with_partial_success(ws):
    records, _ = orders_for_day(ws.clean(), parse_day("2018-07-16"))
    rows = [*records[:3], {**records[3], "customer_state": "XX"}]
    conversation = Conversation(attachment={"name": "orders.csv", "records": rows, "ignored": []})
    result = _tools(ws, conversation)["score_attached_file"]()
    assert result["summary"]["scored"] == 3 and result["summary"]["rejected"] == 1
    assert conversation.outputs[0]["title"] == "Scored orders.csv"
    assert "error" in _tools(ws)["score_attached_file"]()


def test_log_search_and_run_list(ws):
    tools = _tools(ws)
    gates = tools["search_log"](pattern="GATE gate_contract RESULT")
    assert gates["matches"] == 1 and "PASS" in gates["lines"][0]
    assert any(run["run_id"] == ws.promoted_run_id() for run in tools["list_runs"]()["runs"])


def test_tools_never_write_anything(ws):
    before = {p: p.stat().st_mtime_ns for p in ws.artifacts.rglob("*") if p.is_file()}
    tools = _tools(ws)
    tools["get_run_summary"]()
    tools["get_model_metrics"]()
    tools["replay_day"](day="2018-07-16")
    tools["score_order"](customer_state="BA", seller_state="SP", total_price=50.0)
    after = {p: p.stat().st_mtime_ns for p in ws.artifacts.rglob("*") if p.is_file()}
    assert before == after


def test_gate_summary_counts_checks():
    report = {"gates": {"g": {"passed": False, "checks": [{"passed": True}, {"passed": False}]}}}
    assert gate_summary(report)["g"] | {"checks": []} == {
        "passed": False,
        "n_passed": 1,
        "n_total": 2,
        "checks": [],
    }
