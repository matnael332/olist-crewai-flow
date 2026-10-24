"""The ops copilot's offline path on the toy artifacts: tools, quick replies and attachments."""

import pytest

from pulse_api.batch import frame_to_records, read_records_csv
from pulse_copilot.answer import Conversation
from pulse_copilot.quick import ACTIONS, quick_answer
from pulse_copilot.tools import build_tools
from pulse_copilot.workspace import Workspace


@pytest.fixture()
def workspace(tmp_path, pipeline_out):
    return Workspace(tmp_path, pipeline_out, tmp_path / "runs")


def test_tools_read_the_artifacts_and_cite_sources(workspace):
    tools = {f.__name__: f for f in build_tools(workspace, Conversation())}
    assert len(tools) == 8
    contract = tools["get_contract"]()
    assert contract["required_features"] == ["x_num", "x_cat"]
    assert contract["source"] == ["dataset_contract.json"]
    metrics = tools["get_model_metrics"]()
    assert metrics["selected_model"] in {"logistic regression", "gradient boosting"}
    assert "positive_rate_in_riskiest_10pct" in next(iter(metrics["candidates"].values()))
    card = tools["read_document"]("model_card", "Limitations")
    assert card["section"] == "Limitations" and card["text"]
    assert tools["get_gate_checks"]("nope")["error"].startswith("unknown gate")


def test_quick_replies_work_without_an_llm(workspace, bundle):
    conversation = Conversation()
    csv = b"row_id,x_num,x_cat,x_flag,x_num_sq\nr1,2.0,a,0,4.0\nr2,-2.0,c,1,4.0\n"
    records, ignored = frame_to_records(bundle, read_records_csv(bundle, csv))
    conversation.attachment = {"name": "toy.csv", "records": records, "ignored": ignored}
    for action in ACTIONS:
        answer = quick_answer(action, workspace, conversation)
        assert answer.offline and answer.text, action
    scored = quick_answer("attachment", workspace, conversation)
    assert "2 of 2 rows scored" in scored.text
    assert scored.outputs and scored.outputs[0]["result"].summary["scored"] == 2
    assert "model.joblib" in scored.sources


def test_workspace_reads_the_customer_folders_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    workspace = Workspace.from_env()
    assert workspace.artifacts == tmp_path / "artifacts"
    assert workspace.runs == tmp_path / "runs"
