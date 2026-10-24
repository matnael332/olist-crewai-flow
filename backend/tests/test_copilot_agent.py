"""The Gemini loop with a fake client: tool calls, sources, grounding, limits and failures."""

from types import SimpleNamespace

import pytest
from google.genai import errors

from pulse_copilot.agent import MAX_INPUT_CHARS, Copilot
from pulse_copilot.answer import Conversation
from pulse_copilot.quick import ACTIONS, quick_answer
from pulse_copilot.workspace import Workspace


class FakeModels:
    """Stands in for client.models: runs scripted tool calls like automatic function calling."""

    def __init__(self, calls=(), text="", raises=None):
        self.calls, self.text, self.raises, self.seen = list(calls), text, raises, None

    def generate_content(self, model, contents, config):
        self.seen = SimpleNamespace(model=model, contents=contents, config=config)
        if self.raises:
            raise self.raises
        tools = {tool.__name__: tool for tool in config.tools}
        for name, kwargs in self.calls:
            tools[name](**kwargs)
        usage = SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20, total_token_count=120
        )
        return SimpleNamespace(text=self.text, usage_metadata=usage)


def _client(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(models=FakeModels(**kwargs))


@pytest.fixture(scope="module")
def ws() -> Workspace:
    return Workspace.from_env()


def test_answer_carries_the_text_sources_tool_calls_and_usage(ws):
    client = _client(calls=[("get_run_summary", {})], text="All three gates passed.")
    answer = Copilot(ws, client=client).ask("How did the last run go?")
    assert answer.text == "All three gates passed." and answer.error is None
    assert answer.sources[0] == "run_status.json"
    assert [call.name for call in answer.tool_calls] == ["get_run_summary"]
    assert answer.usage == {"prompt_tokens": 100, "output_tokens": 20, "total_tokens": 120}


def test_replay_outputs_reach_the_answer_for_tables_and_downloads(ws):
    client = _client(calls=[("replay_day", {"day": "2018-07-16"})], text="Done.")
    conversation = Conversation()
    answer = Copilot(ws, client=client).ask("Replay 2018-07-16", conversation=conversation)
    assert answer.outputs == conversation.outputs
    assert answer.outputs[0]["title"] == "Orders purchased on 2018-07-16"


def test_numbers_missing_from_the_tool_results_are_flagged(ws):
    text = "The test ROC-AUC is 0.733 and the riskiest 10% are late 12.1% of the time, not 99.9%."
    client = _client(calls=[("get_model_metrics", {})], text=text)
    answer = Copilot(ws, client=client).ask("Why logistic regression?")
    assert answer.ungrounded == ["99.9%"]


def test_config_uses_the_system_instruction_and_caps_tool_calls(ws):
    client = _client(text="ok")
    Copilot(ws, client=client, max_tool_calls=5).ask("hi")
    config = client.models.seen.config
    assert config.automatic_function_calling.maximum_remote_calls == 5
    assert "read-only" in config.system_instruction
    assert len(config.tools) == 11 and config.max_output_tokens == 1024
    assert client.models.seen.model == "gemini-3.5-flash-lite"


def test_history_is_trimmed_and_roles_are_mapped(ws):
    client = _client(text="ok")
    history = [{"role": "user" if i % 2 == 0 else "assistant", "text": f"m{i}"} for i in range(10)]
    Copilot(ws, client=client, history_messages=6).ask("latest", history=history)
    contents = client.models.seen.contents
    assert len(contents) == 7 and contents[-1].parts[0].text == "latest"
    assert [c.role for c in contents[:2]] == ["user", "model"]


def test_long_messages_are_truncated(ws):
    client = _client(text="ok")
    Copilot(ws, client=client).ask("x" * (MAX_INPUT_CHARS + 500))
    assert len(client.models.seen.contents[-1].parts[0].text) == MAX_INPUT_CHARS


def test_a_quota_error_becomes_a_friendly_answer(ws):
    quota = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "x"}})
    answer = Copilot(ws, client=_client(raises=quota)).ask("hi")
    assert "quota" in answer.text.lower() and answer.error == "ClientError"


def test_without_a_key_the_copilot_is_offline(ws, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    copilot = Copilot(ws)
    assert not copilot.available
    answer = copilot.ask("hi")
    assert answer.offline and answer.error == "no_key"


def test_quick_answers_give_exact_numbers_without_an_llm(ws):
    report = ws.json("validation_report.json")["gates"]["gate_contract"]
    contract_checks = f"{len(report['checks'])}/{len(report['checks'])}"
    answers = {action: quick_answer(action, ws, Conversation()) for action in ACTIONS}
    assert all(answer.offline for answer in answers.values())
    assert f"gate_contract PASS {contract_checks}" in answers["last_run"].text
    assert answers["replay"].text.startswith("238 orders were purchased on 2018-07-16")
    assert answers["replay"].outputs[0]["backtest"]["high"]["orders"] == 42
    assert "0.733" in answers["model"].text and "12.1%" in answers["model"].text
    assert answers["attachment"].text.startswith("I could not score a file")
    with pytest.raises(ValueError):
        quick_answer("dance", ws, Conversation())
