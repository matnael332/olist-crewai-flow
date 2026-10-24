"""The Gemini loop: one question in, a short grounded answer out.

google-genai's automatic function calling runs the tools (at most ``max_tool_calls`` per turn).
Every call is recorded, so the Sources line, the tables and the downloads come from what the
tools returned, never from the model's text. The key comes from GEMINI_API_KEY only.
"""

from __future__ import annotations

import os

from pulse_copilot.answer import Answer, Conversation, Recorder
from pulse_copilot.grounding import ungrounded
from pulse_copilot.prompts import SYSTEM_INSTRUCTION
from pulse_copilot.tools import build_tools
from pulse_copilot.workspace import Workspace

DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_INPUT_CHARS = 2000


def friendly_error(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if code == 429:
        return (
            "The Gemini quota is used up for now. The quick replies above still work without "
            "Gemini; or ask again in a minute."
        )
    if code in (401, 403):
        return "The Gemini key was rejected. Check GEMINI_API_KEY in the app's secrets."
    if "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower():
        return "Gemini did not answer in time. Try again, or use a quick reply."
    return f"Something went wrong while asking Gemini ({type(exc).__name__}). Try a quick reply."


class Copilot:
    def __init__(
        self,
        workspace: Workspace | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client=None,
        max_tool_calls: int = 5,
        history_messages: int = 6,
    ) -> None:
        self.workspace = workspace or Workspace.from_env()
        self.model = model or os.getenv("COPILOT_MODEL", DEFAULT_MODEL)
        self.max_tool_calls = max_tool_calls
        self.history_messages = history_messages
        self._key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self._client = client

    @property
    def available(self) -> bool:
        return self._client is not None or bool(self._key)

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            retry = types.HttpRetryOptions(
                attempts=3,
                initial_delay=2.0,
                max_delay=20.0,
                exp_base=2.0,
                jitter=1.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            )
            self._client = genai.Client(
                api_key=self._key,
                http_options=types.HttpOptions(timeout=60_000, retry_options=retry),
            )
        return self._client

    def _config(self, tools: list):
        from google.genai import types

        lite = "flash-lite" in self.model
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=self.max_tool_calls
            ),
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL if lite else types.ThinkingLevel.LOW,
                include_thoughts=False,
            ),
            max_output_tokens=1024,
        )

    def _contents(self, history: list[dict], message: str) -> list:
        from google.genai import types

        contents = []
        for turn in history[-self.history_messages :]:
            role = "model" if turn.get("role") in {"assistant", "model"} else "user"
            if turn.get("text"):
                contents.append(types.Content(role=role, parts=[types.Part(text=turn["text"])]))
        contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
        return contents

    def ask(
        self,
        message: str,
        history: list[dict] | None = None,
        conversation: Conversation | None = None,
    ) -> Answer:
        conversation = conversation or Conversation()
        mark = len(conversation.outputs)
        recorder = Recorder()
        message = message.strip()[:MAX_INPUT_CHARS]
        if not self.available:
            return Answer(
                text="The copilot is offline (no GEMINI_API_KEY). The quick replies still work.",
                offline=True,
                error="no_key",
            )
        tools = [recorder.wrap(tool) for tool in build_tools(self.workspace, conversation)]
        try:
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=self._contents(history or [], message),
                config=self._config(tools),
            )
            text = (response.text or "").strip()
            usage = getattr(response, "usage_metadata", None)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a readable answer
            return Answer(
                text=friendly_error(exc),
                sources=recorder.sources(),
                tool_calls=recorder.calls,
                outputs=conversation.outputs[mark:],
                error=type(exc).__name__,
            )
        if not text:
            text = "I could not put an answer together from the tools. Try rephrasing."
        return Answer(
            text=text,
            sources=recorder.sources(),
            tool_calls=recorder.calls,
            outputs=conversation.outputs[mark:],
            ungrounded=ungrounded(text, [call.result for call in recorder.calls]),
            usage={
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            },
        )
