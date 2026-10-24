"""CrewAI event listener -> the run's "pulse" logger (crew, task, tool and LLM call ledger).

Handlers run in CrewAI's event thread pool, so they only call logging (thread-safe) and log
metadata, never prompts, responses or object reprs (which could contain the API key).
"""

import threading

from crewai.events import BaseEventListener
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from crewai.events.types.task_events import TaskCompletedEvent, TaskFailedEvent
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)

from pulse_core.logging_setup import get_logger

log = get_logger("crew")


def _short(text: object, limit: int = 300) -> str:
    value = str(text).replace("\n", " ")
    return value if len(value) <= limit else value[:limit] + "..."


class FileEventLogger(BaseEventListener):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = 0
        super().__init__()

    def setup_listeners(self, crewai_event_bus) -> None:
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def crew_started(source, event):
            log.info("CREW %s START", event.crew_name)

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def crew_done(source, event):
            log.info("CREW %s END tokens=%s", event.crew_name, event.total_tokens)

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def crew_failed(source, event):
            log.error("CREW %s FAILED %s", event.crew_name, _short(event.error))

        @crewai_event_bus.on(TaskCompletedEvent)
        def task_done(source, event):
            log.info("TASK %s DONE agent=%s", event.output.name, event.output.agent)

        @crewai_event_bus.on(TaskFailedEvent)
        def task_failed(source, event):
            log.error("TASK FAILED %s", _short(event.error))

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def tool_started(source, event):
            log.info("TOOL %s START agent=%s", event.tool_name, event.agent_role)

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def tool_done(source, event):
            log.info("TOOL %s DONE", event.tool_name)

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def tool_error(source, event):
            log.error("TOOL %s ERROR %s", event.tool_name, _short(event.error))

        @crewai_event_bus.on(LLMCallStartedEvent)
        def llm_started(source, event):
            with self._lock:
                self.llm_calls += 1
                number = self.llm_calls
            log.info("LLM call #%d START model=%s", number, event.model)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def llm_done(source, event):
            usage = event.usage or {}
            tokens = usage.get("total_tokens") or usage.get("total_token_count")
            log.info("LLM call DONE model=%s tokens=%s", event.model, tokens)

        @crewai_event_bus.on(LLMCallFailedEvent)
        def llm_failed(source, event):
            log.error("LLM call FAILED model=%s %s", event.model, _short(event.error))


event_logger = FileEventLogger()
