"""Crew kickoff indirection used by the Flow.

narrate(phase, ...) kicks off the crew for a phase and re-renders the narrative artifacts with the
agents' structured outputs. Deterministic artifacts already exist (run-then-narrate), so a missing
or invalid narrative only degrades that narrative to its template. Tests monkeypatch `kickoff`.
"""

import json
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from olist_core import stages
from olist_core.logging_setup import get_logger
from olist_core.narratives import (
    CleaningNarrative,
    ContractNarrative,
    EvaluationNarrative,
    FeatureNarrative,
    Insights,
    ModelCardNarrative,
)
from olist_flow.llm import resolve_model
from olist_flow.runctx import RunContext, set_context

log = get_logger("crew")
DATASET_NAME = "Olist Brazilian E-Commerce"
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def kickoff(crew, inputs: dict):
    """The only place a crew is kicked off (monkeypatched in tests)."""
    from olist_flow import events  # noqa: F401  (registers the event logger once)

    return crew.kickoff(inputs=inputs)


def extract(output, task_name: str, model: type[BaseModel], warnings: list[str]):
    """The task's pydantic output, or a parse of its raw JSON, or None (template fallback)."""
    for task_output in getattr(output, "tasks_output", []) or []:
        if task_output.name != task_name:
            continue
        if isinstance(task_output.pydantic, model):
            return task_output.pydantic
        match = _JSON_BLOCK.search(task_output.raw or "")
        if match:
            try:
                return model.model_validate(json.loads(match.group(0)))
            except (ValueError, ValidationError):
                pass
        break
    warnings.append(f"narrative_degraded ({task_name}): no valid {model.__name__}; template used")
    return None


def usage_of(output) -> dict[str, int]:
    usage = getattr(output, "token_usage", None)
    if usage is None:
        return {}
    keys = ("total_tokens", "prompt_tokens", "completion_tokens", "successful_requests")
    return {k: int(getattr(usage, k, 0) or 0) for k in keys}


def narrate(phase: str, artifacts_dir: Path, context: dict) -> dict:
    """Returns {"sources": {...}, "warnings": [...], "token_usage": {...}}."""
    art = Path(artifacts_dir)
    model = resolve_model(context.get("llm_model") or None)
    set_context(RunContext(artifacts_dir=art, run_dir=art.parent, model=model))
    warnings: list[str] = []
    digest = {k: str(v) for k, v in (context.get("contract_digest") or {}).items()}

    if phase == "analyst":
        from olist_flow.crews.data_analyst.crew import DataAnalystCrew

        instance = DataAnalystCrew()
        crew = instance.crew()
        log.info("CREW data_analyst agents=%s model=%s", [a.role for a in crew.agents], model)
        output = kickoff(crew, {"run_id": context["run_id"], "dataset_name": DATASET_NAME})
        sources = stages.assemble_analyst_narratives(
            art,
            extract(output, "cleaning_narrative_task", CleaningNarrative, warnings),
            extract(output, "insights_task", Insights, warnings),
            extract(output, "contract_narrative_task", ContractNarrative, warnings),
        )
    elif phase == "features":
        from olist_flow.crews.data_scientist.crew import DataScientistCrew

        instance = DataScientistCrew()
        crew = instance.feature_phase()
        log.info("CREW data_scientist phase=1/2 agents=%s", [a.role for a in crew.agents])
        output = kickoff(crew, {"run_id": context["run_id"], **digest})
        narrative = extract(output, "feature_narrative_task", FeatureNarrative, warnings)
        sources = {"feature_spec.json": stages.assemble_feature_narrative(art, narrative)}
    elif phase == "modeling":
        from olist_flow.crews.data_scientist.crew import DataScientistCrew

        instance = DataScientistCrew()
        crew = instance.modeling_phase()
        log.info("CREW data_scientist phase=2/2 agents=%s", [a.role for a in crew.agents])
        output = kickoff(crew, {"run_id": context["run_id"], **digest})
        sources = stages.assemble_modeling_narratives(
            art,
            extract(output, "evaluation_task", EvaluationNarrative, warnings),
            extract(output, "model_card_task", ModelCardNarrative, warnings),
        )
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return {"sources": sources, "warnings": warnings, "token_usage": usage_of(output)}
