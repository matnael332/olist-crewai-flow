"""Flow state: small, JSON-friendly, every field defaulted (CrewAI builds it with no arguments)."""

from typing import Literal

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    run_id: str = ""
    run_dir: str = ""
    mode: Literal["llm", "deterministic"] = "deterministic"
    inject_fault: str = ""
    seed: int = 42
    raw_dir: str = "data/raw"
    lookups_dir: str = "data/lookups"
    artifacts_dir: str = "artifacts"
    promote: bool = True
    llm_model: str = ""
    start_at: str = ""
    from_run: str = ""
    spec_ref: str = ""
    spec_params: dict = Field(default_factory=dict)
    language: Literal["en", "he"] = "en"

    stage: str = "init"
    stages_done: list[str] = Field(default_factory=list)
    status: Literal["running", "success", "failed", "promotion_failed"] = "running"
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    gates: dict[str, dict] = Field(default_factory=dict)
    contract_digest: dict[str, str] = Field(default_factory=dict)
    cleaning_counts: dict[str, float] = Field(default_factory=dict)
    training_summary: dict = Field(default_factory=dict)
    narrative_sources: dict[str, str] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
