"""Process-level run context read by the zero-argument crew tools and the crew classes."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunContext:
    artifacts_dir: Path
    run_dir: Path
    model: str

    def crew_log(self, name: str) -> str:
        path = self.run_dir / "crew_logs"
        path.mkdir(parents=True, exist_ok=True)
        return str(path / f"{name}.json")


_current: RunContext | None = None


def set_context(context: RunContext) -> None:
    global _current
    _current = context


def get_context() -> RunContext:
    if _current is None:
        raise RuntimeError("run context not set (the Flow sets it before kicking off a crew)")
    return _current
