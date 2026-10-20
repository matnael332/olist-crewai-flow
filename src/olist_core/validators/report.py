"""GateReport: the result type of every validation gate.

A gate records named checks and never raises; the Flow decides what a failed report means.
Pattern ported from the earlier attempt's ValidationReport (AI_DEV_FINAL/backend/pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateReport:
    gate: str
    checks: list[dict] = field(default_factory=list)

    def record(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})
        return bool(passed)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check["passed"] for check in self.checks)

    @property
    def errors(self) -> list[str]:
        return [f"{self.gate}: {c['detail']}" for c in self.checks if not c["passed"]]

    @property
    def summary(self) -> str:
        failed = sum(not c["passed"] for c in self.checks)
        status = "PASS" if self.passed else "FAIL"
        return f"{status} ({len(self.checks) - failed}/{len(self.checks)} checks)"

    def to_dict(self) -> dict:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "summary": self.summary,
            "checks": list(self.checks),
            "errors": self.errors,
        }
