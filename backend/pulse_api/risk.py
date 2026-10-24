"""Risk policy: the one place that turns a probability into a level and an action.

Used by the Flask API, the batch scorer, the ops copilot and the dashboard, so every channel
shows the same level for the same record.

    high      p >= top-10% cutoff     (the riskiest 10% of test-window rows)
    elevated  p >= decision threshold (tuned on out-of-fold F1)
    normal    otherwise
"""

from __future__ import annotations

from dataclasses import dataclass

LEVELS = ("high", "elevated", "normal")
ACTIONS = {
    "high": "Act now: review this record and contact the owner.",
    "elevated": "Monitor: re-check at the next refresh and act if the score rises.",
    "normal": "No action needed.",
}


@dataclass(frozen=True)
class RiskPolicy:
    threshold: float
    top10_cutoff: float

    @property
    def high_from(self) -> float:
        return max(self.threshold, self.top10_cutoff)

    def level(self, probability: float) -> str:
        if probability >= self.high_from:
            return "high"
        if probability >= self.threshold:
            return "elevated"
        return "normal"

    @staticmethod
    def action(level: str) -> str:
        return ACTIONS[level]

    def describe(self) -> dict:
        return {
            "elevated_from": round(self.threshold, 6),
            "high_from": round(self.high_from, 6),
        }
