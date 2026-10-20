"""Risk policy: the one place that turns a late-delivery probability into a level and an action.

Used by the Flask API, the batch scorer, the ops copilot and the dashboard, so every channel
shows the same level for the same order.

    high      p >= top-10% cutoff     (the riskiest 10% of test-window orders)
    elevated  p >= decision threshold (tuned on out-of-fold F1)
    normal    otherwise
"""

from __future__ import annotations

from dataclasses import dataclass

LEVELS = ("high", "elevated", "normal")
ACTIONS = {
    "high": "Notify the customer proactively and review a carrier upgrade.",
    "elevated": "Monitor the shipment; contact the customer if tracking stalls.",
    "normal": "No action needed.",
}
CUSTOMER_MESSAGE = (
    "Hi! Your order{ref} is on its way{to}. Deliveries to your region are taking longer than "
    "usual right now, so we are keeping a close eye on it and will update you as soon as "
    "anything changes."
)


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

    @staticmethod
    def customer_message(level: str, order_id: str | None = None, state: str | None = None) -> str:
        """A template draft for high-risk orders (no LLM); empty for the other levels."""
        if level != "high":
            return ""
        ref = f" {order_id}" if order_id else ""
        to = f" to {state}" if state else ""
        return CUSTOMER_MESSAGE.format(ref=ref, to=to)

    def describe(self) -> dict:
        return {
            "elevated_from": round(self.threshold, 6),
            "high_from": round(self.high_from, 6),
        }
