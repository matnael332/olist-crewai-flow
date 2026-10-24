"""Crew tools: zero-argument, idempotent read-through views onto the deterministic pipeline.

The Flow computes every artifact before kicking off a crew ("run-then-narrate"); the tools let the
agents inspect those results and return them verbatim (result_as_answer) to the narrator agents.
Tools never raise: failures come back as {"ok": false, "error": ...}.
"""
