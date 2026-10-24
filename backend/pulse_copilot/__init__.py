"""Ops copilot: a chat assistant for the team that operates the CrewAI agents.

It reads what the agents produced (runs, gates, metrics, insights, documents, the contract) and
scores records through pulse_api. It never runs, changes or deletes anything. The Gemini loop lives
in ``agent``; ``quick`` answers the main questions without any LLM (offline mode).
"""
