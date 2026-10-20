"""Narrative schemas shared by the LLM narrators (CrewAI output_pydantic) and the templates.

Kept flat (strings and lists of strings, one nested level at most) so Gemini's JSON mode returns
them reliably. Every artifact can be rendered from a template instance (deterministic mode) or from
an LLM instance (llm mode) with the same renderer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CleaningNarrative(BaseModel):
    summary: str = Field(description="Two to four sentences on what the cleaning did and why")
    issues_found: list[str] = Field(description="Data quality issues found, with exact counts")
    decisions: list[str] = Field(description="Cleaning decisions and their rationale")
    caveats: list[str] = Field(description="What remains imperfect after cleaning")


class Finding(BaseModel):
    title: str
    evidence: str = Field(description="The exact numbers from the EDA summary behind the claim")
    business_meaning: str


class Insights(BaseModel):
    headline: str = Field(description="One sentence: the most important thing that happened")
    key_findings: list[Finding]
    recommendations: list[str]
    data_caveats: list[str]
    modeling_implications: list[str]


class ColumnNote(BaseModel):
    name: str
    note: str


class ContractNarrative(BaseModel):
    assumptions: list[str]
    business_rules: list[str]
    column_notes: list[ColumnNote]
    known_limitations: list[str]


class FeatureNarrative(BaseModel):
    contract_compliance: str = Field(
        description="States contract version, row count, required and forbidden feature counts"
    )
    leakage_statement: str
    feature_groups: list[str] = Field(description="Feature groups and why each matters")
    concerns: list[str]


class EvaluationNarrative(BaseModel):
    executive_summary: str
    comparison_summary: str
    strengths: list[str]
    weaknesses: list[str]
    threshold_recommendation: str
    next_steps: list[str]


class ModelCardNarrative(BaseModel):
    purpose: str
    intended_users: str
    out_of_scope: list[str]
    training_data_summary: str
    limitations: list[str]
    ethical_considerations: list[str]
    monitoring_recommendations: list[str]
