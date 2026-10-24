"""The LLM mapper: Gemini proposes table and column pairs from names and kinds only.

Its proposal is merged with the heuristic one: a pair both agree on is certain; a pair only one
of them found is kept with its own confidence; a column the two map differently keeps the
heuristic choice when it is an exact name match, else the LLM's, and is flagged for review.
Without a key (or with ``--no-llm``) the heuristic proposal stands alone.
"""

from __future__ import annotations

import json
import os

from pydantic import BaseModel

from pulse_ingest.contract import IngestContract
from pulse_ingest.llm_payloads import mapping_payload
from pulse_ingest.mapping import ColumnSource, Mapping, TableMapping
from pulse_ingest.profile import SourceTable

DEFAULT_MODEL = "gemini-3.5-flash-lite"
LLM_CONFIDENCE = 0.75
INSTRUCTION = (
    "You map a customer's data export onto a fixed target schema. You see only table names, "
    "column names and inferred kinds (no data). Names may be in Hebrew or English. For each "
    "target table pick the one source table that holds it, if any, and pair target columns with "
    "source columns of a compatible kind. Never pair a column twice; leave out what has no good "
    "match. A date column and a separate time column may together fill one datetime target: give "
    "the date column as source_column and the time column as time_column."
)


class ColumnPair(BaseModel):
    target_column: str
    source_column: str
    time_column: str | None = None


class TablePair(BaseModel):
    target_table: str
    source_table: str
    columns: list[ColumnPair]


class LlmMapping(BaseModel):
    tables: list[TablePair]


def ask(profile: list[SourceTable], contract: IngestContract, client=None, model=None):
    """The LLM's pairs and its token usage; ``client`` is injectable for tests."""
    from google.genai import types

    if client is None:
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    response = client.models.generate_content(
        model=model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL).removeprefix("gemini/"),
        contents=json.dumps(mapping_payload(profile, contract), ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTION,
            response_mime_type="application/json",
            response_schema=LlmMapping,
            temperature=0.0,
        ),
    )
    parsed = response.parsed
    if not isinstance(parsed, LlmMapping):
        parsed = LlmMapping.model_validate_json(response.text)
    usage = getattr(response, "usage_metadata", None)
    tokens = {
        "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
    }
    return parsed, tokens


def merge(
    mapping: Mapping, llm: LlmMapping, profile: list[SourceTable], contract: IngestContract
) -> Mapping:
    """Fold the LLM's pairs into the heuristic mapping (in place, also returned)."""
    sources = {s.name: s for s in profile}
    targets = {t.name for t in contract.tables}
    for pair in llm.tables:
        if pair.target_table not in targets or pair.source_table not in sources:
            continue
        source = sources[pair.source_table]
        names = {c.name for c in source.columns}
        table = mapping.tables.get(pair.target_table)
        if table is None:
            if any(pair.source_table in t.sources for t in mapping.tables.values()):
                mapping.review.append(
                    f"{pair.target_table}: the LLM suggests {pair.source_table}, already used"
                )
                continue
            table = TableMapping(sources=[pair.source_table], confidence=LLM_CONFIDENCE)
            mapping.tables[pair.target_table] = table
            mapping.review.append(f"{pair.target_table}: table found by the LLM only")
        elif table.sources != [pair.source_table]:
            mapping.review.append(
                f"{pair.target_table}: heuristics chose {table.sources}, the LLM "
                f"{pair.source_table}; kept the heuristic choice"
            )
            continue
        target = contract.table(pair.target_table)
        for col in pair.columns:
            if col.source_column not in names or col.target_column not in {
                c.name for c in target.columns
            }:
                continue
            current = table.columns.get(col.target_column)
            if current and current.column == col.source_column:
                current.origin, current.confidence = "both", 1.0
            elif current and current.column:
                if current.confidence < 1.0:
                    mapping.review.append(
                        f"{pair.target_table}.{col.target_column}: heuristics chose "
                        f"{current.column}, the LLM {col.source_column}; took the LLM's"
                    )
                    table.columns[col.target_column] = _from_llm(source, col)
            elif current is None or not (current.constant or current.generate):
                table.columns[col.target_column] = _from_llm(source, col)
    return mapping


def _from_llm(source: SourceTable, col: ColumnPair) -> ColumnSource:
    profile = source.column(col.source_column)
    time = None
    if col.time_column and col.time_column in {c.name for c in source.columns}:
        time = source.column(col.time_column)
    return ColumnSource(
        column=col.source_column,
        format=profile.format,
        time_column=time.name if time else None,
        time_format=time.format if time else None,
        origin="llm",
        confidence=LLM_CONFIDENCE,
    )
