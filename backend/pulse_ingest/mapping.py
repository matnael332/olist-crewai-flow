"""The mapping document: how a customer's tables become the target tables.

It is the reviewable artefact of the ingestion. ``propose`` writes it with ``status: proposed``;
a person approves it (the CLI ``approve`` command now, the website later) and only an approved
mapping is applied. It names columns and the customer's category spellings, never row values.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Origin = Literal["heuristic", "llm", "both", "user", "previous", "default"]


class ColumnSource(BaseModel):
    column: str | None = None  # the source column
    time_column: str | None = None  # a separate time column joined to ``column`` (a date)
    format: str | None = None  # datetime/date format of ``column``
    time_format: str | None = None
    constant: str | None = None
    generate: bool = False  # a stable row number (row keys the export lacks)
    origin: Origin = "heuristic"
    confidence: float = 0.0


class TableMapping(BaseModel):
    sources: list[str]  # source tables (union groups) concatenated into this target table
    columns: dict[str, ColumnSource] = Field(default_factory=dict)
    confidence: float = 0.0


class Mapping(BaseModel):
    contract: str
    version: int = 1
    status: Literal["proposed", "approved"] = "proposed"
    fingerprint: str = ""
    tables: dict[str, TableMapping] = Field(default_factory=dict)
    # "table.column" -> {the customer's spelling: canonical value}
    value_maps: dict[str, dict[str, str]] = Field(default_factory=dict)
    # source table -> its columns the mapping leaves out (never stored after normalisation)
    ignored: dict[str, list[str]] = Field(default_factory=dict)
    review: list[str] = Field(default_factory=list)  # what a person should look at

    def summary(self) -> list[str]:
        lines = [f"mapping v{self.version} ({self.status}), structure {self.fingerprint}"]
        for name, table in self.tables.items():
            mapped = sum(1 for c in table.columns.values() if c.column or c.constant or c.generate)
            lines.append(
                f"  {name:<20} <- {', '.join(table.sources)}  "
                f"({mapped}/{len(table.columns)} columns, confidence {table.confidence:.2f})"
            )
        for source, columns in self.ignored.items():
            if columns:
                lines.append(f"  ignored in {source}: {', '.join(columns)}")
        lines += [f"  review: {item}" for item in self.review]
        return lines
