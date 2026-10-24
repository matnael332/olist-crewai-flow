"""What the ingestion must produce: the target tables and columns, declared by the dataset spec.

A target column is filled from the customer's data by the mapping, or else:
- ``generate`` (a row key the export lacks: a stable row number),
- its ``default`` (a constant),
- empty (an optional column).

``id_kind`` marks identifier columns: ``account`` ids get sequential pseudonyms the customer can
look up (the entity map), any other kind gets a keyed hash; ``row`` keys are left as they are.
``values`` lists the canonical values of a semantic enum (the cleaning compares against some of
them); ``free_values`` columns keep the customer's own categories. ``extra`` columns are not part
of the raw layer: the account name only reaches the entity map.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True, kw_only=True)
class TargetColumn:
    name: str
    kind: str  # string | datetime | int | float | bool
    required: bool = False
    id_kind: str | None = None  # account | employee | site | device | row
    generate: bool = False
    default: object = None
    values: tuple[str, ...] = ()
    value_synonyms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    free_values: bool = False
    synonyms: tuple[str, ...] = ()
    extra: bool = False


@dataclass(frozen=True, kw_only=True)
class TargetTable:
    name: str
    file: str
    group: str
    columns: tuple[TargetColumn, ...]
    primary_key: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()

    def column(self, name: str) -> TargetColumn:
        return next(c for c in self.columns if c.name == name)

    @property
    def raw_columns(self) -> tuple[TargetColumn, ...]:
        return tuple(c for c in self.columns if not c.extra)


class PreflightError(ValueError):
    """The export cannot be modelled; the message says what is missing."""


@dataclass(frozen=True, kw_only=True)
class IngestContract:
    name: str
    tables: tuple[TargetTable, ...]
    core_groups: tuple[str, ...]
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    # (typed target tables, present groups) -> {"spec_params": ..., "report": ...}; raises
    # PreflightError when the export cannot be modelled. ``spec_ref`` is the factory the
    # spec_params are for (printed in the pipeline command).
    preflight: Callable[[dict, tuple[str, ...]], dict] | None = None
    spec_ref: str = ""

    def groups_present(self, tables: list[str]) -> tuple[str, ...]:
        """Groups whose every table is in ``tables``, in contract order."""
        groups = dict.fromkeys(t.group for t in self.tables)
        return tuple(
            g for g in groups if all(t.name in tables for t in self.tables if t.group == g)
        )

    def table(self, name: str) -> TargetTable:
        return next(t for t in self.tables if t.name == name)

    def raw_schema(self, tables: list[str]) -> dict:
        """The raw_schema.json of the given target tables, in contract order."""
        return {
            "schema_version": 1,
            "datetime_format": self.datetime_format,
            "renames": {},
            "tables": {
                t.name: {
                    "group": t.group,
                    "file": t.file,
                    "primary_key": list(t.primary_key),
                    "columns": {c.name: c.kind for c in t.raw_columns},
                }
                for t in self.tables
                if t.name in tables
            },
        }
