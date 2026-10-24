"""Profile the source tables: what each column looks like, without keeping what it holds.

For every column: the inferred kind (int, float, bool, datetime, date, time or string), the
datetime format that parses all of it, the null rate and whether it is unique. Source tables with
the same header (a file per month, a table split over sheets) form one union group. The profile
holds names, kinds and rates only, never a cell value, so it is safe to show and to hand to the
LLM mapper; the distinct values of enum columns are read from the staged tables when needed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath

import pandas as pd

DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
)
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y")
TIME_FORMATS = ("%H:%M:%S", "%H:%M")
TRUE_WORDS = ("true", "yes", "y", "כן", "v", "✓")
FALSE_WORDS = ("false", "no", "n", "לא", "")
SAMPLE_ROWS = 5000


@dataclass
class ColumnProfile:
    name: str
    kind: str
    format: str | None = None
    null_rate: float = 0.0
    unique: bool = False


@dataclass
class SourceTable:
    name: str
    members: list[str]
    rows: int
    columns: list[ColumnProfile] = field(default_factory=list)

    def column(self, name: str) -> ColumnProfile:
        return next(c for c in self.columns if c.name == name)


def _parses(values: pd.Series, fmt: str) -> bool:
    return bool(pd.to_datetime(values, format=fmt, errors="coerce").notna().all())


def infer_kind(values: pd.Series) -> tuple[str, str | None]:
    """The kind of a column of strings (empty strings are nulls) and its format, if any."""
    present = values[values.str.strip() != ""].str.strip()
    if present.empty:
        return "string", None
    sample = present.iloc[:SAMPLE_ROWS] if len(present) > SAMPLE_ROWS else present
    lowered = set(sample.str.lower().unique())
    if lowered <= set(TRUE_WORDS) | set(FALSE_WORDS) and not lowered <= {"0", "1"}:
        return "bool", None
    numbers = pd.to_numeric(sample, errors="coerce")
    if numbers.notna().all():
        whole = bool((numbers == numbers.round()).all()) and not sample.str.contains(r"\.").any()
        return ("int" if whole else "float"), None
    for kind, formats in (
        ("datetime", DATETIME_FORMATS),
        ("date", DATE_FORMATS),
        ("time", TIME_FORMATS),
    ):
        for fmt in formats:
            if _parses(sample, fmt):
                return kind, fmt
    return "string", None


def profile_table(name: str, members: list[str], df: pd.DataFrame) -> SourceTable:
    table = SourceTable(name=name, members=members, rows=int(len(df)))
    for column in df.columns:
        values = df[column]
        kind, fmt = infer_kind(values)
        empty = values.str.strip() == ""
        table.columns.append(
            ColumnProfile(
                name=column,
                kind=kind,
                format=fmt,
                null_rate=round(float(empty.mean()), 4) if len(values) else 0.0,
                unique=bool(len(values) and not empty.any() and values.is_unique),
            )
        )
    return table


def _group_name(members: list[str]) -> str:
    """A name for a union group: the shared folder, else the shared stem without its counter."""
    paths = [PurePosixPath(m) for m in members]
    if len(members) == 1:
        return members[0]
    parents = {p.parent.as_posix() for p in paths}
    if len(parents) == 1 and next(iter(parents)) not in (".", ""):
        return next(iter(parents))
    stems = [p.name for p in paths]
    prefix = stems[0]
    for stem in stems[1:]:
        while not stem.startswith(prefix):
            prefix = prefix[:-1]
    return re.sub(r"[\s_\-\d.]+$", "", prefix) or members[0]


def union_groups(tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Source tables with identical headers, by group name (members sorted)."""
    by_header: dict[tuple[str, ...], list[str]] = {}
    for name, df in tables.items():
        by_header.setdefault(tuple(df.columns), []).append(name)
    groups = {}
    for members in by_header.values():
        members = sorted(members)
        name = _group_name(members)
        while name in groups:
            name += "+"
        groups[name] = members
    return dict(sorted(groups.items()))


def profile_tables(tables: dict[str, pd.DataFrame]) -> list[SourceTable]:
    profiled = []
    for name, members in union_groups(tables).items():
        df = pd.concat([tables[m] for m in members], ignore_index=True)
        profiled.append(profile_table(name, members, df))
    return profiled


def fingerprint(profile: list[SourceTable]) -> str:
    """The structure of an export: table and column names with their kinds, not sizes."""
    shape = sorted((t.name, sorted((c.name, c.kind) for c in t.columns)) for t in profile)
    return hashlib.sha256(json.dumps(shape, ensure_ascii=False).encode()).hexdigest()[:16]


def to_json(profile: list[SourceTable]) -> list[dict]:
    return [asdict(t) for t in profile]


def from_json(data: list[dict]) -> list[SourceTable]:
    return [
        SourceTable(
            name=t["name"],
            members=t["members"],
            rows=t["rows"],
            columns=[ColumnProfile(**c) for c in t["columns"]],
        )
        for t in data
    ]
