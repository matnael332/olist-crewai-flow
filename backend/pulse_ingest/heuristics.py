"""The rule-based mapper: names, synonyms and kinds, no model.

Names are compared after normalising (case, separators, quotes); a pair scores 1.0 when the
normalised source name equals the target name or one of its synonyms, else the best ``difflib``
ratio. Kinds must be compatible. Tables are matched first (their own name plus how many required
columns find a match), then columns inside each matched table, greedily by score. A required
datetime with no match takes an unmatched date column plus a time column (a punch exported as a
date and a time). Enum spellings map by the contract's value synonyms; what stays unknown is
listed for review and kept, lower-cased, as the customer wrote it.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import PurePosixPath

import pandas as pd

from pulse_ingest.contract import IngestContract, TargetColumn, TargetTable
from pulse_ingest.mapping import ColumnSource, Mapping, TableMapping
from pulse_ingest.profile import SourceTable

COLUMN_THRESHOLD = 0.8
TABLE_THRESHOLD = 0.5
VALUE_THRESHOLD = 0.85
COMPATIBLE = {
    "string": {"string", "int", "float", "bool", "datetime", "date", "time"},
    "int": {"int", "float"},
    "float": {"int", "float"},
    "bool": {"bool", "int"},
    "datetime": {"datetime", "date"},
}


def norm(text: str) -> str:
    text = text.lower().replace('"', "").replace("'", "")
    text = re.sub(r"[_\-./\\:()]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: str, names: tuple[str, ...]) -> float:
    a = norm(a)
    best = 0.0
    for name in names:
        b = norm(name)
        if a == b:
            return 1.0
        best = max(best, SequenceMatcher(None, a, b).ratio())
    return best


def _table_names(source: SourceTable) -> list[str]:
    """What a source table is called: its file or sheet stem and the folder or workbook."""
    names = set()
    for member in [source.name, *source.members]:
        path = PurePosixPath(member)
        names.add(path.stem)
        if path.parent.as_posix() not in (".", ""):
            names.add(PurePosixPath(path.parent).stem)
    return sorted(re.sub(r"[\s_\-\d.]+$", "", n) or n for n in names)


def _column_score(source_kind: str, column: TargetColumn, source_name: str) -> float:
    if source_kind not in COMPATIBLE.get(column.kind, {source_kind}):
        return 0.0
    return similarity(source_name, (column.name, *column.synonyms))


def match_columns(source: SourceTable, target: TargetTable) -> dict[str, ColumnSource]:
    pairs = []
    for col in source.columns:
        for target_col in target.columns:
            score = _column_score(col.kind, target_col, col.name)
            if score >= COLUMN_THRESHOLD:
                pairs.append((score, target_col.required, col.name, target_col.name))
    pairs.sort(key=lambda p: (-p[0], not p[1], p[2], p[3]))
    used_source, out = set(), {}
    for score, _required, src, dst in pairs:
        if src in used_source or dst in out:
            continue
        col = source.column(src)
        out[dst] = ColumnSource(
            column=src, format=col.format, origin="heuristic", confidence=round(score, 3)
        )
        used_source.add(src)
    # a required datetime from a date column and a time column
    dates = [c for c in source.columns if c.kind == "date" and c.name not in used_source]
    times = [c for c in source.columns if c.kind == "time" and c.name not in used_source]
    for target_col in target.columns:
        if target_col.name in out or target_col.kind != "datetime" or not target_col.required:
            continue
        if len(dates) == 1 and len(times) == 1:
            date, time = dates.pop(), times.pop()
            out[target_col.name] = ColumnSource(
                column=date.name,
                format=date.format,
                time_column=time.name,
                time_format=time.format,
                origin="heuristic",
                confidence=0.7,
            )
    for target_col in target.columns:
        if target_col.name in out:
            continue
        if target_col.generate:
            out[target_col.name] = ColumnSource(generate=True, origin="default", confidence=1.0)
        elif target_col.default is not None:
            out[target_col.name] = ColumnSource(
                constant=str(target_col.default), origin="default", confidence=1.0
            )
    return out


def _table_score(source: SourceTable, target: TargetTable) -> float:
    name = max(
        (similarity(n, (target.name, *target.synonyms)) for n in _table_names(source)),
        default=0.0,
    )
    columns = match_columns(source, target)
    required = [c for c in target.columns if c.required]
    found = sum(1 for c in required if c.name in columns and columns[c.name].column)
    coverage = found / len(required) if required else 0.0
    return 0.4 * name + 0.6 * coverage


def propose(profile: list[SourceTable], contract: IngestContract, fingerprint: str) -> Mapping:
    scores = sorted(
        ((_table_score(s, t), s.name, t.name) for s in profile for t in contract.tables),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    by_source = {s.name: s for s in profile}
    assigned_source, mapping = set(), Mapping(contract=contract.name, fingerprint=fingerprint)
    for score, src, dst in scores:
        if score < TABLE_THRESHOLD or src in assigned_source or dst in mapping.tables:
            continue
        target = contract.table(dst)
        columns = match_columns(by_source[src], target)
        mapping.tables[dst] = TableMapping(
            sources=[src], columns=columns, confidence=round(score, 3)
        )
        assigned_source.add(src)
    return finalize(mapping, profile, contract)


def finalize(mapping: Mapping, profile: list[SourceTable], contract: IngestContract) -> Mapping:
    """Recompute the ignored columns and the missing-required notes (after any change)."""
    for source in profile:
        used = set()
        for table in mapping.tables.values():
            if source.name in table.sources:
                used |= {c.column for c in table.columns.values()}
                used |= {c.time_column for c in table.columns.values()}
        mapping.ignored[source.name] = [c.name for c in source.columns if c.name not in used]
    mapping.review = [r for r in mapping.review if "is required and has no source" not in r]
    mapping.review += missing_required(mapping, contract)
    return mapping


def missing_required(mapping: Mapping, contract: IngestContract) -> list[str]:
    problems = []
    for name, table in mapping.tables.items():
        for column in contract.table(name).columns:
            source = table.columns.get(column.name)
            if column.required and not (source and (source.column or source.constant)):
                problems.append(f"{name}.{column.name} is required and has no source column")
    return problems


def propose_values(
    mapping: Mapping, contract: IngestContract, tables: dict[str, pd.DataFrame]
) -> None:
    """Fill ``mapping.value_maps`` for the enum columns from the staged tables' distinct values."""
    for name, table in mapping.tables.items():
        for column in contract.table(name).columns:
            source = table.columns.get(column.name)
            if not column.values or not source or not source.column:
                continue
            seen = sorted(
                {
                    v.strip()
                    for m in table.sources
                    for v in source_values(tables, m, source.column)
                    if v.strip()
                }
            )
            known, unknown = {}, []
            for value in seen:
                canonical = _canonical(value, column)
                if canonical:
                    known[value] = canonical
                else:
                    unknown.append(value)
            mapping.value_maps[f"{name}.{column.name}"] = known
            if unknown:
                mapping.review.append(
                    f"{name}.{column.name}: {len(unknown)} spelling(s) not recognised; "
                    f"kept as written (expected values: {', '.join(column.values)})"
                )


def source_values(tables: dict[str, pd.DataFrame], source: str, column: str) -> list[str]:
    """Distinct values of a column of a source table (``tables`` holds the union groups)."""
    df = tables.get(source)
    return list(df[column].unique()) if df is not None and column in df else []


def _canonical(value: str, column: TargetColumn) -> str | None:
    best, best_score = None, 0.0
    for canonical, spellings in column.value_synonyms.items():
        score = similarity(value, (canonical, *spellings))
        if score > best_score:
            best, best_score = canonical, score
    return best if best_score >= VALUE_THRESHOLD else None
