"""Apply an approved mapping: source tables of strings -> typed target tables.

Per target table the source groups are concatenated, then every raw target column is built from
its source (parsed with the format the profile found, a date joined to its time column), its
constant, a stable row number, or left empty. Enum spellings go through the value map (unknown
ones are kept, lower-cased); free categories are trimmed and lower-cased; identifiers are
pseudonymised. Only target columns survive: everything the mapping ignored is gone from here on.
"""

from __future__ import annotations

import pandas as pd

from pulse_ingest.contract import IngestContract, TargetColumn
from pulse_ingest.heuristics import norm
from pulse_ingest.mapping import ColumnSource, Mapping
from pulse_ingest.profile import FALSE_WORDS, TRUE_WORDS
from pulse_ingest.pseudonymize import EntityMap, canonical_ids, hashed


class MappingError(ValueError):
    """The mapping cannot be applied (not approved, or a required column has no source)."""


def _parse(values: pd.Series, column: TargetColumn, source: ColumnSource) -> pd.Series:
    text = values.astype("string").str.strip().replace("", pd.NA)
    if column.kind == "datetime":
        if source.format:
            return pd.to_datetime(text, format=source.format, errors="coerce")
        return pd.to_datetime(text, errors="coerce", dayfirst=True)
    if column.kind in ("int", "float"):
        numbers = pd.to_numeric(text, errors="coerce")
        return (
            numbers.round().astype("Int64") if column.kind == "int" else numbers.astype("float64")
        )
    if column.kind == "bool":
        lowered = text.str.lower()
        out = pd.Series(pd.NA, index=text.index, dtype="boolean")
        out[lowered.isin(TRUE_WORDS)] = True
        out[lowered.isin([w for w in FALSE_WORDS if w])] = False
        return out
    return text


def _enum(values: pd.Series, value_map: dict[str, str]) -> pd.Series:
    lookup = {norm(k): v for k, v in value_map.items()}
    text = values.astype("string").str.strip()
    mapped = text.map(lambda v: lookup.get(norm(v)) if isinstance(v, str) else pd.NA)
    return mapped.fillna(text.str.lower()).replace("", pd.NA)


def build_table(
    name: str,
    frames: list[pd.DataFrame],
    mapping: Mapping,
    contract: IngestContract,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """The typed target table (ids not yet pseudonymised) and the account names, if mapped."""
    table_map = mapping.tables[name]
    target = contract.table(name)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    for column in target.raw_columns:
        source = table_map.columns.get(column.name)
        if source and source.column:
            values = df[source.column]
            if source.time_column:
                joined = values.str.strip() + " " + df[source.time_column].str.strip()
                fmt = f"{source.format} {source.time_format}"
                out[column.name] = pd.to_datetime(joined, format=fmt, errors="coerce")
                continue
            parsed = _parse(values, column, source)
            if column.values:
                parsed = _enum(values, mapping.value_maps.get(f"{name}.{column.name}", {}))
            elif column.free_values:
                parsed = parsed.str.lower()
            out[column.name] = parsed
        elif source and source.generate:
            out[column.name] = [f"{name}_{i + 1:09d}" for i in range(len(df))]
        elif source and source.constant is not None:
            out[column.name] = source.constant
        elif column.required:
            raise MappingError(f"{name}.{column.name} is required and has no source")
        else:
            out[column.name] = pd.Series(pd.NA, index=df.index, dtype="object")
    names = None
    name_source = table_map.columns.get("account_name")
    if name_source and name_source.column and "account_id" in out:
        names = pd.Series(
            df[name_source.column].str.strip().to_numpy(),
            index=canonical_ids(out["account_id"]),
        )
    return out, names


def normalize(
    mapping: Mapping,
    contract: IngestContract,
    tables: dict[str, pd.DataFrame],
    entity_map: EntityMap,
    secret: bytes,
) -> dict[str, pd.DataFrame]:
    """Typed, pseudonymised target tables of an approved mapping (the entity map is updated)."""
    if mapping.status != "approved":
        raise MappingError("the mapping is not approved")
    built, names = {}, {}
    for name, table_map in mapping.tables.items():
        frames = [tables[s] for s in table_map.sources]
        built[name], table_names = build_table(name, frames, mapping, contract)
        if table_names is not None:
            names.update(table_names.dropna().to_dict())
    accounts = pd.concat(
        [df["account_id"] for df in built.values() if "account_id" in df], ignore_index=True
    )
    entity_map.assign(accounts, names)
    for name, df in built.items():
        for column in contract.table(name).raw_columns:
            if column.id_kind == "account":
                df[column.name] = entity_map.pseudonyms(df[column.name])
            elif column.id_kind in ("employee", "site", "device"):
                df[column.name] = hashed(df[column.name], column.id_kind, secret)
    return built
