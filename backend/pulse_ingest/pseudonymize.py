"""Identifiers the pipeline sees instead of the customer's own.

- Accounts get sequential pseudonyms (``acc_000001``...), in the natural order of the customer's
  ids, kept stable from one upload to the next by the entity map: the only place where a
  pseudonym meets the customer's id and the account name. The website reads it to show names;
  the pipeline and the LLM never do.
- Employees, sites and devices get a keyed hash (``emp_`` + 10 hex digits); the key is the
  customer's own secret, so the same id maps the same way every month and cannot be reversed
  without it. National ids used as employee numbers end here.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path

import pandas as pd

PREFIX = {"account": "acc_", "employee": "emp_", "site": "site_", "device": "dev_"}
MAP_COLUMNS = ["pseudo_id", "source_id", "name"]


def canonical_ids(values: pd.Series) -> pd.Series:
    """Ids as text, trimmed, with a spreadsheet's ``.0`` removed from whole numbers."""
    text = values.astype("string").str.strip()
    return text.str.replace(r"^(\d+)\.0+$", r"\1", regex=True).replace("", pd.NA)


def load_secret(path: Path) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(secrets.token_bytes(32))
    return path.read_bytes()


def hashed(values: pd.Series, kind: str, secret: bytes) -> pd.Series:
    """Keyed hash of every non-empty value; empty stays empty (NA)."""
    cache: dict[str, str] = {}

    def one(value: str) -> str:
        if value not in cache:
            digest = hmac.new(secret, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()
            cache[value] = PREFIX[kind] + digest[:10]
        return cache[value]

    return canonical_ids(values).map(one, na_action="ignore").astype("string")


def _natural(value: str) -> tuple:
    return (0, int(value), "") if value.isdigit() else (1, 0, value)


class EntityMap:
    """pseudo_id <-> source_id (and the account name), stored as a CSV in the customer folder."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        else:
            frame = pd.DataFrame(columns=MAP_COLUMNS)
        self.by_source = dict(zip(frame["source_id"], frame["pseudo_id"], strict=True))
        self.names = dict(zip(frame["pseudo_id"], frame["name"], strict=True))

    def assign(self, source_ids: pd.Series, names: dict[str, str] | None = None) -> None:
        """Give every new source id the next pseudonym, in natural order; update names."""
        seen = set(canonical_ids(source_ids).dropna())
        new = sorted(seen - set(self.by_source), key=_natural)
        start = len(self.by_source)
        for i, source in enumerate(new, start=start + 1):
            self.by_source[source] = f"{PREFIX['account']}{i:06d}"
        for source, name in (names or {}).items():
            if source in self.by_source and name:
                self.names[self.by_source[source]] = name

    def pseudonyms(self, values: pd.Series) -> pd.Series:
        return canonical_ids(values).map(self.by_source).astype("string")

    def save(self) -> None:
        rows = [
            {"pseudo_id": p, "source_id": s, "name": self.names.get(p, "")}
            for s, p in self.by_source.items()
        ]
        frame = pd.DataFrame(rows, columns=MAP_COLUMNS).sort_values("pseudo_id")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.path, index=False, encoding="utf-8", lineterminator="\n")
