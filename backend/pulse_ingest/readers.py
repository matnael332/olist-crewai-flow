"""Read staged files into source tables of strings.

Every cell is read as text (an empty cell is an empty string); types and formats are inferred by
``profile`` and applied by ``normalize``, so nothing is guessed twice. A delimited file is one
table; an Excel sheet is one table named ``<workbook>/<sheet>``. Encodings: UTF-8 (with or
without a BOM), the Windows Hebrew code page (cp1255), or whatever charset-normalizer finds.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from charset_normalizer import from_bytes

DELIMITERS = ",;\t|"
SAMPLE = 1 << 16


def detect_encoding(head: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1255"):
        try:
            head.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding == "cp1255" and not any(0xE0 <= b <= 0xFA for b in head):
            break  # cp1255 decodes almost anything: only take it when Hebrew letters are there
        return encoding
    best = from_bytes(head).best()
    return best.encoding if best else "utf-8"


def detect_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text, delimiters=DELIMITERS).delimiter
    except csv.Error:
        return ","


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("").astype(str)
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed:")]]


def read_delimited(path: Path) -> tuple[pd.DataFrame, dict]:
    head = path.read_bytes()[:SAMPLE]
    encoding = detect_encoding(head)
    text = head.decode(encoding, errors="replace")
    delimiter = detect_delimiter(text.splitlines()[0] if text else "")
    df = pd.read_csv(
        path, encoding=encoding, sep=delimiter, dtype=str, keep_default_na=False, na_filter=False
    )
    return _clean(df), {"encoding": encoding, "delimiter": delimiter}


def read_workbook(path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, engine="openpyxl")
    return {name: _clean(df) for name, df in sheets.items() if not df.empty}


def read_files(files: list[Path], root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Source tables by name (the file path inside the export) and how each was read."""
    tables, info = {}, {}
    for path in files:
        name = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".xlsx":
            for sheet, df in read_workbook(path).items():
                tables[f"{name}/{sheet}"] = df
                info[f"{name}/{sheet}"] = {"file": name, "sheet": sheet}
        else:
            df, how = read_delimited(path)
            tables[name] = df
            info[name] = {"file": name, **how}
    return tables, info
