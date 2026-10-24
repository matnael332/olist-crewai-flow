"""Turn the synthetic raw layer into an export that looks like a real customer's.

    uv run python -m pulse_specs.workforce_saas.scramble data/raw --out customer.zip

What a real attendance export does to us, all at once:
- Hebrew file, sheet and column names, some files in the Windows Hebrew code page (cp1255);
- one Excel workbook with a sheet per table, and a table too long for one sheet split over
  several sheets;
- punches as one file per month, with the date and the time in separate columns;
- Israeli dates (day/month/year), yes/no as כן/לא, enum values in Hebrew;
- numeric customer ids instead of acc_ ids (in the same order), an account name column, and
  columns we must never keep (employee name, national id, phone);
- optionally, whole optional data groups left out.

Industry, region and plan stay as the generator wrote them, so ingesting the scrambled export must
give back the original clean table byte for byte: that is the ingestion's golden test.
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pulse_core.raw import load_raw_schema, load_raw_tables
from pulse_specs.workforce_saas import columns as wf

EXCEL_MAX_ROWS = 1_048_575  # rows per sheet below the header

# per table: (where it goes, its name there, column renames, encoding, separator, style)
# where: "workbook" (export.xlsx), "usage" (usage.xlsx), "monthly" (a folder of CSVs), "csv".
# style "he" writes Hebrew enums, כן/לא and Israeli dates; "en" writes Israeli dates only.
LAYOUT = {
    "accounts": (
        "workbook",
        "לקוחות",
        {
            "account_id": "מספר לקוח",
            "industry": "ענף",
            "region": "אזור",
            "signed_up_at": "תאריך הצטרפות",
            "trial_ends_at": "סוף ניסיון",
            "status": "סטטוס",
            "cancelled_at": "תאריך ביטול",
        },
        None,
        None,
        "he",
    ),
    "billing_monthly": (
        "workbook",
        "חיובים",
        {
            "account_id": "מספר לקוח",
            "period": "חודש חיוב",
            "plan": "מסלול",
            "seats": "רישיונות",
            "mrr": "סכום חודשי",
            "invoice_amount": "סכום חשבונית",
            "paid_at": "תאריך תשלום",
            "days_late": "ימי איחור",
        },
        None,
        None,
        "he",
    ),
    "sites": (
        "workbook",
        "סניפים",
        {
            "site_id": "קוד סניף",
            "account_id": "מספר לקוח",
            "opened_at": "תאריך פתיחה",
            "closed_at": "תאריך סגירה",
        },
        None,
        None,
        "he",
    ),
    "employees": (
        "workbook",
        "עובדים",
        {
            "employee_id": "מספר עובד",
            "account_id": "מספר לקוח",
            "site_id": "קוד סניף",
            "role": "תפקיד",
            "employment_type": "סוג העסקה",
            "hired_at": "תאריך קליטה",
            "left_at": "תאריך סיום עבודה",
        },
        None,
        None,
        "he",
    ),
    "wages": (
        "workbook",
        "תעריפים",
        {
            "account_id": "מספר לקוח",
            "employee_id": "מספר עובד",
            "effective_from": "בתוקף מ",
            "hourly_rate": "תעריף שעתי",
        },
        None,
        None,
        "he",
    ),
    "devices": (
        "workbook",
        "שעונים",
        {
            "device_id": "מספר שעון",
            "account_id": "מספר לקוח",
            "site_id": "קוד סניף",
            "device_type": "סוג שעון",
            "installed_at": "תאריך התקנה",
            "retired_at": "תאריך הסרה",
        },
        None,
        None,
        "he",
    ),
    "usage_daily": (
        "usage",
        "שימוש",
        {
            "account_id": "Customer ID",
            "usage_date": "Date",
            "admin_logins": "Admin Logins",
            "reports_viewed": "Reports Viewed",
            "scheduling_active": "Scheduling Module",
            "absences_active": "Absences Module",
            "documents_active": "Documents Module",
            "messages_sent": "Messages Sent",
            "app_installs_total": "App Installs",
            "support_tickets": "Support Tickets",
        },
        None,
        None,
        "en",
    ),
    "punches": (
        "monthly",
        "החתמות",
        {
            "punch_id": "מספר החתמה",
            "account_id": "מספר לקוח",
            "employee_id": "מספר עובד",
            "site_id": "קוד סניף",
            "device_id": "מספר שעון",
            "direction": "כניסה/יציאה",
            "method": "אמצעי החתמה",
            "edited": "תיקון ידני",
            "edited_by": "נערך על ידי",
        },
        "cp1255",
        ",",
        "he",
    ),
    "shifts": (
        "csv",
        "shifts.csv",
        {
            "shift_id": "Shift ID",
            "account_id": "Customer ID",
            "site_id": "Branch",
            "employee_id": "Employee Number",
            "planned_start": "Shift Start",
            "planned_end": "Shift End",
            "published_at": "Published",
            "outcome": "Shift Status",
        },
        "utf-8",
        ",",
        "en",
    ),
    "absence_requests": (
        "csv",
        "היעדרויות.csv",
        {
            "request_id": "מספר בקשה",
            "account_id": "מספר לקוח",
            "employee_id": "מספר עובד",
            "kind": "סוג היעדרות",
            "start_date": "מתאריך",
            "end_date": "עד תאריך",
            "requested_at": "תאריך בקשה",
            "decided_at": "תאריך החלטה",
            "decision": "החלטה",
        },
        "cp1255",
        ",",
        "he",
    ),
    "hours_daily": (
        "csv",
        "דוח שעות.txt",
        {
            "account_id": "מספר לקוח",
            "employee_id": "מספר עובד",
            "work_date": "תאריך עבודה",
            "regular_hours": "שעות רגילות",
            "overtime_125_hours": "שעות 125",
            "overtime_150_hours": "שעות 150",
            "night_hours": "שעות לילה",
            "rest_violation": "הפרת מנוחה",
        },
        "utf-8-sig",
        "\t",
        "he",
    ),
    "payroll_exports": (
        "csv",
        "payroll_exports.csv",
        {
            "export_id": "Export ID",
            "account_id": "Customer ID",
            "period": "Payroll Period",
            "exported_at": "Export Date",
            "payroll_software": "Payroll Software",
            "corrections_after_export": "Corrections",
        },
        "utf-8",
        ";",
        "en",
    ),
    "device_health_daily": (
        "csv",
        "clock_status.csv",
        {
            "device_id": "Clock ID",
            "account_id": "Customer ID",
            "health_date": "Day",
            "offline_hours": "Offline Hours",
            "failed_recognitions": "Failed Reads",
        },
        "utf-8",
        ",",
        "en",
    ),
}
HEBREW_VALUES = {
    "method": {
        "app": "אפליקציה",
        "mobile": "נייד",
        "nfc": "כרטיס",
        "biometric": "ביומטרי",
        "web": "מחשב",
    },
    "direction": {"in": "כניסה", "out": "יציאה"},
    "outcome": {
        "worked": "בוצעה",
        "open": "פתוחה",
        "late": "איחור",
        "no_show": "לא הגיע",
        "swapped": "הוחלפה",
    },
    "kind": {"sick": "מחלה", "vacation": "חופשה", "unpaid": "ללא תשלום", "other": "אחר"},
    "decision": {"approved": "אושר", "rejected": "נדחה"},
    "edited_by": {"admin": "מנהל", "employee": "עובד"},
}


def _customer_id(account_id: pd.Series) -> pd.Series:
    """acc_000047 -> 10047: numeric, same width, same order."""
    number = pd.to_numeric(account_id.str.removeprefix("acc_"), errors="coerce")
    return (number + 10000).astype("Int64").astype(str).where(account_id.notna(), "")


def _dates(column: pd.Series) -> pd.Series:
    """Israeli day/month/year; the time only when the column has one."""
    has_time = (column.dropna().dt.normalize() != column.dropna()).any()
    fmt = "%d/%m/%Y %H:%M:%S" if has_time else "%d/%m/%Y"
    return column.dt.strftime(fmt).fillna("")


def _hebrew(name: str, column: pd.Series) -> pd.Series:
    values = HEBREW_VALUES.get(name)
    if values is None:
        return column
    lowered = column.str.strip().str.lower()
    return lowered.map(values).fillna(column)  # an unknown spelling stays as it was


def _styled(table: str, df: pd.DataFrame, kinds: dict[str, str], style: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, kind in kinds.items():
        column = df[name]
        if name == "account_id":
            out[name] = _customer_id(column)
        elif kind == "datetime":
            out[name] = _dates(column)
        elif kind == "bool":
            yes, no = ("כן", "לא") if style == "he" else ("Y", "N")
            out[name] = column.map({True: yes, False: no}).fillna("")
        elif style == "he" and kind == "string":
            out[name] = _hebrew(name, column)
        else:
            out[name] = column
    if table == "accounts":
        out["account_name"] = "לקוח " + out["account_id"]
    if table == "employees":
        n = np.arange(len(df))
        out["employee_name"] = ["עובד " + str(i + 1) for i in n]
        out["national_id"] = [f"{300000000 + 7919 * i % 99999999:09d}" for i in n]
        out["phone"] = [f"05{2 + i % 7}-{1000000 + 104729 * i % 8999999:07d}" for i in n]
    return out


EXTRA_HEADERS = {
    "account_name": "שם לקוח",
    "employee_name": "שם עובד",
    "national_id": "ת.ז.",
    "phone": "טלפון",
}


def _headers(table: str, frame: pd.DataFrame) -> pd.DataFrame:
    renames = {**LAYOUT[table][2], **EXTRA_HEADERS}
    return frame.rename(columns=renames)


def _write_sheets(path: Path, sheets: list[tuple[str, pd.DataFrame]]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name, index=False)


def _split(frame: pd.DataFrame, name: str, rows: int) -> list[tuple[str, pd.DataFrame]]:
    if len(frame) <= rows:
        return [(name, frame)]
    return [
        (f"{name} {i + 1}", frame.iloc[start : start + rows])
        for i, start in enumerate(range(0, len(frame), rows))
    ]


def scramble(
    raw_dir: str | Path,
    out_zip: str | Path,
    *,
    drop_groups: tuple[str, ...] = (),
    rows_per_sheet: int = EXCEL_MAX_ROWS,
) -> Path:
    """Write the scrambled export of ``raw_dir`` to ``out_zip``; returns the zip path."""
    raw_dir, out_zip = Path(raw_dir), Path(out_zip)
    dropped = {t for g in drop_groups for t in wf.GROUP_TABLES[g]}
    keep = [t for t in LAYOUT if t not in dropped]
    tables = load_raw_tables(raw_dir, keep)
    schema = load_raw_schema(raw_dir)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workbook, usage = [], []
        for table in keep:
            where, name, _renames, encoding, sep, style = LAYOUT[table]
            kinds = schema["tables"][table]["columns"]
            styled = _styled(table, tables[table], kinds, style)
            if table == "punches":
                stamp = tables[table]["punched_at"]
                styled = styled.drop(columns=["punched_at"])
                styled.insert(5, "תאריך", stamp.dt.strftime("%d/%m/%Y").fillna(""))
                styled.insert(6, "שעה", stamp.dt.strftime("%H:%M:%S").fillna(""))
            frame = _headers(table, styled)
            if where == "workbook":
                workbook.append((name, frame))
            elif where == "usage":
                usage += _split(frame, name, rows_per_sheet)
            elif where == "monthly":
                folder = root / name
                folder.mkdir()
                month = tables[table]["punched_at"].dt.strftime("%Y-%m").fillna("unknown")
                for key in sorted(month.unique()):
                    part = frame[month == key]
                    part.to_csv(
                        folder / f"punches_{key}.csv", index=False, encoding=encoding, sep=sep
                    )
            else:
                frame.to_csv(root / name, index=False, encoding=encoding, sep=sep)
        if workbook:
            _write_sheets(root / "export.xlsx", workbook)
        if usage:
            _write_sheets(root / "usage.xlsx", usage)
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(root).as_posix())
    return out_zip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scramble", description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", help="the raw/ folder written by `uv run synth`")
    parser.add_argument("--out", default="customer_export.zip")
    parser.add_argument(
        "--drop-group", action="append", default=[], choices=sorted(wf.GROUP_TABLES)
    )
    parser.add_argument("--rows-per-sheet", type=int, default=EXCEL_MAX_ROWS)
    args = parser.parse_args(argv)
    path = scramble(
        args.raw_dir,
        args.out,
        drop_groups=tuple(args.drop_group),
        rows_per_sheet=args.rows_per_sheet,
    )
    print(f"written {path} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
