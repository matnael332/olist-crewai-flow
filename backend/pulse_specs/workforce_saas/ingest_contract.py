"""The ingestion contract of workforce_saas: what a customer's export must become.

Built from ``raw_schema.json`` (tables, columns, types, files, keys) plus what the cleaning needs:
which columns are required, which are identifiers, which semantic enums it compares against
(``punches.method == "app"``, ``shifts.outcome in {open, no_show}``, ``absence_requests.kind ==
"sick"`` and ``decision == "approved"``), and the Hebrew and English names an attendance export
tends to use. The synonyms drive the heuristic mapper; the LLM mapper sees the same target names.
"""

from __future__ import annotations

from pulse_core.io import read_json
from pulse_ingest.contract import IngestContract, TargetColumn, TargetTable
from pulse_specs.workforce_saas import columns as wf
from pulse_specs.workforce_saas.preflight import preflight
from pulse_specs.workforce_saas.synth import SCHEMA_PATH

GROUP_OF_TABLE = {t: g for g, tables in wf.GROUP_TABLES.items() for t in tables}

TABLE_SYNONYMS = {
    "accounts": ("customers", "clients", "accounts", "לקוחות", "חשבונות", "רשימת לקוחות"),
    "billing_monthly": ("billing", "invoices", "subscriptions", "חיובים", "חשבוניות", "מנויים"),
    "employees": ("employees", "workers", "staff", "עובדים", "רשימת עובדים"),
    "sites": ("sites", "branches", "locations", "אתרים", "סניפים"),
    "punches": ("punches", "clockings", "attendance", "החתמות", "דיווחי נוכחות", "נוכחות"),
    "shifts": ("shifts", "schedule", "roster", "משמרות", "סידור עבודה"),
    "absence_requests": ("absences", "leave requests", "היעדרויות", "בקשות היעדרות"),
    "hours_daily": ("daily hours", "hours", "timesheet", "שעות יומיות", "שעות", "דוח שעות"),
    "payroll_exports": ("payroll exports", "payroll", "ייצוא שכר", "קליטות שכר", "שכר"),
    "usage_daily": ("usage", "system usage", "activity", "שימוש", "שימוש במערכת", "פעילות"),
    "devices": ("devices", "clocks", "terminals", "שעונים", "מכשירים"),
    "device_health_daily": ("device health", "clock status", "תקינות שעונים", "מצב שעונים"),
    "wages": ("wages", "hourly rates", "pay rates", "תעריפים", "שכר שעתי"),
}

# Column synonyms by column name (shared across tables: an account id is an account id).
COLUMN_SYNONYMS = {
    "account_id": (
        "customer id",
        "client id",
        "account",
        "customer number",
        "מספר לקוח",
        "קוד לקוח",
        "לקוח",
    ),
    "account_name": ("customer name", "client name", "company", "שם לקוח", "שם חברה", "שם החשבון"),
    "industry": ("industry", "sector", "ענף", "תחום"),
    "region": ("region", "area", "אזור", "מחוז"),
    "signed_up_at": (
        "signup date",
        "joined",
        "start date",
        "created",
        "תאריך הצטרפות",
        "תאריך פתיחה",
    ),
    "trial_ends_at": ("trial end", "end of trial", "סוף ניסיון", "תום תקופת ניסיון"),
    "status": ("status", "state", "סטטוס", "מצב"),
    "cancelled_at": ("cancellation date", "churn date", "cancelled", "תאריך ביטול", "תאריך עזיבה"),
    "period": ("period", "month", "billing month", "payroll period", "תקופה", "חודש", "חודש חיוב"),
    "plan": ("plan", "package", "tier", "מסלול", "חבילה"),
    "seats": ("seats", "licenses", "users", "מושבים", "רישיונות", "כמות משתמשים"),
    "mrr": ("mrr", "monthly amount", "monthly fee", "סכום חודשי", "דמי מנוי"),
    "invoice_amount": ("invoice amount", "total", "סכום חשבונית", 'סה"כ'),
    "paid_at": ("payment date", "paid", "תאריך תשלום", "שולם בתאריך"),
    "days_late": ("days late", "delay days", "ימי איחור"),
    "employee_id": ("employee id", "worker id", "employee number", "מספר עובד", "קוד עובד"),
    "site_id": ("site id", "branch id", "site", "branch", "מספר אתר", "קוד סניף", "סניף"),
    "role": ("role", "position", "job", "תפקיד"),
    "employment_type": ("employment type", "contract type", "סוג העסקה", "היקף משרה"),
    "hired_at": ("hire date", "start date", "תאריך תחילת עבודה", "תאריך קליטה"),
    "left_at": ("termination date", "end date", "left", "תאריך סיום עבודה", "תאריך עזיבה"),
    "opened_at": ("opened", "open date", "תאריך פתיחה"),
    "closed_at": ("closed", "close date", "תאריך סגירה"),
    "punch_id": ("punch id", "record id", "מספר החתמה", "מזהה רשומה"),
    "device_id": ("device id", "clock id", "terminal", "מספר שעון", "קוד מכשיר"),
    "punched_at": (
        "punch time",
        "timestamp",
        "clock time",
        "date time",
        "זמן החתמה",
        "תאריך ושעה",
        "מועד",
    ),
    "direction": ("direction", "in out", "type", "כניסה יציאה", "כיוון", "סוג החתמה"),
    "method": ("method", "source", "punch method", "אמצעי החתמה", "אמצעי", "מקור"),
    "edited": ("edited", "manual", "corrected", "נערך", "תיקון ידני", "ידני"),
    "edited_by": ("edited by", "corrected by", "נערך על ידי", "תוקן על ידי"),
    "shift_id": ("shift id", "מספר משמרת"),
    "planned_start": ("planned start", "shift start", "תחילת משמרת", "שעת התחלה מתוכננת"),
    "planned_end": ("planned end", "shift end", "סוף משמרת", "שעת סיום מתוכננת"),
    "published_at": ("published", "publish date", "תאריך פרסום", "פורסם"),
    "outcome": ("outcome", "result", "shift status", "תוצאה", "סטטוס משמרת"),
    "request_id": ("request id", "מספר בקשה"),
    "kind": ("absence type", "type", "reason", "סוג היעדרות", "סיבה", "סוג"),
    "start_date": ("from", "start date", "from date", "מתאריך", "תאריך התחלה"),
    "end_date": ("to", "end date", "to date", "עד תאריך", "תאריך סיום"),
    "requested_at": ("requested", "request date", "תאריך בקשה", "הוגש"),
    "decided_at": ("decided", "decision date", "approval date", "תאריך החלטה", "תאריך אישור"),
    "decision": ("decision", "approval", "החלטה", "אישור", "סטטוס בקשה"),
    "work_date": ("work date", "date", "day", "תאריך עבודה", "תאריך", "יום"),
    "regular_hours": ("regular hours", "hours 100", "שעות רגילות", "100%"),
    "overtime_125_hours": ("overtime 125", "ot 125", "שעות 125", "125%"),
    "overtime_150_hours": ("overtime 150", "ot 150", "שעות 150", "150%"),
    "night_hours": ("night hours", "שעות לילה"),
    "rest_violation": ("rest violation", "no rest", "הפרת מנוחה", "חריגת מנוחה"),
    "export_id": ("export id", "מספר ייצוא"),
    "exported_at": ("export date", "exported", "תאריך ייצוא", "יוצא בתאריך"),
    "payroll_software": ("payroll software", "payroll system", "תוכנת שכר", "מערכת שכר"),
    "corrections_after_export": (
        "corrections",
        "fixes after export",
        "תיקונים אחרי ייצוא",
        "תיקונים",
    ),
    "usage_date": ("usage date", "date", "day", "תאריך שימוש", "תאריך", "יום"),
    "admin_logins": ("admin logins", "logins", "כניסות מנהל", "כניסות"),
    "reports_viewed": ("reports viewed", "reports", "דוחות שנצפו", "דוחות"),
    "scheduling_active": ("scheduling module", "scheduling", "מודול משמרות", "סידור עבודה פעיל"),
    "absences_active": ("absences module", "מודול היעדרויות"),
    "documents_active": ("documents module", "מודול מסמכים"),
    "messages_sent": ("messages sent", "messages", "הודעות שנשלחו", "הודעות"),
    "app_installs_total": ("app installs", "installs", "התקנות אפליקציה", "התקנות"),
    "support_tickets": ("support tickets", "tickets", "פניות תמיכה", "קריאות שירות"),
    "device_type": ("device type", "clock type", "סוג שעון", "סוג מכשיר"),
    "installed_at": ("installed", "install date", "תאריך התקנה"),
    "retired_at": ("retired", "removed", "תאריך הסרה"),
    "health_date": ("date", "day", "תאריך", "יום"),
    "offline_hours": ("offline hours", "downtime", "שעות ללא תקשורת", "שעות מנותק"),
    "failed_recognitions": ("failed recognitions", "failed reads", "זיהויים שנכשלו", "כשלי זיהוי"),
    "effective_from": ("effective from", "valid from", "בתוקף מ", "החל מתאריך"),
    "hourly_rate": ("hourly rate", "rate", "תעריף שעתי", "תעריף", "שכר לשעה"),
}

# Semantic enums: canonical values and the spellings an export uses for them.
ENUMS = {
    ("punches", "method"): {
        "app": ("app", "mobile", "application", "אפליקציה", "נייד", "סלולרי"),
        "nfc": ("nfc", "card", "tag", "כרטיס", "תג"),
        "biometric": ("biometric", "fingerprint", "face", "ביומטרי", "טביעת אצבע"),
        "web": ("web", "browser", "computer", "אתר", "מחשב", "דפדפן"),
    },
    ("punches", "direction"): {
        "in": ("in", "entry", "enter", "כניסה"),
        "out": ("out", "exit", "leave", "יציאה"),
    },
    ("shifts", "outcome"): {
        "worked": ("worked", "done", "בוצעה", "עבד"),
        "open": ("open", "unassigned", "פתוחה", "לא שובצה"),
        "late": ("late", "איחור", "באיחור"),
        "no_show": ("no show", "no_show", "absent", "לא הגיע", "לא הגיעה", "אי הגעה"),
        "swapped": ("swapped", "swap", "הוחלפה", "החלפה"),
    },
    ("absence_requests", "kind"): {
        "sick": ("sick", "illness", "מחלה", "ימי מחלה"),
        "vacation": ("vacation", "holiday", "חופשה", "חופש"),
        "unpaid": ("unpaid", "ללא תשלום", 'חל"ת'),
        "other": ("other", "אחר"),
    },
    ("absence_requests", "decision"): {
        "approved": ("approved", "yes", "אושר", "אושרה", "מאושר"),
        "rejected": ("rejected", "denied", "no", "נדחה", "נדחתה", "לא אושר"),
    },
}
FREE_VALUES = {("accounts", "industry"), ("accounts", "region"), ("billing_monthly", "plan")}

# Columns the cleaning reads and cannot do without (keys come on top); anything else may be
# missing from an export and is left empty.
REQUIRED = {
    "accounts": ("account_id", "signed_up_at", "cancelled_at"),
    "billing_monthly": ("account_id", "period", "plan", "seats", "mrr", "paid_at"),
    "employees": ("employee_id", "account_id", "hired_at"),
    "sites": ("site_id", "account_id"),
    "punches": ("account_id", "employee_id", "punched_at"),
    "shifts": ("account_id", "planned_start", "outcome"),
    "absence_requests": (
        "account_id",
        "kind",
        "start_date",
        "end_date",
        "requested_at",
        "decided_at",
        "decision",
    ),
    "hours_daily": ("account_id", "employee_id", "work_date", "regular_hours"),
    "payroll_exports": ("account_id", "period", "exported_at"),
    "usage_daily": ("account_id", "usage_date", "admin_logins"),
    "devices": ("device_id", "account_id", "installed_at"),
    "device_health_daily": ("device_id", "account_id", "health_date", "offline_hours"),
    "wages": ("account_id", "employee_id", "effective_from", "hourly_rate"),
}
ID_KINDS = {
    "account_id": "account",
    "employee_id": "employee",
    "site_id": "site",
    "device_id": "device",
    "punch_id": "row",
    "shift_id": "row",
    "request_id": "row",
    "export_id": "row",
}
GENERATED = {"punch_id", "shift_id", "request_id", "export_id"}  # row keys an export may lack
DEFAULTS = {("accounts", "industry"): "unknown", ("accounts", "region"): "unknown"}


def _column(table: str, name: str, kind: str) -> TargetColumn:
    enum = ENUMS.get((table, name), {})
    return TargetColumn(
        name=name,
        kind=kind,
        required=name in REQUIRED[table],
        id_kind=ID_KINDS.get(name),
        generate=name in GENERATED,
        default=DEFAULTS.get((table, name)),
        values=tuple(enum),
        value_synonyms=dict(enum),
        free_values=(table, name) in FREE_VALUES,
        synonyms=COLUMN_SYNONYMS.get(name, ()),
    )


def build_contract() -> IngestContract:
    schema = read_json(SCHEMA_PATH)
    tables = []
    for name, spec in schema["tables"].items():
        columns = [_column(name, c, kind) for c, kind in spec["columns"].items()]
        if name == "accounts":
            columns.append(
                TargetColumn(
                    name="account_name",
                    kind="string",
                    synonyms=COLUMN_SYNONYMS["account_name"],
                    extra=True,
                )
            )
        tables.append(
            TargetTable(
                name=name,
                file=spec["file"],
                group=GROUP_OF_TABLE[name],
                columns=tuple(columns),
                primary_key=tuple(spec.get("primary_key", ())),
                synonyms=TABLE_SYNONYMS[name],
            )
        )
    return IngestContract(
        name="workforce_saas",
        tables=tuple(tables),
        core_groups=wf.CORE_GROUPS,
        datetime_format=schema["datetime_format"],
        preflight=preflight,
        spec_ref="pulse_specs.workforce_saas.spec:make_spec",
    )


CONTRACT = build_contract()
