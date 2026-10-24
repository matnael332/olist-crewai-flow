"""The clean layer of workforce_saas: one row per account and calendar month.

Pure declarations, no logic. The decision time is the month end: every column tagged
``month_end`` is known by then; ``post_decision`` columns describe what happened afterwards and
can never be features; ``label`` is the target. See SCHEMA.md for the story behind each column.
"""

from __future__ import annotations

from pulse_core.columns import Column

ID_COLUMN = "account_month_id"
TIME_COLUMN = "month_end"
DECISION_STAGE = "month_end"
TARGET = "churned_60d"
HORIZON_DAYS = 60
ENTITY = "account"
GRAIN = "one row per customer account and calendar month"

INDUSTRIES = ("restaurant", "retail", "manufacturing", "construction", "services", "other")
REGIONS = ("north", "center", "south", "jerusalem")
PLANS = ("basic", "standard", "premium")

ME = DECISION_STAGE  # short alias for the catalogue below

COLUMNS: tuple[Column, ...] = (
    # identifiers
    Column(ID_COLUMN, "string", "identifier", False, "account_id:YYYY-MM; primary key"),
    Column("account_id", "string", "identifier", False, "customer account"),
    Column("month", "string", "identifier", False, "calendar month, YYYY-MM"),
    # commercial (group 8), known at month end
    Column(TIME_COLUMN, "datetime", ME, False, "last second of the month (decision time)"),
    Column("tenure_months", "int", ME, False, "months since sign-up, counting this month"),
    Column("industry", "string", ME, False, "customer's industry"),
    Column("region", "string", ME, False, "customer's region"),
    Column("plan", "string", ME, False, "subscription plan billed for this month"),
    Column("in_trial", "bool", ME, False, "the account was still in its free trial"),
    Column("seats", "int", ME, False, "seats billed this month"),
    Column("mrr", "float", ME, False, "monthly recurring revenue billed, ILS"),
    Column("invoice_days_late", "int", ME, True, "days the invoice was paid late (null: not due)"),
    Column("payment_missing", "bool", ME, False, "the invoice was unpaid at month end"),
    # organisation (groups 1, 2)
    Column("active_employees", "int", ME, False, "employees with at least one punch this month"),
    Column("employees_joined", "int", ME, False, "employees hired this month"),
    Column("employees_left", "int", ME, False, "employees who left this month"),
    Column("active_sites", "int", ME, False, "sites with at least one punch this month"),
    Column("sites_closed", "int", ME, False, "sites closed this month"),
    # punches (group 1)
    Column("punches", "int", ME, False, "punches this month"),
    Column("punch_days", "int", ME, False, "days with at least one punch"),
    Column("app_punch_share", "float", ME, False, "share of punches made from the mobile app"),
    Column("edited_punch_share", "float", ME, False, "share of punches edited after the fact"),
    # planned vs actual (group 3)
    Column("shifts_planned", "int", ME, False, "shifts in the published schedule"),
    Column("no_show_rate", "float", ME, True, "no-show shifts / assigned shifts (null: none)"),
    Column("open_shift_share", "float", ME, True, "unassigned shifts / shifts (null: none)"),
    Column("schedule_lead_days", "float", ME, True, "mean days between publishing and the shift"),
    # absences (group 4)
    Column("absence_requests", "int", ME, False, "absence requests filed this month"),
    Column("pending_request_days", "float", ME, True, "mean days a request waited (null: none)"),
    Column("sick_days_per_employee", "float", ME, False, "approved sick days / active employees"),
    # hours accounting (group 5)
    Column("overtime_share", "float", ME, False, "overtime hours / all hours"),
    Column("rest_violations", "int", ME, False, "employee-days violating rest rules"),
    # payroll export (group 6)
    Column("payroll_exported", "bool", ME, False, "the previous period was exported to payroll"),
    Column("payroll_export_delay_days", "int", ME, True, "days after period end (null: none)"),
    Column("payroll_corrections", "int", ME, False, "corrections after the last export"),
    # system usage (group 7)
    Column("admin_login_days", "int", ME, False, "days with at least one admin login"),
    Column("reports_viewed", "int", ME, False, "reports opened by admins"),
    Column("modules_active", "int", ME, False, "optional modules in use (0-3)"),
    Column("messages_sent", "int", ME, False, "messages sent through the system"),
    Column("app_install_share", "float", ME, True, "app installs / active employees (null: none)"),
    Column("support_tickets", "int", ME, False, "support tickets opened"),
    # device health (group 9)
    Column("devices_total", "int", ME, False, "installed, non-retired devices"),
    Column("device_offline_hours", "float", ME, False, "device-hours offline this month"),
    Column("device_failure_rate", "float", ME, True, "failed recognitions / punches (null: none)"),
    Column("data_gap", "bool", ME, False, "devices were offline most of the month"),
    # money (groups 5 and 10)
    Column("labor_cost", "float", ME, True, "hours times hourly rate, ILS (null: no wages)"),
    # after the decision
    Column("cancelled_at", "datetime", "post_decision", True, "cancellation time, if any"),
    Column("days_to_cancel", "int", "post_decision", True, "days from month end to cancellation"),
    Column(TARGET, "bool", "label", False, f"cancelled within {HORIZON_DAYS} days of month end"),
)

FORBIDDEN = ("cancelled_at", "days_to_cancel", TARGET)

# The data groups (SCHEMA.md) by key: the raw tables each one needs. A customer's export always
# holds the core groups; any other group may be missing, and its columns are then left out of the
# clean table and the model.
GROUP_TABLES = {
    "commercial": ("accounts", "billing_monthly"),
    "organisation": ("employees", "sites"),
    "punches": ("punches",),
    "shifts": ("shifts",),
    "absences": ("absence_requests",),
    "hours": ("hours_daily",),
    "payroll": ("payroll_exports",),
    "usage": ("usage_daily",),
    "devices": ("devices", "device_health_daily"),
    "money": ("wages",),
}
ALL_GROUPS = tuple(GROUP_TABLES)
CORE_GROUPS = ("commercial", "punches", "payroll", "usage")

# Clean columns that need an optional group (all of the listed groups); every other column comes
# from the core groups.
OPTIONAL_COLUMN_GROUPS = {
    "employees_joined": ("organisation",),
    "employees_left": ("organisation",),
    "sites_closed": ("organisation",),
    "shifts_planned": ("shifts",),
    "no_show_rate": ("shifts",),
    "open_shift_share": ("shifts",),
    "schedule_lead_days": ("shifts",),
    "absence_requests": ("absences",),
    "pending_request_days": ("absences",),
    "sick_days_per_employee": ("absences",),
    "overtime_share": ("hours",),
    "rest_violations": ("hours",),
    "devices_total": ("devices",),
    "device_offline_hours": ("devices",),
    "device_failure_rate": ("devices",),
    "data_gap": ("devices",),
    "labor_cost": ("hours", "money"),
}


def columns_for(groups: tuple[str, ...]) -> tuple[Column, ...]:
    """The column catalogue restricted to the columns the given data groups can fill."""
    present = set(groups)
    return tuple(
        c for c in COLUMNS if set(OPTIONAL_COLUMN_GROUPS.get(c.name, ())).issubset(present)
    )


PATTERNS = {
    ID_COLUMN: r"^acc_[0-9]{6}:20[0-9]{2}-(0[1-9]|1[0-2])$",
    "account_id": r"^acc_[0-9]{6}$",
    "month": r"^20[0-9]{2}-(0[1-9]|1[0-2])$",
}

_UNIT = {"min": 0, "max": 1}
_COUNT = {"min": 0}
BOUNDS = {
    "tenure_months": {"min": 1},
    "seats": _COUNT,
    "mrr": _COUNT,
    "invoice_days_late": _COUNT,
    "active_employees": _COUNT,
    "employees_joined": _COUNT,
    "employees_left": _COUNT,
    "active_sites": _COUNT,
    "sites_closed": _COUNT,
    "punches": _COUNT,
    "punch_days": {"min": 0, "max": 31},
    "app_punch_share": _UNIT,
    "edited_punch_share": _UNIT,
    "shifts_planned": _COUNT,
    "no_show_rate": _UNIT,
    "open_shift_share": _UNIT,
    "schedule_lead_days": _COUNT,
    "absence_requests": _COUNT,
    "pending_request_days": _COUNT,
    "sick_days_per_employee": _COUNT,
    "overtime_share": _UNIT,
    "rest_violations": _COUNT,
    "payroll_export_delay_days": _COUNT,
    "payroll_corrections": _COUNT,
    "admin_login_days": {"min": 0, "max": 31},
    "reports_viewed": _COUNT,
    "modules_active": {"min": 0, "max": 3},
    "messages_sent": _COUNT,
    "app_install_share": _UNIT,
    "support_tickets": _COUNT,
    "devices_total": _COUNT,
    "device_offline_hours": _COUNT,
    "device_failure_rate": _UNIT,
    "labor_cost": _COUNT,
    "days_to_cancel": {"min": 1},
}

ALLOWED_VALUES = {
    "industry": list(INDUSTRIES),
    "region": list(REGIONS),
    "plan": list(PLANS),
}

CONSTRAINTS = (
    {
        "id": "C01",
        "type": "unique",
        "columns": [ID_COLUMN],
        "description": "one row per account and month",
    },
    {
        "id": "C02",
        "type": "query",
        "query": f"{TARGET} == ((days_to_cancel >= 1) & (days_to_cancel <= {HORIZON_DAYS}))",
        "where_not_null": ["days_to_cancel"],
        "description": "target definition",
    },
    {
        "id": "C05",
        "type": "query",
        "query": f"~{TARGET} | (days_to_cancel >= 1)",
        "description": "a churned month has a positive days_to_cancel",
    },
    {
        "id": "C03",
        "type": "query",
        "query": "active_employees >= active_sites",
        "description": "an active site needs at least one active employee",
    },
    {
        "id": "C04",
        "type": "query",
        "query": "punch_days <= 31 and admin_login_days <= 31",
        "description": "day counts fit in a month",
    },
)
