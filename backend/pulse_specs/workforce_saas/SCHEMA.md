# workforce_saas: the ten data groups

A cloud time-attendance and scheduling SaaS collects the same tables from every customer account:
restaurants, retail chains, factories, construction firms. This spec reads those tables and
answers one question per account at the end of every month: **will this account cancel within
60 days?**

Two layers:

- **Raw layer** (`raw_schema.json`): the ten data groups as 13 event-level tables, exactly as the
  SaaS exports them. The pipeline's first gate checks every file and column against it.
- **Clean layer** (`columns.py`): one row per account and calendar month, every column aggregated
  from one group, with the decision time at the month end.

No real export exists yet; `synth.py` generates the raw layer from a causal story per account
(`SYNTH.md`).

Conventions: ids are synthetic (`acc_000123`, `emp_000045`, `site_000007`, `dev_000012`,
`shift_...`, `req_...`, `exp_...`, `pun_...`); money is ILS; every timestamp is
`YYYY-MM-DD HH:MM:SS`; a date without a time is a `datetime` at midnight. No names, addresses or
ages anywhere.

## Three layers of meaning

| What the group tells | Groups |
|---|---|
| What is happening in the customer's business | 1 punches, 2 organisation, 3 planned vs actual, 4 absences, 5 hours accounting, 6 payroll export, 10 money |
| What is happening between the customer and the SaaS | 7 system usage, 8 commercial |
| Whether the data can be trusted | 9 device health |

The outcome (group 8, `cancelled_at`) is what the model predicts. The first two layers explain
it; the third filters it. Group 9 never predicts: a dead clock is not a closed site.

## The groups

### 1. Punches (`punches.csv`)

The heartbeat: every clock-in and clock-out, from which device, by which method, and whether an
admin edited it afterwards.

| column | meaning |
|---|---|
| punch_id, account_id, employee_id, site_id | who punched where |
| device_id | the clock used; empty for mobile-app and web punches |
| punched_at, direction | the moment and `in` / `out` |
| method | `app`, `biometric`, `nfc`, `web` |
| edited, edited_by | whether the punch was corrected later, and by `admin` or `employee` |

Tells about the business: real opening hours, who is in. Tells the SaaS: the pulse of the
account (a slow decline in punches is the earliest churn signal) and friction (a high share of
edited punches means the team fights the system).

Clean columns: `punches`, `punch_days`, `app_punch_share`, `edited_punch_share`,
`active_employees`, `active_sites`.

### 2. Organisation (`employees.csv`, `sites.csv`)

Who works for the customer, in what role and employment type, at which site, and when they
joined and left. Sites open and close.

Tells about the business: size, hiring and attrition, growth or shrinkage, branch openings and
closures. Tells the SaaS: the seat count is the revenue; its direction is the forecast.

Clean columns: `employees_joined`, `employees_left`, `sites_closed` (and, with group 1,
`active_employees`, `active_sites`).

### 3. Planned vs actual (`shifts.csv`)

The published schedule against what happened: `worked`, `late`, `no_show`, `swapped`, or still
`open` (unassigned). `published_at` says how far ahead the manager plans.

Tells about the business: order versus chaos; staffing gaps. Tells the SaaS: whether the
scheduling module is used at all, which is probably the strongest loyalty signal.

Clean columns: `shifts_planned`, `no_show_rate`, `open_shift_share`, `schedule_lead_days`.

### 4. Absences (`absence_requests.csv`)

Sick, vacation, unpaid and other requests, when they were filed, when and how they were decided.

Tells about the business: load and burnout; a manager who does not answer. Tells the SaaS: little
directly, mostly whether the module is alive.

Clean columns: `absence_requests`, `pending_request_days`, `sick_days_per_employee`.

### 5. Hours accounting (`hours_daily.csv`)

Per employee and day: regular, 125 percent and 150 percent overtime, night hours, and whether the
legal rest rules were violated.

Tells about the business: where overtime burns and where the customer is exposed legally. Tells
the SaaS: a jump in overtime means something changed, for better or worse; the model learns
which.

Clean columns: `overtime_share`, `rest_violations` (and, with group 10, `labor_cost`).

### 6. Payroll export (`payroll_exports.csv`)

The monthly export to the payroll software: when, to which software, and how many corrections
came after it.

Tells about the business: financial order. Tells the SaaS: this is the moment the customer gets
the value. A skipped export is the clearest churn signal there is.

Clean columns: `payroll_exported`, `payroll_export_delay_days`, `payroll_corrections`.

### 7. System usage (`usage_daily.csv`)

Per account and day: admin logins, reports viewed, which optional modules are active (scheduling,
absences, documents), messages sent, total app installs, support tickets.

Tells about the business: almost nothing. Tells the SaaS: everything about engagement. The admin
disappearing before the employees do is the signal nobody else sees.

Clean columns: `admin_login_days`, `reports_viewed`, `modules_active`, `messages_sent`,
`app_install_share`, `support_tickets`.

### 8. Commercial (`accounts.csv`, `billing_monthly.csv`)

The account itself (industry, region, sign-up, trial end, status, cancellation) and the monthly
bill (plan, seats, MRR, invoice, payment date, days late).

Tells the SaaS: the outcome. Who stayed, who cancelled, who upgraded, who pays late.

Clean columns: `tenure_months`, `industry`, `region`, `plan`, `in_trial`, `seats`, `mrr`,
`invoice_days_late`, `payment_missing`; after the decision: `cancelled_at`, `days_to_cancel`.

`accounts.csv` is the only table the target comes from. The clean step reads from it only what
was known at the month end (the status at that time) and moves `cancelled_at` to the
post-decision columns.

### 9. Device health (`devices.csv`, `device_health_daily.csv`)

Installed clocks (type, site, install and retire dates) and, per device and day, hours offline
and failed recognitions.

Tells whether the data can be trusted: a site that vanishes from the punches is either closed or
has a broken clock, and these are opposite stories. Also a churn cause the SaaS owns: devices
that keep failing drive customers away.

Clean columns: `devices_total`, `device_offline_hours`, `device_failure_rate`, `data_gap`.

### 10. Money (`wages.csv`)

Hourly rates per employee, with their effective date. Sensitive: it stays at employee level in
the raw layer and reaches the clean layer only as a per-account total.

Tells about the business: the cost of labour, which turns every other group into money. Tells the
SaaS: how big the customer's business really is, beyond the seat count.

Clean column: `labor_cost`.

## The clean layer

**Grain.** One row per account and calendar month, keyed by `account_month_id` (`acc_000123:2024-03`).

**Population.** Every calendar month in which the account was in trial or active at the start
of the month, from the sign-up month to the cancellation month inclusive. Months whose end plus
60 days is later than the snapshot date are dropped: their outcome is not known yet (right
censoring).

**Decision time.** The last second of the month (`month_end`). Every column tagged `month_end`
is computable from events up to that moment. Splits are day-granular windows on `month_end`.

**Target.** `churned_60d`: `cancelled_at` falls in `(month_end, month_end + 60 days]`.

**Post-decision columns** (never features): `cancelled_at`, `days_to_cancel`.

**Binding rules** (checked by the contract gate): C01 one row per account and month; C02 the
target equals `1 <= days_to_cancel <= 60` wherever `days_to_cancel` is known; C03 an active site
needs an active employee; C04 day counts fit in a month; C05 a churned month has a positive
`days_to_cancel`; every share and rate in [0, 1]; counts non-negative; `modules_active` in 0..3;
industry, region and plan from fixed lists.

**The clean step** (`clean.py`, step 4). The spine is every calendar month from the sign-up month
to the last month alive; the funnel drops R1 months whose outcome is unknown at the snapshot
(`month_end + 60 days` after it), R2 the cancellation month itself, R3 months without a billing
row. Before that, the raw export is normalised and every fix counted (`raw_fixes`): exact
duplicate rows, enum spelling (`App`, `WORKED`, `Sick`, `mobile` -> `app`), swapped start/end
dates. `in_trial` means the trial was still running at the month start; `payroll_exported` means
the previous month was exported by this month end; `days_to_cancel` is the ceiling of the seconds
between the month end and the cancellation, in days; shares are 0 when there is nothing to count
and the nullable rates are null.

**Time windows** are derived from the snapshot by `spec.make_spec(snapshot_date)`: the test window
is the last six months with a known outcome, a two-month gap keeps every training label resolved
before the first test month end, and the train window runs from the first month to the gap. For
the synthetic data (snapshot 2025-09-01): train 2022-01 .. 2024-10, gap 2024-11 .. 2024-12, test
2025-01 .. 2025-06.

## Features (step 4)

57 features, declared in `features.py`:

- **Level** (41): every `month_end` column except the identifiers and `month_end` itself. Booleans
  as 0/1; `industry`, `region` and `plan` are categorical; the nullable rates stay nullable.
- **Trend** (14): for `punches`, `active_employees`, `seats`, `admin_login_days`, `modules_active`,
  `overtime_share` and `payroll_exported`, the change versus the previous month (`_delta_1m`) and
  versus the mean of the previous three months (`_delta_3m`), inside the account's own history;
  null in the account's first month.
- **Streaks** (2): `months_without_payroll_export` and `months_without_admin_login`, consecutive
  months of silence up to and including this one.

Required features: `tenure_months`, `seats`, `punches`, `active_employees`, `admin_login_days`,
`modules_active`, `payroll_exported`, `plan`, `industry`. Forbidden: `cancelled_at`,
`days_to_cancel`, `churned_60d`.
