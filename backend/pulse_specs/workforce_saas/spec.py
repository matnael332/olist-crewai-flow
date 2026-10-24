"""The workforce_saas DatasetSpec.

``make_spec(snapshot_date)`` derives the time windows from the snapshot: the test window is the
last ``test_months`` months whose 60-day outcome is known at the snapshot, a gap of
``gap_months`` keeps every training label resolved before the first test month end, and the
train window runs from ``train_start`` to the gap. ``SPEC`` is the spec of the synthetic dataset
written by ``synth.py``; the hosted service builds one spec per customer with that customer's
snapshot and first month: ``--spec pulse_specs.workforce_saas.spec:make_spec --spec-params
<file>``, the file holding ``make_spec``'s keyword arguments as JSON.
"""

from __future__ import annotations

from functools import partial

import pandas as pd

from pulse_core.spec import DatasetSpec
from pulse_specs.workforce_saas import columns as wf
from pulse_specs.workforce_saas.clean import HORIZON, clean_tables
from pulse_specs.workforce_saas.eda import eda_figures, eda_summary
from pulse_specs.workforce_saas.features import (
    FEATURE_DEFINITIONS,
    REQUIRED_FEATURES,
    derive_features,
)

DEFAULT_SNAPSHOT = "2025-09-01"
DEFAULT_TRAIN_START = "2022-01-01"

POPULATION = {
    "entity": "customer account; one row per account and calendar month",
    "included": (
        "every account-month from the sign-up month on, while the account is still a customer at "
        "the month end (trial or paying)"
    ),
    "excluded": (
        "months whose month_end + 60 days is after the snapshot (outcome unknown); the "
        "cancellation month itself; months without a billing row"
    ),
}
PRIVACY = {
    "note": (
        "Pseudonymous ids only (acc_, emp_, site_, dev_); no names, addresses or ages anywhere. "
        "The customer uploads its export to our server, where it is processed and stored under "
        "the customer's own folder and kept until the customer deletes it or asks us to. Hourly "
        "wages stay employee-level in the raw layer and reach the clean layer only as a monthly "
        "per-account total (labor_cost)."
    ),
    "processing": "hosted; one folder per customer; no data is shared between customers",
    "sensitive_raw_columns": ["wages.hourly_rate", "employees.role", "employees.employment_type"],
}
MODEL_CARD_ITEMS = {
    "out_of_scope": [
        "Decisions about individual employees: the model scores customer accounts only.",
        "Pricing, collections or any automated action towards the customer.",
        "Customers of other SaaS products or other countries without a new backtest.",
    ],
    "limitations": [
        "Trained on a synthetic export whose churn story was planted by the generator; real "
        "customers will need a backtest on their own history.",
        "A 60-day horizon: slower declines are only visible across several monthly scores.",
        "Absolute trend features (deltas) scale with account size; the model sees the plan and "
        "seat count alongside them.",
        "Invoice lateness is only measured inside the month; later payment shows up as "
        "payment_missing.",
    ],
    "ethical_considerations": [
        "Scores rank accounts for retention outreach by the SaaS vendor; they are never about "
        "employees, and no protected attributes are used.",
        "Processing happens on our servers on pseudonymous data; see the privacy note in the "
        "data contract.",
    ],
    "monitoring_recommendations": [
        "Positive rate per month and the share of months with a payroll export (drift in the "
        "strongest signal).",
        "Share of data_gap months: broken clocks look like silence.",
        "Null rates of the nullable features and the distribution of seats and tenure.",
        "Retrain whenever the snapshot moves: the windows are derived from it.",
    ],
}


def windows(
    snapshot_date: str,
    train_start: str = DEFAULT_TRAIN_START,
    test_months: int = 6,
    gap_months: int = 2,
) -> dict[str, tuple[str, str]]:
    """Day-granular, inclusive windows on month_end, derived from the snapshot."""
    snapshot = pd.Timestamp(snapshot_date)
    last = (snapshot - HORIZON).to_period("M")
    while last.end_time.floor("s") + HORIZON > snapshot:
        last -= 1
    test_start = last - (test_months - 1)
    gap_end = test_start - 1
    gap_start = gap_end - (gap_months - 1)
    train_end = gap_start - 1
    day = lambda p, attr: str(getattr(p, attr).date())  # noqa: E731
    if pd.Timestamp(train_start) > train_end.end_time:
        raise ValueError(
            f"train_start {train_start} is after the train window end {day(train_end, 'end_time')}"
        )
    return {
        "train": (train_start, day(train_end, "end_time")),
        "horizon_gap": (day(gap_start, "start_time"), day(gap_end, "end_time")),
        "test": (day(test_start, "start_time"), day(last, "end_time")),
    }


def make_spec(
    snapshot_date: str = DEFAULT_SNAPSHOT,
    *,
    train_start: str = DEFAULT_TRAIN_START,
    test_months: int = 6,
    gap_months: int = 2,
    source: str = "workforce_saas raw export (synthetic, pulse_specs.workforce_saas.synth)",
) -> DatasetSpec:
    split = windows(snapshot_date, train_start, test_months, gap_months)
    return DatasetSpec(
        name="workforce_saas",
        source=source,
        license="none (synthetic)",
        grain=wf.GRAIN,
        snapshot_date=snapshot_date,
        contract_version="1.0",
        seed=42,
        id_column=wf.ID_COLUMN,
        time_column=wf.TIME_COLUMN,
        decision_stage=wf.DECISION_STAGE,
        entity=wf.ENTITY,
        decision_time="the last second of the calendar month (month_end)",
        target=wf.TARGET,
        target_definition=f"cancelled_at falls in (month_end, month_end + {wf.HORIZON_DAYS} days]",
        target_label="churned",
        rejected_target=(
            "accounts.status == 'cancelled' at the snapshot: not time-consistent (every month of a "
            "late canceller would be positive) and not evaluable on a time split"
        ),
        splits=split,
        split_note=(
            f"{gap_months}-month gap between train and test: every train label is resolved "
            f"({wf.HORIZON_DAYS} days after its month end) before the first test month end; months "
            "after the test window have no known outcome and are not in the data"
        ),
        cv_note="TimeSeriesSplit(3) over train rows ordered by month_end then account_id",
        columns=wf.COLUMNS,
        clean=partial(clean_tables, snapshot=pd.Timestamp(snapshot_date)),
        patterns=wf.PATTERNS,
        bounds=wf.BOUNDS,
        constraints=wf.CONSTRAINTS,
        allowed_values=wf.ALLOWED_VALUES,
        population=POPULATION,
        privacy=PRIVACY,
        feature_definitions=FEATURE_DEFINITIONS,
        required_features=REQUIRED_FEATURES,
        forbidden_features=wf.FORBIDDEN,
        derive_features=derive_features,
        eda_summary=eda_summary,
        eda_figures=eda_figures,
        model_card_items=MODEL_CARD_ITEMS,
    )


SPEC = make_spec()
