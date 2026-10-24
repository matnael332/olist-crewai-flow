"""The causal model behind the workforce_saas synthetic data.

One account at a time: ``plan_account`` decides the story (industry, archetype, a monthly health
path and the cancellation, if any) and ``build_account_tables`` turns the story into the thirteen
raw tables through one employees-by-days grid. Every draw comes from the account's own random
generator, so adding accounts never changes earlier ones. ``SYNTH.md`` tells the story in words.

Frames returned here carry local integer keys (``_acc``, ``_emp``, ``_site``, ``_dev``) instead of
ids; ``synth.generate`` sorts, assigns the public ids and drops the helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pulse_specs.workforce_saas.columns import INDUSTRIES, REGIONS

ARCHETYPES = (
    "steady",
    "growing",
    "shrinking",
    "seasonal",
    "trial_bounce",
    "slow_fade",
    "price_sensitive",
    "device_trouble",
)
ARCHETYPE_PROBS = (0.34, 0.12, 0.08, 0.08, 0.14, 0.12, 0.07, 0.05)
INDUSTRY_PROBS = (0.25, 0.22, 0.13, 0.12, 0.18, 0.10)  # same order as INDUSTRIES
REGION_PROBS = (0.25, 0.40, 0.20, 0.15)  # same order as REGIONS
HEADCOUNT_MEDIAN = {
    "restaurant": 8,
    "retail": 7,
    "manufacturing": 12,
    "construction": 9,
    "services": 5,
    "other": 6,
}
SEVEN_DAY = {"restaurant", "retail"}
_SUN_THU = (6, 0, 1, 2, 3)  # pandas weekday numbers
WORK_WEEKDAYS = {
    "restaurant": tuple(range(7)),
    "retail": tuple(range(7)),
    "manufacturing": _SUN_THU,
    "construction": (6, 0, 1, 2, 3, 4),
    "services": _SUN_THU,
    "other": _SUN_THU,
}
# (first start minute, last start minute, probability) per industry
SHIFT_WINDOWS = {
    "restaurant": ((420, 660, 0.55), (900, 1080, 0.45)),
    "retail": ((480, 600, 0.60), (840, 960, 0.40)),
    "manufacturing": ((360, 480, 1.0),),
    "construction": ((360, 420, 1.0),),
    "services": ((480, 540, 1.0),),
    "other": ((480, 540, 1.0),),
}
SHIFT_MEAN_MIN = {"manufacturing": 528}
SHIFT_DEFAULT_MIN = 498
ROLES = ("staff", "shift_lead", "manager")
ROLE_PROBS = (0.80, 0.12, 0.08)
ROLE_MULT = (1.0, 1.2, 1.5)
EMPLOYMENT_TYPES = ("full_time", "part_time", "contractor")
EMPLOYMENT_PROBS = (0.65, 0.30, 0.05)
ATTEND_P = (0.85, 0.50, 0.35)
PLAN_PRICE = {"basic": 18, "standard": 28, "premium": 42}
PAYROLL_SOFTWARE = ("hilan", "michpal", "shiklulit", "priority", "other")
ABSENCE_KINDS = ("sick", "vacation", "unpaid", "other")
ABSENCE_PROBS = (0.45, 0.35, 0.12, 0.08)
ABSENCE_MIN_DAYS = np.array((1, 2, 1, 1))
ABSENCE_MAX_DAYS = np.array((3, 10, 5, 2))
DEVICE_TYPES = ("biometric", "nfc")
HAZARD_ALPHA = {
    "steady": -6.0,
    "seasonal": -6.0,
    "device_trouble": -6.0,
    "growing": -6.5,
    "shrinking": -5.0,
    "slow_fade": -5.5,
    "price_sensitive": -5.5,
}
HAZARD_BETA = 6.0
HAZARD_ALPHA_LATE = -4.0
HIRE_LEAVE = {"steady": (0.025, 0.025), "growing": (0.06, 0.02), "shrinking": (0.01, 0.05)}
BASE_HIRE = 0.025
TROUBLE_ATTENDANCE = 0.58  # punches left at a site while its clock is down

NAT = np.datetime64("NaT")
SECOND = np.timedelta64(1, "s")
HOUR_S = 3600


@dataclass(frozen=True)
class SynthConfig:
    """Knobs of the generator; the defaults are the shipped dataset."""

    accounts: int = 200
    seed: int = 42
    start_month: str = "2022-01"  # first possible sign-up month
    snapshot_date: str = "2025-09-01"  # events stop one second before; sign-ups end a month before
    dirt: float = 0.02  # 0 disables every dirt kind
    emp_sigma: float = 0.5  # lognormal sigma of the initial headcount
    emp_cap: int = 150
    device_share: float = 0.55  # accounts that install clocks
    scheduling_share: float = 0.55
    absences_share: float = 0.70
    documents_share: float = 0.40

    @property
    def snapshot(self) -> pd.Timestamp:
        return pd.Timestamp(self.snapshot_date)

    @property
    def months(self) -> pd.PeriodIndex:
        """Calendar months in which an account may sign up."""
        last = self.snapshot.to_period("M") - 1
        return pd.period_range(self.start_month, last, freq="M")


@dataclass
class AccountPlan:
    """Everything decided about an account before its daily grid is built."""

    index: int
    industry: str
    region: str
    archetype: str
    signed_up_at: pd.Timestamp
    trial_ends_at: pd.Timestamp
    cancelled_at: pd.Timestamp | None
    cancel_reason: str
    forced_cancel: bool
    decline_start: int | None  # month offset since sign-up
    trouble_start: pd.Timestamp | None
    end: pd.Timestamp  # exclusive bound of every event
    months: pd.PeriodIndex  # calendar months of the account's life
    health: np.ndarray
    hazard: np.ndarray
    n0: int
    has_devices: bool
    scheduling_on: np.ndarray
    absences_on: np.ndarray
    documents_on: np.ndarray
    admin_rate: float
    payroll_software: str
    fade_length: int | None = None
    streaks: list[tuple[int, int, np.datetime64, np.datetime64]] = field(default_factory=list)


# ----------------------------------------------------------------------------- the story


def _health_path(
    archetype: str, life: pd.PeriodIndex, decline_start: int | None, rng: np.random.Generator
) -> tuple[np.ndarray, int | None]:
    n = len(life)
    m = np.arange(n, dtype=float)
    fade = None
    if archetype == "steady":
        noise = np.zeros(n)
        for t in range(n):
            prev = noise[t - 1] if t else 0.0
            noise[t] = 0.7 * prev + rng.normal(0, 0.04 * (1 - 0.49) ** 0.5)
        base = 0.85 + noise
    elif archetype == "growing":
        base = 0.80 + 0.01 * m
    elif archetype == "shrinking":
        base = 0.80 - 0.02 * np.maximum(0, m - 6)
    elif archetype == "seasonal":
        base = 0.80 + 0.15 * np.sin(2 * np.pi * (np.asarray(life.month) - 4) / 12)
    elif archetype == "trial_bounce":
        base = np.full(n, 0.40)
    elif archetype == "slow_fade":
        base = np.full(n, 0.85)
        if decline_start is not None:
            fade = int(rng.integers(4, 7))
            after = m >= decline_start
            base[after] = np.maximum(0.15, 0.85 - 0.70 * (m[after] - decline_start + 1) / fade)
    elif archetype == "price_sensitive":
        base = np.full(n, 0.75)
    else:  # device_trouble
        base = np.full(n, 0.80)
    health = np.clip(base + rng.normal(0, 0.03, n), 0.05, 1.0)
    return health, fade


def _random_moment(month: pd.Period, rng: np.random.Generator, hours=(9, 18)) -> pd.Timestamp:
    return month.start_time + pd.Timedelta(
        days=int(rng.integers(month.days_in_month)),
        hours=int(rng.integers(*hours)),
        minutes=int(rng.integers(60)),
    )


def plan_account(index: int, config: SynthConfig, rng: np.random.Generator) -> AccountPlan:
    industry = str(rng.choice(INDUSTRIES, p=INDUSTRY_PROBS))
    region = str(rng.choice(REGIONS, p=REGION_PROBS))
    archetype = str(rng.choice(ARCHETYPES, p=ARCHETYPE_PROBS))
    all_months = config.months
    first = all_months[int(rng.integers(len(all_months)))]
    signed_up_at = _random_moment(first, rng, hours=(8, 20))
    trial_ends_at = signed_up_at + pd.Timedelta(days=14 if rng.random() < 0.6 else 30)
    life = pd.period_range(first, all_months[-1], freq="M")
    n = len(life)
    trial_month = (trial_ends_at.to_period("M") - first).n
    decline_start = None
    if archetype in ("slow_fade", "price_sensitive") and n > 3:
        decline_start = int(rng.integers(3, n))
    health, fade = _health_path(archetype, life, decline_start, rng)

    m = np.arange(n)
    hazard = np.zeros(n)
    if archetype != "trial_bounce":
        alpha = np.full(n, HAZARD_ALPHA[archetype])
        if archetype == "price_sensitive" and decline_start is not None:
            alpha[m >= decline_start] = HAZARD_ALPHA_LATE
        logit = alpha + HAZARD_BETA * (1 - health)
        eligible = m > trial_month
        hazard[eligible] = (1 / (1 + np.exp(-logit)))[eligible]

    cancelled_at: pd.Timestamp | None = None
    reason, forced = "random", False
    if archetype == "trial_bounce":
        when = trial_ends_at + pd.Timedelta(
            days=int(rng.integers(0, 15)), hours=int(rng.integers(9, 18))
        )
        if when < config.snapshot:
            cancelled_at, reason = when, "trial_not_converted"
    else:
        fired = np.flatnonzero(rng.random(n) < hazard)
        cancel_month = int(fired[0]) if len(fired) else None
        forced_month = None
        if decline_start is not None and archetype == "slow_fade":
            forced_month = decline_start + fade + 1
        elif decline_start is not None and archetype == "price_sensitive":
            forced_month = decline_start + int(rng.integers(2, 5))
        if forced_month is not None and forced_month < n:
            if cancel_month is None or forced_month < cancel_month:
                cancel_month, forced = forced_month, True
        if cancel_month is not None:
            cancelled_at = _random_moment(life[cancel_month], rng)
            if decline_start is not None and cancel_month >= decline_start:
                reason = "fade" if archetype == "slow_fade" else "price"

    if cancelled_at is not None:
        months = life[: (cancelled_at.to_period("M") - first).n + 1]
    else:
        months = life
    kept = len(months)
    end = cancelled_at if cancelled_at is not None else config.snapshot
    health, hazard = health[:kept], hazard[:kept]

    has_devices = archetype == "device_trouble" or rng.random() < config.device_share
    trouble_start = None
    if archetype == "device_trouble":
        life_days = max(int((end - signed_up_at).days), 1)
        offset = int(rng.integers(30, max(31, life_days - 20))) if life_days >= 40 else 5
        trouble_start = signed_up_at.normalize() + pd.Timedelta(days=offset)
    median = HEADCOUNT_MEDIAN[industry]
    n0 = int(np.clip(np.rint(rng.lognormal(np.log(median), config.emp_sigma)), 2, config.emp_cap))
    scheduling = rng.random() < config.scheduling_share
    absences = rng.random() < config.absences_share
    documents = rng.random() < config.documents_share
    return AccountPlan(
        index=index,
        industry=industry,
        region=region,
        archetype=archetype,
        signed_up_at=signed_up_at,
        trial_ends_at=trial_ends_at,
        cancelled_at=cancelled_at,
        cancel_reason=reason if cancelled_at is not None else "",
        forced_cancel=forced and cancelled_at is not None,
        decline_start=decline_start,
        trouble_start=trouble_start,
        end=end,
        months=months,
        health=health,
        hazard=hazard,
        n0=n0,
        has_devices=has_devices,
        scheduling_on=scheduling & (health >= 0.5),
        absences_on=absences & (health >= 0.35),
        documents_on=documents & (health >= 0.3),
        admin_rate=float(rng.uniform(0.3, 1.2)),
        payroll_software=str(rng.choice(PAYROLL_SOFTWARE)),
        fade_length=fade,
    )


# ----------------------------------------------------------------------------- the tables


def _s(ts: pd.Timestamp) -> np.datetime64:
    return ts.to_datetime64().astype("datetime64[s]")


def _dt(values) -> np.ndarray:
    """Any sequence of timestamps (numpy, pandas or NaT) as a datetime64[ns] array."""
    arr = np.asarray(values)
    if arr.dtype.kind != "M":
        arr = pd.to_datetime(pd.Series(list(values), dtype=object)).to_numpy()
    return arr.astype("datetime64[ns]")


def _organisation(plan: AccountPlan, rng: np.random.Generator):
    n_sites = 1 if plan.n0 < 12 else 1 + int(min(3, rng.poisson(plan.n0 / 15)))
    day0 = plan.signed_up_at.normalize()
    opened = [day0 - pd.Timedelta(days=int(d)) for d in rng.integers(0, 366, n_sites)]
    closed: list = [pd.NaT] * n_sites
    if plan.archetype == "shrinking" and n_sites >= 2 and len(plan.months) > 7:
        site = int(rng.integers(1, n_sites))
        month = plan.months[int(rng.integers(6, len(plan.months)))]
        when = month.start_time + pd.Timedelta(days=int(rng.integers(month.days_in_month)))
        if when < plan.end:
            closed[site] = when

    last_day = (plan.end - pd.Timedelta(seconds=1)).normalize()
    hired = [
        min(day0 + pd.Timedelta(days=int(d)), last_day) for d in rng.integers(-1095, 31, plan.n0)
    ]
    left: list = [pd.NaT] * plan.n0
    for m, month in enumerate(plan.months):
        h = float(plan.health[m])
        active = [i for i in range(len(hired)) if left[i] is pd.NaT]
        hire_rate, leave_rate = HIRE_LEAVE.get(plan.archetype, (BASE_HIRE, 0.025 + 0.08 * (1 - h)))
        n_hire = int(rng.poisson(len(active) * hire_rate))
        n_leave = int(rng.binomial(len(active), leave_rate)) if active else 0
        start, dim = month.start_time, month.days_in_month
        for d in rng.integers(0, dim, n_hire):
            when = start + pd.Timedelta(days=int(d))
            if when < plan.end:
                hired.append(when)
                left.append(pd.NaT)
        for i in rng.choice(active, n_leave, replace=False) if n_leave else []:
            when = start + pd.Timedelta(days=int(rng.integers(dim)))
            when = max(when, hired[i] + pd.Timedelta(days=1))
            if when < plan.end:
                left[i] = when

    n = len(hired)
    site = rng.integers(0, n_sites, n)
    role = rng.choice(len(ROLES), n, p=ROLE_PROBS)
    etype = rng.choice(len(EMPLOYMENT_TYPES), n, p=EMPLOYMENT_PROBS)
    for s, when in enumerate(closed):
        if when is pd.NaT:
            continue
        for i in np.flatnonzero(site == s):
            if hired[i] >= when:
                site[i] = 0
            elif left[i] is pd.NaT or left[i] > when:
                left[i] = when
    employees = pd.DataFrame(
        {
            "_acc": plan.index,
            "_emp": np.arange(n),
            "_site": site,
            "role": np.array(ROLES)[role],
            "employment_type": np.array(EMPLOYMENT_TYPES)[etype],
            "hired_at": _dt(hired),
            "left_at": _dt(left),
            "_uses_app": rng.random(n) < (0.3 if plan.has_devices else 0.9),
            "_p_attend": np.array(ATTEND_P)[etype],
        }
    )
    sites = pd.DataFrame(
        {
            "_acc": plan.index,
            "_site": np.arange(n_sites),
            "opened_at": _dt(opened),
            "closed_at": _dt(closed),
        }
    )

    rate = np.round(34 * rng.lognormal(0, 0.25, n) * np.array(ROLE_MULT)[role], 2)
    wage_rows = []
    for i in range(n):
        current = float(rate[i])
        wage_rows.append((i, hired[i], current))
        until = left[i] if left[i] is not pd.NaT else plan.end
        k = 1
        while True:
            anniversary = hired[i] + pd.DateOffset(years=k)
            if anniversary >= until:
                break
            if rng.random() < 0.4:
                current = round(current * (1 + rng.uniform(0.03, 0.08)), 2)
                wage_rows.append((i, anniversary, current))
            k += 1
    wages = pd.DataFrame(wage_rows, columns=["_emp", "effective_from", "hourly_rate"])
    wages.insert(0, "_acc", plan.index)
    wages["effective_from"] = _dt(wages["effective_from"])
    return employees, sites, wages


def _devices(plan: AccountPlan, sites: pd.DataFrame, employees: pd.DataFrame, rng):
    columns = ["_acc", "_dev", "_site", "device_type", "installed_at", "retired_at"]
    if not plan.has_devices:
        return pd.DataFrame(columns=columns)
    rows = []
    day0 = plan.signed_up_at.normalize()
    for s in range(len(sites)):
        headcount = int((employees["_site"] == s).sum())
        for _ in range(2 if headcount > 25 else 1):
            installed = max(
                sites["opened_at"].iloc[s], day0 + pd.Timedelta(days=int(rng.integers(0, 21)))
            )
            if installed >= plan.end:
                continue
            retired = sites["closed_at"].iloc[s]
            kind = str(rng.choice(DEVICE_TYPES, p=(0.6, 0.4)))
            span_days = int((plan.end - installed).days)
            if rng.random() < 0.10 and span_days > 60:
                swap = installed + pd.Timedelta(days=int(rng.integers(30, span_days - 10)))
                rows.append((s, kind, installed, swap))
                rows.append((s, kind, swap, retired))
            else:
                rows.append((s, kind, installed, retired))
    devices = pd.DataFrame(rows, columns=["_site", "device_type", "installed_at", "retired_at"])
    devices.insert(0, "_dev", np.arange(len(devices)))
    devices.insert(0, "_acc", plan.index)
    devices["installed_at"] = _dt(devices["installed_at"])
    devices["retired_at"] = _dt(devices["retired_at"])

    if plan.archetype == "device_trouble" and len(devices) and plan.trouble_start is not None:
        start = plan.trouble_start
        active = devices[
            (devices["installed_at"] <= start)
            & (devices["retired_at"].isna() | (devices["retired_at"] > start))
        ]
        chosen = active.iloc[0] if len(active) else devices.iloc[0]
        cursor = max(start, chosen["installed_at"])
        for _ in range(int(rng.integers(1, 4))):
            stop = cursor + pd.Timedelta(days=int(rng.integers(10, 41)))
            if cursor >= plan.end:
                break
            plan.streaks.append(
                (int(chosen["_dev"]), int(chosen["_site"]), _s(cursor), _s(min(stop, plan.end)))
            )
            cursor = stop + pd.Timedelta(days=int(rng.integers(15, 61)))
    return devices


@dataclass
class Grid:
    days: np.ndarray  # datetime64[s], one per calendar day of the account's life
    month_idx: np.ndarray
    h_day: np.ndarray
    open_day: np.ndarray
    employed: np.ndarray  # employees x days
    attend: np.ndarray
    trouble: np.ndarray
    planned_start: np.ndarray
    punch_in: np.ndarray
    punch_out: np.ndarray


def _attendance_grid(plan: AccountPlan, employees: pd.DataFrame, rng) -> Grid:
    first_month = plan.months[0]
    index = pd.date_range(
        plan.signed_up_at.normalize(), (plan.end - pd.Timedelta(seconds=1)).normalize(), freq="D"
    )
    days = index.values.astype("datetime64[s]")
    n_days = len(days)
    month_idx = (index.year - first_month.year) * 12 + (index.month - first_month.month)
    month_idx = np.clip(np.asarray(month_idx), 0, len(plan.months) - 1)
    h_day = plan.health[month_idx]
    open_day = np.asarray(index.weekday.isin(WORK_WEEKDAYS[plan.industry]))

    hired = employees["hired_at"].values.astype("datetime64[s]")
    left = employees["left_at"].values.astype("datetime64[s]")
    employed = (hired[:, None] <= days[None, :]) & (
        np.isnat(left)[:, None] | (left[:, None] > days[None, :])
    )
    n_emp = len(employees)
    site = employees["_site"].to_numpy()
    trouble = np.zeros((n_emp, n_days), dtype=bool)
    for _dev, s, start, stop in plan.streaks:
        trouble[np.ix_(site == s, (days >= start) & (days < stop))] = True

    p = employees["_p_attend"].to_numpy()[:, None] * open_day[None, :]
    if plan.industry in SEVEN_DAY:
        p = p * (5 / 7)
    p = p * np.minimum(1.0, 1.3 * h_day)[None, :] * np.where(trouble, TROUBLE_ATTENDANCE, 1.0)
    attend = employed & (rng.random((n_emp, n_days)) < p)

    windows = SHIFT_WINDOWS[plan.industry]
    choice = rng.choice(len(windows), n_emp, p=[w[2] for w in windows])
    lo = np.array([windows[c][0] for c in choice])
    hi = np.array([windows[c][1] for c in choice])
    base = rng.integers(lo, hi + 1)
    planned_min = base[:, None] + np.rint(rng.normal(0, 10, (n_emp, n_days)))
    mean_len = SHIFT_MEAN_MIN.get(plan.industry, SHIFT_DEFAULT_MIN)
    length = rng.normal(mean_len, 60, (n_emp, n_days)) * (1 + 0.15 * (1 - h_day))[None, :]
    length = np.clip(length, 180, 840)
    to_s = lambda minutes: (np.rint(minutes) * 60).astype("int64").astype("timedelta64[s]")  # noqa: E731
    planned_start = days[None, :] + to_s(planned_min)
    punch_in = planned_start + to_s(rng.normal(2, 8, (n_emp, n_days)))
    punch_out = punch_in + to_s(length + rng.normal(0, 20, (n_emp, n_days)))
    return Grid(
        days,
        month_idx,
        h_day,
        open_day,
        employed,
        attend,
        trouble,
        planned_start,
        punch_in,
        punch_out,
    )


def _overlap(a, b, lo, hi):
    return np.maximum(0, np.minimum(b, hi) - np.maximum(a, lo))


def _punches_and_hours(plan: AccountPlan, grid: Grid, employees, devices, rng):
    end = _s(plan.end)
    e, d = np.nonzero(grid.attend)
    t_in, t_out = grid.punch_in[e, d], grid.punch_out[e, d]
    keep = t_out < end
    e, d, t_in, t_out = e[keep], d[keep], t_in[keep], t_out[keep]
    n = len(e)
    site = employees["_site"].to_numpy()[e]

    n_sites = int(employees["_site"].max()) + 1 if len(employees) else 1
    site_dev = np.full((n_sites, len(grid.days)), -1)
    for k in range(len(devices)):
        row = devices.iloc[k]
        installed = np.datetime64(row["installed_at"], "s")
        active = grid.days >= installed
        if not pd.isna(row["retired_at"]):
            active &= grid.days < np.datetime64(row["retired_at"], "s")
        site_dev[int(row["_site"]), active] = k
    dev = site_dev[site, d]
    has_dev = dev >= 0
    trouble = grid.trouble[e, d]
    p_dev = np.where(has_dev, np.where(trouble, 0.5, 0.8), 0.0)
    p_app = np.where(has_dev, np.where(trouble, 0.4, 0.15), 0.8)
    u = rng.random(n)
    method = np.where(u < p_dev, "device", np.where(u < p_dev + p_app, "app", "web")).astype(object)
    via_device = method == "device"
    if via_device.any():
        method[via_device] = devices["device_type"].to_numpy()[dev[via_device]]
    dev_local = np.where(via_device, dev, -1)

    h = grid.h_day[d]
    p_edit = np.tile(0.03 + 0.12 * (1 - h), 2)
    edited = rng.random(2 * n) < p_edit
    edited_by = np.where(edited, np.where(rng.random(2 * n) < 0.8, "admin", "employee"), "")
    punches = pd.DataFrame(
        {
            "_acc": plan.index,
            "_emp": np.concatenate([e, e]),
            "_site": np.concatenate([site, site]),
            "_dev": np.concatenate([dev_local, dev_local]),
            "punched_at": _dt(np.concatenate([t_in, t_out])),
            "direction": np.repeat(["in", "out"], n),
            "method": np.concatenate([method, method]),
            "edited": edited,
            "edited_by": edited_by,
        }
    )

    total = (t_out - t_in).astype("int64") / HOUR_S
    a = (t_in - grid.days[d]).astype("int64") / 60
    b = (t_out - grid.days[d]).astype("int64") / 60
    night = (_overlap(a, b, 0, 360) + _overlap(a, b, 1320, 1800)) / 60
    short_rest = np.zeros(n, dtype=bool)
    if n > 1:
        gap = (t_in[1:] - t_out[:-1]).astype("int64") / HOUR_S
        short_rest[1:] = (e[1:] == e[:-1]) & (gap < 8)
    hours = pd.DataFrame(
        {
            "_acc": plan.index,
            "_emp": e,
            "work_date": _dt(grid.days[d]),
            "regular_hours": np.round(np.minimum(total, 8.6), 2),
            "overtime_125_hours": np.round(np.clip(total - 8.6, 0, 2), 2),
            "overtime_150_hours": np.round(np.maximum(total - 10.6, 0), 2),
            "night_hours": np.round(night, 2),
            "rest_violation": (total > 12) | short_rest,
        }
    )
    dev_punches = np.zeros((len(devices), len(grid.days)))
    if via_device.any():
        np.add.at(dev_punches, (dev[via_device], d[via_device]), 2)
    return punches, hours, dev_punches


def _shifts(plan: AccountPlan, grid: Grid, employees, rng):
    columns = ["_acc", "_emp", "_site", "planned_start", "planned_end", "published_at", "outcome"]
    sched = plan.scheduling_on[grid.month_idx]
    if not sched.any() or not len(employees):
        return pd.DataFrame(columns=columns)
    end = _s(plan.end)
    site = employees["_site"].to_numpy()
    shape = grid.employed.shape
    frames = []

    def make(e, d, outcome, assigned=True):
        starts = grid.planned_start[e, d]
        keep = starts < end
        e, d, starts = e[keep], d[keep], starts[keep]
        n = len(e)
        if not n:
            return
        h = grid.h_day[d]
        length = (np.rint(np.clip(rng.normal(SHIFT_DEFAULT_MIN, 30, n), 300, 720)) * 60).astype(
            "int64"
        )
        lead = np.maximum(rng.gamma(2.0, (2 + 8 * h) / 2) * 86400, HOUR_S).astype("int64")
        label = outcome if isinstance(outcome, np.ndarray) else np.full(n, outcome, dtype=object)
        if isinstance(outcome, np.ndarray):
            label = outcome[keep]
        frames.append(
            pd.DataFrame(
                {
                    "_acc": plan.index,
                    "_emp": e if assigned else -1,
                    "_site": site[e],
                    "planned_start": _dt(starts),
                    "planned_end": _dt(starts + length.astype("timedelta64[s]")),
                    "published_at": _dt(starts - lead.astype("timedelta64[s]")),
                    "outcome": label,
                }
            )
        )

    e, d = np.nonzero(grid.attend & sched[None, :])
    late = (grid.punch_in[e, d] - grid.planned_start[e, d]) > np.timedelta64(15, "m")
    outcome = np.where(
        rng.random(len(e)) < 0.03, "swapped", np.where(late, "late", "worked")
    ).astype(object)
    make(e, d, outcome)
    base = grid.employed & grid.open_day[None, :] & sched[None, :]
    p_no_show = (0.03 + 0.08 * (1 - grid.h_day))[None, :]
    make(*np.nonzero(base & ~grid.attend & (rng.random(shape) < p_no_show)), "no_show")
    p_open = (0.02 + 0.10 * (1 - grid.h_day))[None, :]
    make(*np.nonzero(base & (rng.random(shape) < p_open)), "open", assigned=False)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _absences(plan: AccountPlan, employees, rng):
    columns = [
        "_acc",
        "_emp",
        "kind",
        "start_date",
        "end_date",
        "requested_at",
        "decided_at",
        "decision",
    ]
    hired = employees["hired_at"].to_numpy()
    left = employees["left_at"].to_numpy()
    end = np.datetime64(plan.end, "ns")
    frames = []
    for m, month in enumerate(plan.months):
        if not plan.absences_on[m]:
            continue
        start, stop = np.datetime64(month.start_time, "ns"), np.datetime64(month.end_time, "ns")
        idx = np.flatnonzero((hired <= stop) & (np.isnat(left) | (left >= start)))
        emp = np.repeat(idx, rng.poisson(0.25, len(idx)))
        n = len(emp)
        if not n:
            continue
        kind = rng.choice(len(ABSENCE_KINDS), n, p=ABSENCE_PROBS)
        length = rng.integers(ABSENCE_MIN_DAYS[kind], ABSENCE_MAX_DAYS[kind] + 1)
        day = lambda k: k.astype("timedelta64[D]")  # noqa: E731
        start_date = start + day(rng.integers(0, month.days_in_month, n))
        ahead = np.where(kind == 0, 0, rng.integers(3, 31, n))
        requested = start_date - day(ahead) + rng.integers(7, 23, n).astype("timedelta64[h]")
        requested = np.maximum(requested, hired[emp] + np.timedelta64(8, "h"))
        keep = requested < end
        emp, kind, length, start_date, requested = (
            emp[keep],
            kind[keep],
            length[keep],
            start_date[keep],
            requested[keep],
        )
        n = len(emp)
        h = float(plan.health[m])
        delay = (
            (rng.exponential(1 + 6 * (1 - h), n) * 86400).astype("int64").astype("timedelta64[s]")
        )
        decided_at = requested + delay
        decided = (rng.random(n) < 0.9) & (decided_at < end)
        decision = np.where(decided, np.where(rng.random(n) < 0.85, "approved", "rejected"), "")
        frames.append(
            pd.DataFrame(
                {
                    "_acc": plan.index,
                    "_emp": emp,
                    "kind": np.array(ABSENCE_KINDS)[kind],
                    "start_date": _dt(start_date),
                    "end_date": _dt(start_date + day(length - 1)),
                    "requested_at": _dt(requested),
                    "decided_at": _dt(np.where(decided, decided_at, NAT)),
                    "decision": decision,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _payroll(plan: AccountPlan, employees, rng):
    rows = []
    trial_month = (plan.trial_ends_at.to_period("M") - plan.months[0]).n
    for m, month in enumerate(plan.months):
        h = float(plan.health[m])
        if m <= trial_month and rng.random() >= 0.3:
            continue
        if rng.random() < (1 - h) ** 2:
            continue
        exported = (
            month.end_time.normalize()
            + pd.Timedelta(days=1 + rng.gamma(2.0, 2.0) + 4 * (1 - h))
            + pd.Timedelta(hours=int(rng.integers(8, 18)))
        ).floor("min")
        if exported >= plan.end:
            continue
        n_emp = int(
            (
                (employees["hired_at"] <= month.end_time)
                & (employees["left_at"].isna() | (employees["left_at"] >= month.start_time))
            ).sum()
        )
        rows.append(
            (month.start_time, exported, int(rng.poisson(0.5 + 3 * (1 - h) + 0.02 * n_emp)))
        )
    frame = pd.DataFrame(rows, columns=["period", "exported_at", "corrections_after_export"])
    frame.insert(0, "_acc", plan.index)
    frame["payroll_software"] = plan.payroll_software
    frame["period"] = _dt(frame["period"])
    frame["exported_at"] = _dt(frame["exported_at"])
    return frame


def _usage(plan: AccountPlan, grid: Grid, employees, rng):
    h, n_days = grid.h_day, len(grid.days)
    admin = rng.poisson(plan.admin_rate * h**2 * grid.open_day)
    n_day = grid.employed.sum(axis=0)
    uses_app = employees["_uses_app"].to_numpy()[:, None]
    late = np.zeros(n_days, dtype=bool)
    if plan.archetype == "price_sensitive" and plan.decline_start is not None:
        late = grid.month_idx >= plan.decline_start
    trouble_day = grid.trouble.any(axis=0)
    return pd.DataFrame(
        {
            "_acc": plan.index,
            "usage_date": _dt(grid.days),
            "admin_logins": admin,
            "reports_viewed": rng.poisson(0.6 * admin),
            "scheduling_active": plan.scheduling_on[grid.month_idx],
            "absences_active": plan.absences_on[grid.month_idx],
            "documents_active": plan.documents_on[grid.month_idx],
            "messages_sent": rng.poisson(0.15 * n_day * h),
            "app_installs_total": (grid.employed & uses_app).sum(axis=0),
            "support_tickets": rng.poisson(0.01 + 0.04 * trouble_day + 0.05 * late),
        }
    )


def _device_health(plan: AccountPlan, grid: Grid, devices, dev_punches, rng):
    columns = ["_acc", "_dev", "health_date", "offline_hours", "failed_recognitions"]
    frames = []
    for k in range(len(devices)):
        row = devices.iloc[k]
        active = grid.days >= np.datetime64(row["installed_at"], "s")
        if not pd.isna(row["retired_at"]):
            active &= grid.days < np.datetime64(row["retired_at"], "s")
        dd = np.flatnonzero(active)
        n = len(dd)
        if not n:
            continue
        offline = np.where(rng.random(n) < 0.9, 0.0, np.minimum(rng.exponential(1.5, n), 24))
        boost = np.ones(n)
        for dev, _site, start, stop in plan.streaks:
            if dev != k:
                continue
            days = grid.days[dd]
            streak = (days >= start) & (days < stop)
            offline[streak] = 24 * rng.beta(8, 2, int(streak.sum()))
            boost[(days >= start - np.timedelta64(5, "D")) & (days < start)] = 5
        failed = rng.poisson((0.2 + 0.01 * dev_punches[k, dd]) * boost)
        frames.append(
            pd.DataFrame(
                {
                    "_acc": plan.index,
                    "_dev": k,
                    "health_date": _dt(grid.days[dd]),
                    "offline_hours": np.round(offline, 2),
                    "failed_recognitions": failed,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _billing(plan: AccountPlan, employees, snapshot: pd.Timestamp, rng):
    hired = employees["hired_at"].to_numpy()
    left = employees["left_at"].to_numpy()
    rows = []
    late_k = 0
    last = len(plan.months) - 1
    for m, month in enumerate(plan.months):
        period = month.start_time
        at = max(period, plan.signed_up_at)
        seats = int(
            ((hired <= np.datetime64(at)) & (np.isnat(left) | (left > np.datetime64(at)))).sum()
        )
        seats = max(1, seats)
        plan_name = "basic" if seats < 8 else "standard" if seats <= 25 else "premium"
        if period <= plan.trial_ends_at:
            rows.append((period, plan_name, seats, 0.0, 0.0, pd.NaT, pd.NA))
            continue
        mrr = float(seats * PLAN_PRICE[plan_name])
        paid_at: pd.Timestamp | None = None
        days_late: int | None = None
        in_decline = plan.decline_start is not None and m >= plan.decline_start
        if plan.archetype == "price_sensitive" and in_decline:
            late_k += 1
            days_late = 10 + 10 * late_k
            if m == last and plan.cancelled_at is not None:
                days_late = None
        else:
            days_late = 0 if rng.random() < 0.7 else int(rng.geometric(0.2))
        if days_late is not None:
            paid_at = period + pd.Timedelta(days=14 + days_late, hours=10)
            if paid_at >= snapshot:
                paid_at, days_late = None, None
        rows.append(
            (
                period,
                plan_name,
                seats,
                mrr,
                mrr,
                paid_at or pd.NaT,
                pd.NA if days_late is None else days_late,
            )
        )
    frame = pd.DataFrame(
        rows, columns=["period", "plan", "seats", "mrr", "invoice_amount", "paid_at", "days_late"]
    )
    frame.insert(0, "_acc", plan.index)
    frame["period"] = _dt(frame["period"])
    frame["paid_at"] = _dt(frame["paid_at"].to_numpy())
    frame["days_late"] = frame["days_late"].astype("Int64")
    return frame


def build_account_tables(
    plan: AccountPlan, config: SynthConfig, rng: np.random.Generator
) -> dict[str, pd.DataFrame]:
    employees, sites, wages = _organisation(plan, rng)
    devices = _devices(plan, sites, employees, rng)
    grid = _attendance_grid(plan, employees, rng)
    punches, hours, dev_punches = _punches_and_hours(plan, grid, employees, devices, rng)
    if plan.cancelled_at is not None:
        status = "cancelled"
    elif plan.trial_ends_at > plan.end:
        status = "trial"
    else:
        status = "active"
    accounts = pd.DataFrame(
        {
            "_acc": [plan.index],
            "industry": [plan.industry],
            "region": [plan.region],
            "signed_up_at": _dt([plan.signed_up_at]),
            "trial_ends_at": _dt([plan.trial_ends_at]),
            "status": [status],
            "cancelled_at": _dt([plan.cancelled_at if plan.cancelled_at is not None else pd.NaT]),
        }
    )
    return {
        "accounts": accounts,
        "billing_monthly": _billing(plan, employees, config.snapshot, rng),
        "employees": employees,
        "sites": sites,
        "punches": punches,
        "shifts": _shifts(plan, grid, employees, rng),
        "absence_requests": _absences(plan, employees, rng),
        "hours_daily": hours,
        "payroll_exports": _payroll(plan, employees, rng),
        "usage_daily": _usage(plan, grid, employees, rng),
        "devices": devices,
        "device_health_daily": _device_health(plan, grid, devices, dev_punches, rng),
        "wages": wages,
    }
