"""EDA figures (matplotlib Figure API + seaborn, Agg backend, deterministic PNG metadata).

The Figure API (not pyplot) is thread-safe for Flow worker threads. Seaborn bar plots use
errorbar=None because bootstrap confidence intervals are random.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

DPI = 110
PALETTE = {"primary": "#2563eb", "accent": "#dc2626", "muted": "#94a3b8", "good": "#16a34a"}

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    return path.name


def monthly_orders_all_vs_delivered(summary: dict, out: Path) -> str:
    monthly = pd.DataFrame(summary["all_orders"]["monthly"])
    fig = Figure(figsize=(10, 4.2))
    ax = fig.subplots()
    ax.bar(monthly["month"], monthly["delivered"], color=PALETTE["primary"], label="delivered")
    ax.bar(
        monthly["month"],
        monthly["not_delivered"],
        bottom=monthly["delivered"],
        color=PALETTE["accent"],
        label="not delivered (canceled, shipped, unavailable, ...)",
    )
    ax.set_title("Orders per purchase month, all statuses")
    ax.set_ylabel("orders")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out / "monthly_orders_all_vs_delivered.png")


def late_rate_by_month(summary: dict, out: Path) -> str:
    monthly = pd.DataFrame(summary["delivered_population"]["monthly"])
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.plot(monthly["month"], monthly["late_rate"] * 100, marker="o", color=PALETTE["accent"])
    overall = summary["delivered_population"]["late_rate"] * 100
    ax.axhline(overall, color=PALETTE["muted"], linestyle="--", label=f"overall {overall:.1f}%")
    ax.set_title("Late-delivery rate by purchase month (delivered orders)")
    ax.set_ylabel("late orders (%)")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.legend(fontsize=8)
    return _save(fig, out / "late_rate_by_month.png")


def late_rate_by_state(summary: dict, out: Path) -> str:
    states = pd.DataFrame(summary["delivered_population"]["late_rate_by_state"])
    states["late_pct"] = states["late_rate"] * 100
    fig = Figure(figsize=(8, 7))
    ax = fig.subplots()
    sns.barplot(data=states, y="state", x="late_pct", ax=ax, color=PALETTE["accent"], errorbar=None)
    ax.set_title("Late-delivery rate by customer state (delivered orders)")
    ax.set_xlabel("late orders (%)")
    ax.set_ylabel("")
    return _save(fig, out / "late_rate_by_state.png")


def delivery_days_hist(clean: pd.DataFrame, out: Path) -> str:
    days = clean["delivery_days"].astype(float).clip(upper=60)
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    sns.histplot(x=days, bins=60, ax=ax, color=PALETTE["primary"])
    ax.axvline(days.median(), color=PALETTE["accent"], linestyle="--", label="median")
    ax.set_title("Days from purchase to delivery (capped at 60)")
    ax.set_xlabel("delivery days")
    ax.legend(fontsize=8)
    return _save(fig, out / "delivery_days_hist.png")


def review_score_vs_late(clean: pd.DataFrame, out: Path) -> str:
    reviewed = clean[clean["review_score"].notna()].copy()
    reviewed["delivery"] = reviewed["is_late"].astype(bool).map({True: "late", False: "on time"})
    reviewed["score"] = reviewed["review_score"].astype(int)
    shares = (
        reviewed.groupby(["delivery", "score"], sort=True).size()
        / reviewed.groupby("delivery", sort=True).size()
    ).reset_index(name="share")
    shares["share_pct"] = shares["share"] * 100
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    sns.barplot(
        data=shares,
        x="score",
        y="share_pct",
        hue="delivery",
        hue_order=["on time", "late"],
        palette=[PALETTE["good"], PALETTE["accent"]],
        ax=ax,
        errorbar=None,
    )
    ax.set_title("Review score distribution: late vs on-time orders")
    ax.set_ylabel("share of reviews (%)")
    ax.set_xlabel("review score")
    return _save(fig, out / "review_score_vs_late.png")


def gmv_by_category_top10(summary: dict, out: Path) -> str:
    cats = pd.DataFrame(summary["delivered_population"]["categories_top10_by_gmv"])
    cats["gmv_k"] = cats["gmv_brl"] / 1000
    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    sns.barplot(data=cats, y="category", x="gmv_k", ax=ax, color=PALETTE["primary"], errorbar=None)
    ax.set_title("Top 10 categories by GMV (delivered orders)")
    ax.set_xlabel("GMV (thousand BRL)")
    ax.set_ylabel("")
    return _save(fig, out / "gmv_by_category_top10.png")


def promised_vs_actual_by_month(summary: dict, out: Path) -> str:
    monthly = pd.DataFrame(summary["delivered_population"]["monthly"])
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.plot(
        monthly["month"],
        monthly["median_promised_days"],
        marker="o",
        color=PALETTE["primary"],
        label="median promised days",
    )
    ax.plot(
        monthly["month"],
        monthly["median_delivery_days"],
        marker="o",
        color=PALETTE["accent"],
        label="median actual days",
    )
    ax.set_title("Promised vs actual delivery time by purchase month")
    ax.set_ylabel("days")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.legend(fontsize=8)
    return _save(fig, out / "promised_vs_actual_by_month.png")


def make_eda_figures(summary: dict, clean: pd.DataFrame, figures_dir: str | Path) -> list[str]:
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        monthly_orders_all_vs_delivered(summary, out),
        late_rate_by_month(summary, out),
        late_rate_by_state(summary, out),
        delivery_days_hist(clean, out),
        review_score_vs_late(clean, out),
        gmv_by_category_top10(summary, out),
        promised_vs_actual_by_month(summary, out),
    ]
