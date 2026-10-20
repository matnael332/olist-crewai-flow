"""Altair chart builders: one palette, tooltips everywhere, data from the artifacts only."""

from __future__ import annotations

import altair as alt
import pandas as pd

BLUE, RED, GREEN, AMBER = "#2563EB", "#DC2626", "#16A34A", "#F59E0B"
SLATE, LIGHT, VIOLET, NAVY = "#64748B", "#CBD5E1", "#7C3AED", "#0F172A"

REGIONS = {
    "North": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Northeast": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Center-West": ["DF", "GO", "MT", "MS"],
    "Southeast": ["ES", "MG", "RJ", "SP"],
    "South": ["PR", "RS", "SC"],
}
REGION_COLORS = {
    "North": VIOLET,
    "Northeast": RED,
    "Center-West": AMBER,
    "Southeast": BLUE,
    "South": GREEN,
}
MODEL_NAMES = {
    "logistic_regression": "Logistic regression",
    "hist_gradient_boosting": "Gradient boosting",
    "baseline_prior": "Baseline",
}
MODEL_COLORS = {"Logistic regression": BLUE, "Gradient boosting": RED, "Baseline": SLATE}

_MONTH = alt.X("month:O", title=None, axis=alt.Axis(labelAngle=-45))
_LEGEND_TOP = alt.Legend(orient="top", title=None, labelLimit=0)


def region_of(state: str) -> str:
    return next((name for name, states in REGIONS.items() if state in states), "Other")


def model_label(name: str) -> str:
    return MODEL_NAMES.get(name, name.replace("_", " "))


def _rule(value: float, label: str, axis: str = "y", color: str = SLATE) -> alt.LayerChart:
    """A dashed reference line with a small label."""
    base = alt.Chart(pd.DataFrame({"v": [value], "label": [label]}))
    line = base.mark_rule(strokeDash=[5, 4], color=color)
    if axis == "y":
        rule = line.encode(y="v:Q")
        text = base.mark_text(color=color, align="left", dx=4, dy=-7, fontSize=11).encode(
            y="v:Q", x=alt.value(0), text="label:N"
        )
    else:
        rule = line.encode(x="v:Q")
        text = base.mark_text(color=color, align="left", dx=4, dy=-4, fontSize=11).encode(
            x="v:Q", y=alt.value(-6), text="label:N"
        )
    return rule + text


def monthly_orders(monthly: list[dict], peak_month: str) -> alt.LayerChart:
    df = pd.DataFrame(monthly)
    long = df[["month", "delivered", "not_delivered"]].melt(
        id_vars=["month"],
        value_vars=["delivered", "not_delivered"],
        var_name="status",
        value_name="orders",
    )
    long["status"] = long["status"].map(
        {"delivered": "Delivered", "not_delivered": "Not delivered"}
    )
    bars = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=_MONTH,
            y=alt.Y("orders:Q", title="Orders", stack="zero"),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=["Delivered", "Not delivered"], range=[BLUE, AMBER]),
                legend=_LEGEND_TOP,
            ),
            order=alt.Order("status:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("month:O", title="Month"),
                alt.Tooltip("status:N", title="Status"),
                alt.Tooltip("orders:Q", title="Orders", format=","),
            ],
        )
    )
    peak = df.loc[df["month"] == peak_month, ["month", "orders"]]
    label = (
        alt.Chart(peak)
        .mark_text(dy=-9, fontWeight=600, color=NAVY, fontSize=11)
        .encode(x="month:O", y="orders:Q", text=alt.value("Black Friday"))
    )
    return (bars + label).properties(height=320)


def late_rate_by_month(monthly: list[dict], overall: float) -> alt.LayerChart:
    df = pd.DataFrame(monthly)[["month", "late_rate", "orders"]]
    line = (
        alt.Chart(df)
        .mark_line(point=True, color=RED)
        .encode(
            x=_MONTH,
            y=alt.Y("late_rate:Q", title="Late orders", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("month:O", title="Month"),
                alt.Tooltip("late_rate:Q", title="Late rate", format=".1%"),
                alt.Tooltip("orders:Q", title="Delivered orders", format=","),
            ],
        )
    )
    return (line + _rule(overall, f"overall {overall:.1%}")).properties(height=280)


def promised_vs_actual(monthly: list[dict]) -> alt.Chart:
    df = pd.DataFrame(monthly)[["month", "median_promised_days", "median_delivery_days"]]
    df = df.rename(columns={"median_promised_days": "Promised", "median_delivery_days": "Actual"})
    long = df.melt("month", var_name="series", value_name="days")
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=_MONTH,
            y=alt.Y("days:Q", title="Median days"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Promised", "Actual"], range=[SLATE, BLUE]),
                legend=_LEGEND_TOP,
            ),
            tooltip=[
                alt.Tooltip("month:O", title="Month"),
                alt.Tooltip("series:N", title="Median"),
                alt.Tooltip("days:Q", title="Days", format=".0f"),
            ],
        )
        .properties(height=280)
    )


def late_rate_by_state(states: list[dict], overall: float, min_orders: int) -> alt.LayerChart:
    df = pd.DataFrame(states)
    df = df[df["orders"] >= min_orders].copy()
    df["region"] = df["state"].map(region_of)
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("state:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
            x=alt.X("late_rate:Q", title="Late orders", axis=alt.Axis(format="%")),
            color=alt.Color(
                "region:N",
                scale=alt.Scale(domain=list(REGION_COLORS), range=list(REGION_COLORS.values())),
                legend=_LEGEND_TOP,
            ),
            tooltip=[
                alt.Tooltip("state:N", title="State"),
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("late_rate:Q", title="Late rate", format=".1%"),
                alt.Tooltip("orders:Q", title="Delivered orders", format=","),
                alt.Tooltip("median_delivery_days:Q", title="Median delivery (days)"),
            ],
        )
    )
    return (bars + _rule(overall, f"overall {overall:.1%}", axis="x", color=NAVY)).properties(
        height=alt.Step(20)
    )


def review_distribution(shares: pd.DataFrame) -> alt.Chart:
    """``shares``: columns score, delivery ('On time'/'Late'), share."""
    return (
        alt.Chart(shares)
        .mark_bar()
        .encode(
            x=alt.X("score:O", title="Review score (stars)", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("delivery:N", sort=["On time", "Late"]),
            y=alt.Y("share:Q", title="Share of reviews", axis=alt.Axis(format="%")),
            color=alt.Color(
                "delivery:N",
                scale=alt.Scale(domain=["On time", "Late"], range=[GREEN, RED]),
                legend=_LEGEND_TOP,
            ),
            tooltip=[
                alt.Tooltip("delivery:N", title="Delivery"),
                alt.Tooltip("score:O", title="Stars"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=300)
    )


def top_categories(categories: list[dict]) -> alt.Chart:
    df = pd.DataFrame(categories)
    df["category"] = df["category"].str.replace("_", " ")
    return (
        alt.Chart(df)
        .mark_bar(color=BLUE)
        .encode(
            y=alt.Y("category:N", sort="-x", title=None),
            x=alt.X("gmv_brl:Q", title="GMV (BRL)", axis=alt.Axis(format="~s")),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("gmv_brl:Q", title="GMV (BRL)", format=",.0f"),
                alt.Tooltip("orders:Q", title="Orders", format=","),
                alt.Tooltip("late_rate:Q", title="Late rate", format=".1%"),
            ],
        )
        .properties(height=300)
    )


def payment_mix(type_share: dict[str, float]) -> alt.Chart:
    df = pd.DataFrame(
        [{"type": k.replace("_", " "), "share": v} for k, v in type_share.items()]
    ).sort_values("share", ascending=False, kind="mergesort")
    return (
        alt.Chart(df)
        .mark_bar(color=SLATE)
        .encode(
            y=alt.Y("type:N", sort="-x", title=None),
            x=alt.X("share:Q", title="Share of orders", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("type:N", title="Payment type"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=170)
    )


def funnel(steps: list[dict]) -> alt.LayerChart:
    """``steps``: [{'step': str, 'rows': int}] from raw orders to the clean table."""
    df = pd.DataFrame(steps)
    df["order"] = range(len(df))
    base = alt.Chart(df).encode(
        y=alt.Y(
            "step:N",
            sort=alt.EncodingSortField("order"),
            title=None,
            axis=alt.Axis(labelLimit=240),
        ),
        x=alt.X("rows:Q", title="Orders"),
        tooltip=[
            alt.Tooltip("step:N", title="Step"),
            alt.Tooltip("rows:Q", title="Orders", format=","),
        ],
    )
    bars = base.mark_bar(color=BLUE)
    text = base.mark_text(align="right", dx=-6, color="white", fontWeight=600).encode(
        text=alt.Text("rows:Q", format=",")
    )
    return (bars + text).properties(height=46 * len(df))


def precision_at_top(candidates: dict, selected: str, base_rate: float) -> alt.LayerChart:
    df = pd.DataFrame(
        [
            {
                "model": model_label(name),
                "precision": c["test"]["precision_at_top10pct"],
                "lift": c["test"]["lift_at_top10pct"],
                "selected": name == selected,
            }
            for name, c in candidates.items()
        ]
    )
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X(
                "precision:Q",
                title="Late orders among the riskiest 10%",
                axis=alt.Axis(format="%"),
            ),
            color=alt.condition("datum.selected", alt.value(BLUE), alt.value(LIGHT)),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("precision:Q", title="Precision, riskiest 10%", format=".1%"),
                alt.Tooltip("lift:Q", title="Lift", format=".2f"),
            ],
        )
    )
    return (bars + _rule(base_rate, f"base rate {base_rate:.1%}", axis="x", color=NAVY)).properties(
        height=170
    )


def _model_color() -> alt.Color:
    return alt.Color(
        "model:N",
        scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
        legend=_LEGEND_TOP,
    )


def roc_curves(curves: pd.DataFrame) -> alt.LayerChart:
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[5, 4], color=LIGHT)
        .encode(x="x:Q", y="y:Q")
    )
    lines = (
        alt.Chart(curves)
        .mark_line()
        .encode(
            x=alt.X("fpr:Q", title="False positive rate"),
            y=alt.Y("tpr:Q", title="True positive rate"),
            color=_model_color(),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("fpr:Q", title="False positive rate", format=".2f"),
                alt.Tooltip("tpr:Q", title="True positive rate", format=".2f"),
            ],
        )
    )
    return (diagonal + lines).properties(height=320)


def pr_curves(curves: pd.DataFrame, prevalence: float) -> alt.LayerChart:
    visible = curves.loc[curves["recall"] >= 0.01, "precision"]
    top = min(1.0, max(float(visible.max()) * 1.25 if len(visible) else 1.0, prevalence * 3))
    lines = (
        alt.Chart(curves)
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X("recall:Q", title="Recall (share of late orders caught)"),
            y=alt.Y(
                "precision:Q",
                title="Precision",
                scale=alt.Scale(domain=[0, top], clamp=True),
                axis=alt.Axis(format="%"),
            ),
            color=_model_color(),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("recall:Q", title="Recall", format=".0%"),
                alt.Tooltip("precision:Q", title="Precision", format=".1%"),
            ],
        )
    )
    return (lines + _rule(prevalence, f"base rate {prevalence:.1%}")).properties(height=320)


def calibration(points: pd.DataFrame) -> alt.LayerChart:
    top = float(max(points["mean_predicted"].max(), points["observed"].max())) * 1.1
    x = alt.X(
        "mean_predicted:Q",
        title="Predicted late probability",
        scale=alt.Scale(domain=[0, top]),
        axis=alt.Axis(format="%"),
    )
    y = alt.Y(
        "observed:Q",
        title="Observed late rate",
        scale=alt.Scale(domain=[0, top]),
        axis=alt.Axis(format="%"),
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"mean_predicted": [0.0, top], "observed": [0.0, top]}))
        .mark_line(strokeDash=[5, 4], color=SLATE)
        .encode(x=x, y=y)
    )
    line = (
        alt.Chart(points)
        .mark_line(point=True, color=BLUE)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("mean_predicted:Q", title="Predicted", format=".1%"),
                alt.Tooltip("observed:Q", title="Observed", format=".1%"),
            ],
        )
    )
    return (diagonal + line).properties(height=300)


def importance(features: list[dict], top: int = 12) -> alt.LayerChart:
    df = pd.DataFrame(features).head(top)
    df["low"] = df["importance_mean"] - df["importance_std"]
    df["high"] = df["importance_mean"] + df["importance_std"]
    base = alt.Chart(df).encode(
        y=alt.Y(
            "feature:N", sort=None, title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)
        )
    )
    bars = base.mark_bar(color=BLUE).encode(
        x=alt.X("importance_mean:Q", title="Drop in PR-AUC when shuffled"),
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("importance_mean:Q", title="Mean drop", format=".4f"),
            alt.Tooltip("importance_std:Q", title="Std", format=".4f"),
        ],
    )
    whiskers = base.mark_rule(color=NAVY).encode(x="low:Q", x2="high:Q")
    return (bars + whiskers).properties(height=alt.Step(22))


def risk_bullet(
    probability: float, base_rate: float, threshold: float, top10_cutoff: float, color: str
) -> alt.LayerChart:
    xmax = max(0.3, probability * 1.25, top10_cutoff * 1.25)
    bar = (
        alt.Chart(pd.DataFrame({"label": ["This order"], "p": [probability]}))
        .mark_bar(size=24, color=color, opacity=0.85, cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "p:Q",
                title=None,
                scale=alt.Scale(domain=[0, xmax]),
                axis=alt.Axis(format="%", grid=False),
            ),
            y=alt.Y("label:N", title=None, axis=None),
            tooltip=[alt.Tooltip("p:Q", title="Late probability", format=".1%")],
        )
    )
    marks = pd.DataFrame(
        {
            "x": [base_rate, threshold, top10_cutoff],
            "name": ["Base rate", "Threshold", "Top 10%"],
        }
    )
    rules = (
        alt.Chart(marks)
        .mark_rule(strokeWidth=3)
        .encode(
            x="x:Q",
            color=alt.Color(
                "name:N",
                scale=alt.Scale(domain=list(marks["name"]), range=[SLATE, NAVY, VIOLET]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("name:N", title="Marker"),
                alt.Tooltip("x:Q", title="Value", format=".1%"),
            ],
        )
    )
    return (bar + rules).properties(height=110)


def stage_timeline(stages: list[dict]) -> alt.LayerChart:
    """``stages``: [{'stage': str, 'seconds': float}] in run order."""
    df = pd.DataFrame(stages)
    df["order"] = range(len(df))
    base = alt.Chart(df).encode(
        y=alt.Y(
            "stage:N",
            sort=alt.EncodingSortField("order"),
            title=None,
            axis=alt.Axis(labelLimit=240),
        ),
        x=alt.X(
            "seconds:Q",
            title="Seconds",
            scale=alt.Scale(domain=[0, float(df["seconds"].max()) * 1.18]),
        ),
        tooltip=[
            alt.Tooltip("stage:N", title="Stage"),
            alt.Tooltip("seconds:Q", title="Seconds", format=".1f"),
        ],
    )
    bars = base.mark_bar(color=BLUE)
    text = base.mark_text(align="left", dx=4, color=NAVY).encode(
        text=alt.Text("seconds:Q", format=".1f")
    )
    return (bars + text).properties(height=42 * len(df) + 20)
