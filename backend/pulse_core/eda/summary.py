"""Two-layer EDA summary (eda_summary.json).

- all_orders: computed from the raw orders table (every status). Business-volume numbers such as
  the Black Friday peak live here, because they count orders that were never delivered.
- delivered_population: computed from clean_data.csv (delivered orders only). Lateness, delivery
  times and reviews live here, and every late rate is "among delivered orders".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse_core.policy import SPLITS, assign_split
from pulse_core.sorting import stable_sort


def _f(value) -> float:
    return float(value) if pd.notna(value) else None


def all_orders_layer(orders: pd.DataFrame) -> dict:
    df = orders.copy()
    df["month"] = df["order_purchase_timestamp"].dt.strftime("%Y-%m")
    df["day"] = df["order_purchase_timestamp"].dt.strftime("%Y-%m-%d")
    df["delivered"] = df["order_status"] == "delivered"
    monthly = (
        df.groupby("month", sort=True)
        .agg(orders=("order_id", "size"), delivered=("delivered", "sum"))
        .reset_index()
    )
    monthly["not_delivered"] = monthly["orders"] - monthly["delivered"]
    monthly["non_delivered_share"] = monthly["not_delivered"] / monthly["orders"]
    daily = df.groupby("day", sort=True).size().reset_index(name="orders")
    daily = stable_sort(daily, ["orders", "day"], ascending=[False, True])
    peak_month = stable_sort(monthly, ["orders", "month"], ascending=[False, True]).iloc[0]
    status = df["order_status"].value_counts(sort=False)
    status = {str(k): int(v) for k, v in sorted(status.items(), key=lambda kv: (-kv[1], kv[0]))}
    in_2018 = monthly[(monthly["month"] >= "2018-01") & (monthly["month"] <= "2018-08")]
    first_2017 = monthly.loc[monthly["month"] == "2017-01", "orders"]
    return {
        "n_orders": int(len(df)),
        "purchase_range": [
            df["order_purchase_timestamp"].min().isoformat(),
            df["order_purchase_timestamp"].max().isoformat(),
        ],
        "status_counts": status,
        "non_delivered_share": _f(1 - df["delivered"].mean()),
        "monthly": [
            {
                "month": r.month,
                "orders": int(r.orders),
                "delivered": int(r.delivered),
                "not_delivered": int(r.not_delivered),
                "non_delivered_share": _f(r.non_delivered_share),
            }
            for r in monthly.itertuples()
        ],
        "peak_day": {"date": daily.iloc[0]["day"], "orders": int(daily.iloc[0]["orders"])},
        "second_busiest_day": {
            "date": daily.iloc[1]["day"],
            "orders": int(daily.iloc[1]["orders"]),
        },
        "peak_month": {"month": peak_month["month"], "orders": int(peak_month["orders"])},
        "orders_2017_01": int(first_2017.iloc[0]) if len(first_2017) else None,
        "mean_monthly_orders_2018_jan_aug": _f(in_2018["orders"].mean()),
    }


def delivered_layer(clean: pd.DataFrame) -> dict:
    df = clean
    n = len(df)
    late = df["is_late"].astype(bool)
    monthly = (
        df.assign(late=late)
        .groupby("purchase_month", sort=True)
        .agg(
            orders=("order_id", "size"),
            gmv=("total_price", "sum"),
            late_rate=("late", "mean"),
            median_delivery_days=("delivery_days", "median"),
            median_promised_days=("promised_days", "median"),
        )
        .reset_index()
    )
    by_state = (
        df.assign(late=late)
        .groupby("customer_state", sort=True)
        .agg(
            orders=("order_id", "size"),
            late_rate=("late", "mean"),
            median_delivery_days=("delivery_days", "median"),
        )
        .reset_index()
    )
    by_state = stable_sort(by_state, ["late_rate", "customer_state"], ascending=[False, True])
    cats = (
        df.assign(late=late)
        .groupby("primary_category", sort=True)
        .agg(orders=("order_id", "size"), gmv=("total_price", "sum"), late_rate=("late", "mean"))
        .reset_index()
    )
    cats = stable_sort(cats, ["gmv", "primary_category"], ascending=[False, True]).head(10)
    states = (
        df.groupby("customer_state", sort=True)
        .agg(orders=("order_id", "size"), gmv=("total_price", "sum"))
        .reset_index()
    )
    states = stable_sort(states, ["orders", "customer_state"], ascending=[False, True]).head(10)
    reviewed = df[df["review_score"].notna()]
    scores = reviewed["review_score"].astype(int)
    bad = scores <= 2
    late_reviewed = reviewed["is_late"].astype(bool)
    payments = df["payment_type"].value_counts(normalize=True, sort=False)
    sellers = df.groupby("primary_seller_id", sort=True)["total_price"].sum()
    top_sellers = np.sort(sellers.to_numpy())[::-1][:10].sum()
    customers = df.groupby("customer_unique_id", sort=True).size()
    split = assign_split(df["purchase_ts"])
    prevalence = {}
    for name in SPLITS:
        mask = split == name
        prevalence[name] = {"orders": int(mask.sum()), "late_rate": _f(late[mask].mean())}
    gmv = float(df["total_price"].sum())
    return {
        "n_orders": int(n),
        "n_customers": int(customers.size),
        "repeat_customer_share": _f((customers > 1).mean()),
        "purchase_range": [
            df["purchase_ts"].min().isoformat(),
            df["purchase_ts"].max().isoformat(),
        ],
        "gmv_brl": gmv,
        "freight_brl": float(df["total_freight"].sum()),
        "aov_brl": gmv / n,
        "median_order_value_brl": _f(df["total_price"].median()),
        "late_orders": int(late.sum()),
        "late_rate": _f(late.mean()),
        "delivery": {
            "median_days": _f(df["delivery_days"].median()),
            "mean_days": _f(df["delivery_days"].mean()),
            "p90_days": _f(df["delivery_days"].quantile(0.9)),
            "median_promised_days": _f(df["promised_days"].median()),
            "mean_slack_days": _f((df["promised_days"] - df["delivery_days"]).mean()),
        },
        "monthly": [
            {
                "month": r.purchase_month,
                "orders": int(r.orders),
                "gmv_brl": _f(r.gmv),
                "late_rate": _f(r.late_rate),
                "median_delivery_days": _f(r.median_delivery_days),
                "median_promised_days": _f(r.median_promised_days),
            }
            for r in monthly.itertuples()
        ],
        "late_rate_by_state": [
            {
                "state": r.customer_state,
                "orders": int(r.orders),
                "late_rate": _f(r.late_rate),
                "median_delivery_days": _f(r.median_delivery_days),
            }
            for r in by_state.itertuples()
        ],
        "reviews": {
            "coverage": _f(len(reviewed) / n),
            "score_distribution": {
                str(k): int(v) for k, v in sorted(scores.value_counts(sort=False).items())
            },
            "bad_review_rate": _f(bad.mean()),
            "bad_review_rate_late": _f(bad[late_reviewed].mean()),
            "bad_review_rate_on_time": _f(bad[~late_reviewed].mean()),
            "mean_score_late": _f(scores[late_reviewed].mean()),
            "mean_score_on_time": _f(scores[~late_reviewed].mean()),
        },
        "categories_top10_by_gmv": [
            {
                "category": r.primary_category,
                "orders": int(r.orders),
                "gmv_brl": _f(r.gmv),
                "late_rate": _f(r.late_rate),
            }
            for r in cats.itertuples()
        ],
        "states_top10_by_orders": [
            {"state": r.customer_state, "orders": int(r.orders), "gmv_brl": _f(r.gmv)}
            for r in states.itertuples()
        ],
        "payments": {
            "type_share": {str(k): _f(v) for k, v in sorted(payments.items())},
            "median_installments": _f(df["max_installments"].median()),
            "share_more_than_one_installment": _f((df["max_installments"] > 1).mean()),
        },
        "freight_share_of_gmv": _f(df["total_freight"].sum() / gmv),
        "sellers": {
            "n_sellers": int(sellers.size),
            "top10_gmv_share": _f(top_sellers / gmv),
        },
        "prevalence_by_split": prevalence,
    }


def compute_eda_summary(
    raw_orders: pd.DataFrame, clean: pd.DataFrame, cleaning_report: dict, figures: list[str]
) -> dict:
    return {
        "all_orders": all_orders_layer(raw_orders),
        "delivered_population": delivered_layer(clean),
        "cleaning": {
            "rows_in": cleaning_report["rows_in"],
            "rows_out": cleaning_report["rows_out"],
            "rules": [
                {"rule": r["rule"], "rows_removed": r["rows_removed"]}
                for r in cleaning_report["rules"]
            ],
        },
        "figures": figures,
    }
