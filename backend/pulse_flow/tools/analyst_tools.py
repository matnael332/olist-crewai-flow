"""Data Analyst crew tools. (No `from __future__ import annotations`: @tool reads annotations.)"""

import json

from crewai.tools import tool

from pulse_core.io import read_json, round_floats
from pulse_core.reporting.fmt import as_percentages  # noqa: F401  (re-exported for callers)
from pulse_flow.runctx import get_context


def _dump(payload: dict) -> str:
    return json.dumps(round_floats(as_percentages(payload), 4), sort_keys=True, ensure_ascii=False)


def _safe(build) -> str:
    try:
        return _dump({"ok": True, **build()})
    except Exception as exc:  # noqa: BLE001 - tools report errors instead of raising
        return _dump({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def ingest_view() -> dict:
    art = get_context().artifacts_dir
    profile = read_json(art / "raw_profile.json")
    report = read_json(art / "cleaning_report.json")
    return {
        "raw_tables": {
            name: {
                "rows": entry["rows"],
                "columns_with_nulls": len(entry["null_counts"]),
                "duplicate_keys": entry.get("duplicate_keys"),
            }
            for name, entry in profile.items()
        },
        "order_status_counts": profile["orders"]["status_counts"],
        "rows_in": report["rows_in"],
        "rows_out": report["rows_out"],
        "rules": [
            {"rule": r["rule"], "rows_removed": r["rows_removed"], "detail": r["detail"]}
            for r in report["rules"]
        ],
        "payment_reconciliation": report["payment_reconciliation"],
        "definitions": report["definitions"],
    }


def eda_view() -> dict:
    summary = read_json(get_context().artifacts_dir / "eda_summary.json")
    ao, dp = summary["all_orders"], summary["delivered_population"]
    floor = max(1, int(0.004 * dp["n_orders"]))
    states = [s for s in dp["late_rate_by_state"] if s["orders"] >= floor]
    return {
        "all_orders": {
            k: ao[k]
            for k in (
                "n_orders",
                "non_delivered_share",
                "status_counts",
                "peak_day",
                "second_busiest_day",
                "peak_month",
                "orders_2017_01",
                "mean_monthly_orders_2018_jan_aug",
            )
        },
        "delivered_orders": {
            **{
                k: dp[k]
                for k in (
                    "n_orders",
                    "n_customers",
                    "repeat_customer_share",
                    "gmv_brl",
                    "aov_brl",
                    "late_orders",
                    "late_rate",
                    "delivery",
                    "reviews",
                    "payments",
                    "freight_share_of_gmv",
                    "sellers",
                    "prevalence_by_split",
                )
            },
            "monthly_late_rate": [
                {"month": m["month"], "orders": m["orders"], "late_rate": m["late_rate"]}
                for m in dp["monthly"]
            ],
            "highest_late_rate_states": states[:5],
            "lowest_late_rate_states": states[-3:],
            "top_categories_by_gmv": dp["categories_top10_by_gmv"][:5],
        },
        "figures": summary["figures"],
    }


def contract_view() -> dict:
    contract = read_json(get_context().artifacts_dir / "dataset_contract.json")
    return {
        "contract_version": contract["contract_version"],
        "row_count": contract["row_count"],
        "primary_key": contract["primary_key"],
        "population": contract["population"],
        "columns": [
            {
                "name": c["name"],
                "dtype": c["dtype"],
                "nullable": c["nullable"],
                "available_at": c["available_at"],
            }
            for c in contract["columns"]
        ],
        "constraints": [
            {"id": c["id"], "description": c.get("description", "")}
            for c in contract["constraints"]
        ],
        "target": {
            "definition": contract["target"]["definition"],
            "expected_positive_rate": contract["target"]["expected_positive_rate"],
            "split_policy": contract["target"]["split_policy"],
        },
        "required_features": [f["name"] for f in contract["required_features"]],
        "forbidden_features": contract["forbidden_features"],
    }


@tool("ingest_and_clean", result_as_answer=True)
def ingest_and_clean() -> str:
    """Profile the nine raw Olist tables and return the cleaning report (rules R1-R10 with the rows
    each rule removed). The deterministic pipeline has already written clean_data.csv."""
    return _safe(ingest_view)


@tool("run_eda", result_as_answer=True)
def run_eda() -> str:
    """Return the EDA summary: an all-orders layer (business volume, every status) and a
    delivered-orders layer (late rates, delivery times, reviews, geography, categories)."""
    return _safe(eda_view)


@tool("draft_contract_schema", result_as_answer=True)
def draft_contract_schema() -> str:
    """Return the machine-generated part of the dataset contract: schema, constraints, target,
    split policy, required and forbidden features."""
    return _safe(contract_view)
