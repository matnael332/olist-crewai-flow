"""Deterministic baseline narratives, computed from the numbers the pipeline produced.

In deterministic mode these are the narratives. In LLM mode the agents' narratives are used for the
prose sections, while factual items (contract assumptions, model-card ethics) keep these baselines
and the LLM's items are appended, so every mode yields spec-complete artifacts.
"""

from __future__ import annotations

from pulse_core.narratives import (
    CleaningNarrative,
    ColumnNote,
    ContractNarrative,
    Finding,
    Insights,
)
from pulse_core.policy import SNAPSHOT_DATE
from pulse_core.reporting.fmt import brl, num, pct


def _rule(report: dict, name: str) -> dict:
    return next(r for r in report["rules"] if r["rule"].startswith(name))


def cleaning_template(report: dict) -> CleaningNarrative:
    r1, r2, r3, r4 = (_rule(report, k) for k in ("R1", "R2", "R3", "R4"))
    r5, r6, r7, r8 = (_rule(report, k) for k in ("R5", "R6", "R7", "R8"))
    removed = ", ".join(f"{k} {v:,}" for k, v in r1["detail"]["removed_by_status"].items())
    recon = report["payment_reconciliation"]
    return CleaningNarrative(
        summary=(
            f"The nine raw Olist tables ({num(report['rows_in'])} orders) were reduced to "
            f"{num(report['rows_out'])} delivered orders, one row per order, purchased between "
            "2017-01-01 and 2018-08-31. Items, payments and reviews were aggregated per order before "
            "joining so revenue is never double-counted."
        ),
        issues_found=[
            f"{num(r1['rows_removed'])} orders were not delivered ({removed}).",
            f"{num(r2['rows_removed'])} orders were marked delivered but had no delivery date.",
            f"{num(r3['rows_removed'])} delivered orders fell outside the 2017-01..2018-08 window.",
            f"{num(r4['detail']['carrier_before_purchase_nulled'])} carrier hand-overs were dated "
            f"before purchase and {num(r4['detail']['carrier_after_delivery_nulled'])} after delivery.",
            f"{num(r5['detail']['orders_with_unknown_category'])} orders have a product without a "
            "category; untranslated categories: "
            f"{', '.join(r5['detail']['untranslated_categories_kept']) or 'none'}.",
            f"{num(r6['detail']['invalid_payment_rows_dropped'])} payment rows were typed "
            "'not_defined' or had a non-positive value.",
            f"{num(r7['detail']['orders_with_multiple_reviews'])} orders had more than one review.",
            f"{num(r8['detail']['distance_unknown'])} orders have a zip prefix without coordinates.",
        ],
        decisions=[
            "Kept only delivered orders: lateness is undefined for orders that never arrived.",
            "Nulled out-of-sequence carrier timestamps instead of dropping orders; these "
            "post-purchase fields are never used as features.",
            "Took category and seller from the first order item; a missing category becomes "
            "'unknown' and untranslated categories keep their Portuguese name.",
            "Used the type of the largest payment as the order's payment type and summed payments.",
            "Kept the latest review per order; review text is never loaded.",
            "Approximated customer-seller distance from median zip-prefix coordinates.",
        ],
        caveats=[
            "The population is delivered orders only, so delivery failures are not represented.",
            f"payment_total differs from price + freight for "
            f"{num(recon['paid_more_than_price_plus_freight'] + recon['paid_less_than_price_plus_freight'])}"
            " orders (vouchers and installment interest).",
            "Distances use zip-prefix centroids, an approximation of the real addresses.",
        ],
    )


def _material_months(dp: dict) -> list[dict]:
    """Months with at least a quarter of the median monthly volume (ignores sparse edges)."""
    counts = sorted(m["orders"] for m in dp["monthly"])
    floor = max(1, counts[len(counts) // 2] // 4)
    return [m for m in dp["monthly"] if m["orders"] >= floor and m["late_rate"] is not None]


def _top_late_states(dp: dict, k: int = 3) -> tuple[list[dict], list[dict]]:
    """States with at least 0.4% of delivered orders, highest and lowest late rates."""
    floor = max(1, int(0.004 * dp["n_orders"]))
    eligible = [s for s in dp["late_rate_by_state"] if s["orders"] >= floor]
    return eligible[:k], eligible[-k:]


def _state_list(states: list[dict]) -> str:
    return ", ".join(f"{s['state']} {pct(s['late_rate'])}" for s in states)


def insights_template(summary: dict) -> Insights:
    ao, dp = summary["all_orders"], summary["delivered_population"]
    rv, dl = dp["reviews"], dp["delivery"]
    monthly = _material_months(dp)
    worst = max(monthly, key=lambda m: m["late_rate"])
    best = min(monthly, key=lambda m: m["late_rate"])
    high, low = _top_late_states(dp)
    top_cat = dp["categories_top10_by_gmv"][0]
    return Insights(
        headline=(
            f"Orders grew from {num(ao['orders_2017_01'])} in January 2017 to "
            f"{num(ao['peak_month']['orders'])} in {ao['peak_month']['month']}; among delivered "
            f"orders {pct(dp['late_rate'])} arrived late, and late orders got 1-2 star reviews "
            f"{pct(rv['bad_review_rate_late'], 0)} of the time versus "
            f"{pct(rv['bad_review_rate_on_time'], 0)} when on time."
        ),
        key_findings=[
            Finding(
                title="Fast growth with a Black Friday spike",
                evidence=(
                    f"All orders: {num(ao['orders_2017_01'])} in 2017-01, peak month "
                    f"{ao['peak_month']['month']} with {num(ao['peak_month']['orders'])}, about "
                    f"{num(ao['mean_monthly_orders_2018_jan_aug'])} per month in 2018; busiest day "
                    f"{ao['peak_day']['date']} with {num(ao['peak_day']['orders'])} orders vs "
                    f"{num(ao['second_busiest_day']['orders'])} on the next busiest day."
                ),
                business_meaning="Capacity and carrier contracts must scale ahead of November.",
            ),
            Finding(
                title="Late deliveries destroy satisfaction",
                evidence=(
                    f"1-2 star reviews: {pct(rv['bad_review_rate_late'])} of late orders vs "
                    f"{pct(rv['bad_review_rate_on_time'])} of on-time orders; mean score "
                    f"{rv['mean_score_late']:.2f} vs {rv['mean_score_on_time']:.2f}."
                ),
                business_meaning="Preventing or announcing a delay protects ratings and retention.",
            ),
            Finding(
                title="Promises are generous, yet lateness persists",
                evidence=(
                    f"Median promised {num(dl['median_promised_days'])} days vs median actual "
                    f"{num(dl['median_days'])} days; still {pct(dp['late_rate'])} late "
                    f"({num(dp['late_orders'])} orders)."
                ),
                business_meaning="Lateness is concentrated in specific orders, so it is predictable.",
            ),
            Finding(
                title="Lateness depends on geography",
                evidence=(
                    f"Highest late rates: {_state_list(high)}; lowest: {_state_list(low)} "
                    "(states with at least 0.4% of delivered orders)."
                ),
                business_meaning="Regional logistics, not only sellers, drive delays.",
            ),
            Finding(
                title="Lateness is not stable over time",
                evidence=(
                    f"Monthly late rate ranged from {pct(best['late_rate'])} ({best['month']}) to "
                    f"{pct(worst['late_rate'])} ({worst['month']})."
                ),
                business_meaning="Thresholds and models need recent data and periodic re-tuning.",
            ),
            Finding(
                title="Revenue is concentrated in a few categories",
                evidence=(
                    f"Top category {top_cat['category']} generated {brl(top_cat['gmv_brl'])} of "
                    f"{brl(dp['gmv_brl'])} delivered GMV; AOV {brl(dp['aov_brl'])}."
                ),
                business_meaning="Delivery improvements in top categories protect most revenue.",
            ),
        ],
        recommendations=[
            "Score every order for late-delivery risk at checkout and proactively notify or "
            "reprioritize high-risk orders.",
            "Prioritize logistics improvements for the states with the highest late rates.",
            "Plan carrier capacity ahead of the November peak.",
            "Track the monthly late rate among delivered orders as an operational KPI.",
        ],
        data_caveats=[
            f"Late rates are measured among delivered orders only; {pct(ao['non_delivered_share'])} "
            "of all orders never reached 'delivered' status.",
            f"The data snapshot ends {SNAPSHOT_DATE}; the last purchase months are incomplete.",
            f"{pct(rv['coverage'])} of delivered orders have a review.",
        ],
        modeling_implications=[
            f"The target is imbalanced ({pct(dp['late_rate'])} positive): prefer PR-AUC and "
            "precision at the top of the ranking over accuracy.",
            "The late rate drifts over time, so evaluate on a later time window, never a random split.",
            "Only checkout-time information may be used: delivery dates and reviews are leakage.",
            "Customer state, promised days and freight are promising predictors.",
        ],
    )


def contract_template(summary: dict, report: dict) -> ContractNarrative:
    ao, dp = summary["all_orders"], summary["delivered_population"]
    monthly = _material_months(dp)
    worst = max(monthly, key=lambda m: m["late_rate"])
    best = min(monthly, key=lambda m: m["late_rate"])
    distance_unknown = _rule(report, "R8")["detail"]["distance_unknown"]
    return ContractNarrative(
        assumptions=[
            "Grain: one row per delivered order; order_id is unique.",
            "Lateness is counted in calendar days from the purchase date.",
            "All amounts are in Brazilian reais (BRL).",
            "The estimated delivery date was shown to the customer at checkout.",
            "The population is delivered orders purchased 2017-01-01..2018-08-31.",
        ],
        business_rules=[
            "An order is late if and only if delivery_days > promised_days.",
            "Category and seller come from the first order item.",
            "The payment type is the type of the largest payment.",
            "Only the latest review per order counts.",
        ],
        column_notes=[
            ColumnNote(
                name="promised_days",
                note="Legitimate feature: the delivery estimate is shown at purchase.",
            ),
            ColumnNote(
                name="payment_total",
                note="Final charged amount, known at checkout; includes vouchers and interest.",
            ),
            ColumnNote(
                name="distance_km",
                note=f"From checkout-known zip prefixes; null for {distance_unknown:,} orders.",
            ),
            ColumnNote(
                name="delivered_carrier_ts",
                note="Nulled when out of sequence; post-purchase, never a feature.",
            ),
            ColumnNote(
                name="customer_unique_id",
                note="Pseudonymous public hash; never a feature; aggregate before display.",
            ),
        ],
        known_limitations=[
            f"Delivered-only population understates delivery failure: "
            f"{pct(ao['non_delivered_share'])} of all orders never reached 'delivered'.",
            f"Right-censoring: the snapshot ends {SNAPSHOT_DATE}; August 2018 is a censored buffer "
            "excluded from evaluation.",
            f"The late rate is non-stationary ({pct(best['late_rate'])} in {best['month']} vs "
            f"{pct(worst['late_rate'])} in {worst['month']}).",
            "Single marketplace, 2017-2018 data: patterns may not transfer to today.",
        ],
    )
