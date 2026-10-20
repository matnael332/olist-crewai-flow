"""Insights: what happened, told with interactive charts (Data Analyst crew outputs)."""

import streamlit as st

from components import cards, charts, layout
from components.format import brl, month_name, num, pct, times
from services import artifacts as art

layout.page_header(
    "What happened",
    "Two layers: all orders, for business volume, and delivered orders, where lateness and reviews "
    "are measured. Hover any chart for exact values.",
    kicker="Insights · Data Analyst crew",
)
eda = art.eda()
if not eda:
    layout.empty_state("No EDA summary in artifacts/ yet.")
    st.stop()

ao, dp = eda["all_orders"], eda["delivered_population"]
headline = art.insights_headline()
if headline:
    st.info(f"**Headline from the Data Analyst crew.** {headline}", icon=":material/auto_awesome:")

# Volume ---------------------------------------------------------------------------------------
layout.section("Business volume", "All orders, every status")
peak, peak_day, second_day = ao["peak_month"], ao["peak_day"], ao["second_busiest_day"]
c1, c2, c3, c4 = st.columns(4)
cards.kpi(c1, "Orders (all statuses)", num(ao["n_orders"]))
cards.kpi(
    c2,
    "Never delivered",
    pct(ao["non_delivered_share"]),
    help="Canceled, unavailable, still shipped or processing at the snapshot.",
)
cards.kpi(c3, f"Peak month: {month_name(peak['month'])}", num(peak["orders"]))
cards.kpi(c4, f"Busiest day: {peak_day['date']}", num(peak_day["orders"]))
layout.chart_card(
    "Orders per purchase month",
    charts.monthly_orders(ao["monthly"], peak["month"]),
    f"Orders grew from {num(ao['orders_2017_01'])} in Jan 2017 to {num(peak['orders'])} in "
    f"{month_name(peak['month'])}. Black Friday alone brought {num(peak_day['orders'])} orders, "
    f"{times(peak_day['orders'] / second_day['orders'])} the next busiest day.",
)

# Delivery -------------------------------------------------------------------------------------
layout.section("Delivery performance", "Delivered orders, purchases Jan 2017 to Aug 2018")
delivery, monthly = dp["delivery"], dp["monthly"]
c1, c2, c3, c4 = st.columns(4)
cards.kpi(c1, "Late rate", pct(dp["late_rate"]), help=f"{num(dp['late_orders'])} late orders")
cards.kpi(c2, "Median delivery", f"{delivery['median_days']:.0f} days")
cards.kpi(c3, "Median promise", f"{delivery['median_promised_days']:.0f} days")
cards.kpi(c4, "90th percentile delivery", f"{delivery['p90_days']:.0f} days")
rates = [m["late_rate"] for m in monthly]
low = min(monthly, key=lambda m: m["late_rate"])
high = max(monthly, key=lambda m: m["late_rate"])
left, right = st.columns(2)
with left:
    layout.chart_card(
        "Late rate by purchase month",
        charts.late_rate_by_month(monthly, dp["late_rate"]),
        f"The late rate swings from {pct(min(rates))} ({month_name(low['month'])}) to "
        f"{pct(max(rates))} ({month_name(high['month'])}), so the model is tested on later months, "
        "never on a random split.",
    )
gaps = [m["median_promised_days"] - m["median_delivery_days"] for m in monthly]
gap_high = high["median_promised_days"] - high["median_delivery_days"]
with right:
    layout.chart_card(
        "Promised vs actual delivery time",
        charts.promised_vs_actual(monthly),
        f"The median promise beats the median delivery by {sum(gaps) / len(gaps):.0f} days on "
        f"average, but by only {gap_high:.0f} days in {month_name(high['month'])}, the month "
        "with the highest late rate.",
    )

# Geography ------------------------------------------------------------------------------------
layout.section("Geography", "Customer state; states with at least 0.4% of delivered orders")
min_orders = int(0.004 * dp["n_orders"])
states = sorted(
    (s for s in dp["late_rate_by_state"] if s["orders"] >= min_orders), key=lambda s: s["late_rate"]
)
worst3, best3 = states[::-1][:3], states[:3]
worst = ", ".join(f"{s['state']} {pct(s['late_rate'])}" for s in worst3)
best = ", ".join(f"{s['state']} {pct(s['late_rate'])}" for s in best3)
worst_regions = sorted({charts.region_of(s["state"]) for s in worst3})
where = (
    f"all in the {worst_regions[0]}"
    if len(worst_regions) == 1
    else "in the " + " and ".join(worst_regions)
)
layout.chart_card(
    "Late rate by customer state",
    charts.late_rate_by_state(dp["late_rate_by_state"], dp["late_rate"], min_orders),
    f"Worst: {worst}, {where}. Best: {best}. The model card flags this regional gap for "
    "fairness monitoring.",
)

# Customers ------------------------------------------------------------------------------------
layout.section("Customer impact", "Reviews of delivered orders")
reviews = dp["reviews"]
left, right = st.columns([3, 2])
with left:
    shares = art.review_shares()
    if shares is not None:
        layout.chart_card(
            "Review scores: late vs on-time orders",
            charts.review_distribution(shares),
            f"{pct(reviews['bad_review_rate_late'])} of late orders get 1-2 stars, against "
            f"{pct(reviews['bad_review_rate_on_time'])} of on-time orders.",
        )
with right:
    cards.kpi(st, "1-2 star reviews, late orders", pct(reviews["bad_review_rate_late"]))
    cards.kpi(st, "1-2 star reviews, on-time orders", pct(reviews["bad_review_rate_on_time"]))
    cards.kpi(
        st,
        "Mean score: late vs on time",
        f"{reviews['mean_score_late']:.1f} vs {reviews['mean_score_on_time']:.1f}",
    )

# Categories and payments ------------------------------------------------------------------------
layout.section("Categories and payments", "Delivered orders")
payments = dp["payments"]
left, right = st.columns([3, 2])
with left:
    layout.chart_card(
        "Top 10 categories by GMV",
        charts.top_categories(dp["categories_top10_by_gmv"]),
        f"Delivered GMV is {brl(dp['gmv_brl'], compact=True)} with an average order of "
        f"{brl(dp['aov_brl'])}; freight adds {pct(dp['freight_share_of_gmv'])} on top.",
    )
with right:
    layout.chart_card(
        "Payment mix",
        charts.payment_mix(payments["type_share"]),
        f"{pct(payments['share_more_than_one_installment'])} of orders are paid in installments "
        f"(median {payments['median_installments']:.0f}).",
    )

# Reports --------------------------------------------------------------------------------------
layout.section("Full reports", "Generated by the pipeline; narratives written by the crew")
c1, c2, _ = st.columns([1, 1, 2])
html, insights_md = art.load_text("eda_report.html"), art.load_text("insights.md")
if html:
    c1.download_button(
        "EDA report (HTML)",
        html,
        "eda_report.html",
        "text/html",
        icon=":material/download:",
        width="stretch",
    )
if insights_md:
    c2.download_button(
        "Insights (Markdown)",
        insights_md,
        "insights.md",
        "text/markdown",
        icon=":material/download:",
        width="stretch",
    )
    with st.expander("Read the full insights written by the Data Analyst crew"):
        st.markdown(insights_md)
layout.footer()
