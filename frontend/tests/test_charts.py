"""Every chart builder produces a schema-valid Vega-Lite spec from the committed artifacts."""

import pandas as pd

from components import charts
from components.format import brl, month_name, pct, points, times


def test_formatters():
    assert pct(0.067889) == "6.8%" and pct(None) == "n/a"
    assert brl(13179777.95, compact=True) == "BRL 13.18M" and brl(137.0) == "BRL 137.00"
    assert points(-0.1257) == "-12.6 pts" and times(3.1719) == "3.2x"
    assert month_name("2017-11") == "Nov 2017"


def test_regions_cover_all_27_states():
    states = [s for members in charts.REGIONS.values() for s in members]
    assert len(states) == len(set(states)) == 27


def test_chart_builders_produce_valid_specs():
    from services import artifacts as art

    eda, metrics = art.eda(), art.metrics()
    ao, dp = eda["all_orders"], eda["delivered_population"]
    chosen = metrics["selected_model"]
    roc, pr, calibration, prevalence = art.model_curves(tuple(metrics["candidates"]), chosen)
    for frame in (roc, pr):
        frame["model"] = frame["model"].map(charts.model_label)
    shares = art.review_shares()
    built = [
        charts.monthly_orders(ao["monthly"], ao["peak_month"]["month"]),
        charts.late_rate_by_month(dp["monthly"], dp["late_rate"]),
        charts.promised_vs_actual(dp["monthly"]),
        charts.late_rate_by_state(dp["late_rate_by_state"], dp["late_rate"], 385),
        charts.review_distribution(shares),
        charts.top_categories(dp["categories_top10_by_gmv"]),
        charts.payment_mix(dp["payments"]["type_share"]),
        charts.funnel([{"step": "raw", "rows": 99441}, {"step": "clean", "rows": 96203}]),
        charts.precision_at_top(metrics["candidates"], chosen, prevalence),
        charts.roc_curves(roc),
        charts.pr_curves(pr, prevalence),
        charts.calibration(calibration),
        charts.importance(metrics["permutation_importance"]["features"]),
        charts.risk_bullet(0.164, prevalence, 0.055, 0.17, charts.AMBER),
        charts.stage_timeline([{"stage": "a", "seconds": 1.0}, {"stage": "b", "seconds": 2.5}]),
    ]
    for chart in built:
        assert chart.to_dict()["$schema"].startswith("https://vega.github.io/schema/vega-lite/")
    assert set(roc["model"]) == {"Logistic regression", "Gradient boosting", "Baseline"}
    assert isinstance(shares, pd.DataFrame) and set(shares["delivery"]) == {"Late", "On time"}
