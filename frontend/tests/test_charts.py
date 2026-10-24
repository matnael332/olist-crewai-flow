"""Every chart builder produces a schema-valid Vega-Lite spec from synthetic inputs."""

import pandas as pd

from components import charts
from components.format import month_name, pct, points, times


def test_formatters():
    assert pct(0.067889) == "6.8%" and pct(None) == "n/a"
    assert points(-0.1257) == "-12.6 pts" and times(3.1719) == "3.2x"
    assert month_name("2017-11") == "Nov 2017"


def test_chart_builders_produce_valid_specs():
    candidates = {
        "baseline_prior": {"test": {"precision_at_top10pct": 0.1, "lift_at_top10pct": 1.0}},
        "logistic_regression": {"test": {"precision_at_top10pct": 0.3, "lift_at_top10pct": 3.0}},
    }
    roc = pd.DataFrame({"model": ["a", "a"], "fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})
    pr = pd.DataFrame({"model": ["a", "a"], "recall": [0.1, 0.9], "precision": [0.5, 0.2]})
    calibration = pd.DataFrame({"mean_predicted": [0.1, 0.5], "observed": [0.12, 0.45]})
    importance = [{"feature": "x", "importance_mean": 0.1, "importance_std": 0.01}]
    built = [
        charts.funnel([{"step": "raw", "rows": 1000}, {"step": "clean", "rows": 900}]),
        charts.precision_at_top(candidates, "logistic_regression", 0.1),
        charts.roc_curves(roc),
        charts.pr_curves(pr, 0.1),
        charts.calibration(calibration),
        charts.importance(importance),
        charts.risk_bullet(0.164, 0.1, 0.055, 0.17, charts.AMBER),
        charts.stage_timeline([{"stage": "a", "seconds": 1.0}, {"stage": "b", "seconds": 2.5}]),
    ]
    for chart in built:
        assert chart.to_dict()["$schema"].startswith("https://vega.github.io/schema/vega-lite/")
    assert charts.model_label("hist_gradient_boosting") == "Gradient boosting"
