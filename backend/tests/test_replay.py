"""Replaying a test-window day: same scores as the stored test predictions, plus a backtest."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from olist_api.replay import ReplayError, backtest, orders_for_day, parse_day
from olist_api.validation import ALLOWED_FIELDS
from olist_core.columns import read_clean_data


def test_replayed_orders_score_exactly_like_the_stored_test_predictions(
    bundle, pipeline_out, busiest_test_day
):
    clean = read_clean_data(pipeline_out / "clean_data.csv")
    records, outcomes = orders_for_day(clean, busiest_test_day)
    assert len(records) == len(outcomes) > 1
    assert all(set(record) <= ALLOWED_FIELDS for record in records)
    valid, errors, _ = bundle.validator.validate_partial(records)
    assert not errors and len(valid) == len(records)
    probabilities = bundle.score_rows([row for _, row in valid])
    stored = pd.read_csv(pipeline_out / "predictions_test.csv", dtype={"order_id": str})
    stored = stored.set_index("order_id")[f"p_{bundle.meta['model_name']}"]
    expected = stored.loc[[record["order_id"] for record in records]].to_numpy()
    assert np.allclose(probabilities, expected, atol=1e-6)


def test_parse_day_accepts_only_test_window_dates():
    assert parse_day("2018-07-16") == date(2018, 7, 16)
    for value, status, code in [
        ("2018-13-01", 400, "invalid_date"),
        (None, 400, "invalid_date"),
        ("2018-04-30", 404, "out_of_window"),
        ("2018-08-01", 404, "out_of_window"),
    ]:
        with pytest.raises(ReplayError) as caught:
            parse_day(value)
        assert (caught.value.status, caught.value.code) == (status, code)


def test_a_day_without_orders_is_a_404(pipeline_out):
    clean = read_clean_data(pipeline_out / "clean_data.csv")
    with pytest.raises(ReplayError) as caught:
        orders_for_day(clean, date(2016, 1, 1))
    assert (caught.value.status, caught.value.code) == (404, "no_orders")


def test_backtest_counts_late_orders_per_level():
    predictions = [
        {"order_id": "a", "risk_level": "high"},
        {"order_id": "b", "risk_level": "high"},
        {"order_id": "c", "risk_level": "normal"},
    ]
    result = backtest(predictions, {"a": 1, "b": 0, "c": 0})
    assert result["high"] == {"orders": 2, "late": 1, "late_rate": 0.5}
    assert result["elevated"] == {"orders": 0, "late": 0, "late_rate": None}
    assert result["all"] == {"orders": 3, "late": 1, "late_rate": 0.3333}
