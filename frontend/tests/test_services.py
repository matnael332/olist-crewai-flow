"""Scoring service: presets, risk levels and what-if scenarios on the committed model."""

import pytest


@pytest.fixture
def scoring():
    from services import scoring as module

    return module


def test_presets_score_through_the_contract(scoring):
    records = [scoring.to_record(order) for order in scoring.PRESETS.values()]
    predictions, errors = scoring.score(records)
    assert not errors and len(predictions) == 2
    far, near = (p["late_probability"] for p in predictions)
    assert 0 < near < far < 1  # same order, long-distance shipping is riskier


def test_risk_levels_follow_the_api_flags(scoring):
    level = scoring.risk_level
    assert level({"top_10pct_risk": True, "predicted_late": True}, 0.055).label == "High risk"
    assert level({"top_10pct_risk": False, "predicted_late": True}, 0.055).label == "Elevated risk"
    normal = level({"top_10pct_risk": False, "predicted_late": False}, 0.055)
    assert normal.label == "Normal risk" and "5.5%" in normal.message


def test_what_if_rescores_one_change_at_a_time(scoring):
    record = scoring.to_record(scoring.PRESETS["Long distance: SP to BA"])
    (prediction,), errors = scoring.score([record])
    assert not errors
    rows = scoring.what_if(record, prediction["late_probability"])
    assert [r["scenario"] for r in rows] == [
        "Seller in the customer's state",
        "Promise 7 more days",
        "Pay by credit card",
    ]
    by_name = {r["scenario"]: r for r in rows}
    assert by_name["Seller in the customer's state"]["delta"] < 0
    assert by_name["Promise 7 more days"]["delta"] < 0
    for row in rows:
        assert row["probability"] - prediction["late_probability"] == pytest.approx(row["delta"])


def test_what_if_skips_changes_that_change_nothing(scoring):
    order = {**scoring.PRESETS["Same state: SP to SP"], "payment_type": "credit_card"}
    record = scoring.to_record(order)
    (prediction,), _ = scoring.score([record])
    rows = scoring.what_if(record, prediction["late_probability"])
    assert [r["scenario"] for r in rows] == ["Promise 7 more days"]
