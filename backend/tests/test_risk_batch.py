"""Risk policy and batch scoring: levels, actions, partial success and file errors."""

import pandas as pd
import pytest

from pulse_api.batch import MAX_BATCH_ROWS, BatchError, read_orders_csv, score_csv
from pulse_api.risk import ACTIONS, LEVELS, RiskPolicy


def _csv(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def test_levels_follow_the_threshold_and_the_top10_cutoff():
    policy = RiskPolicy(threshold=0.055, top10_cutoff=0.17)
    assert policy.level(0.17) == "high" and policy.level(0.1699) == "elevated"
    assert policy.level(0.055) == "elevated" and policy.level(0.0549) == "normal"
    assert policy.action("high") == ACTIONS["high"]
    assert policy.describe() == {"elevated_from": 0.055, "high_from": 0.17}


def test_a_cutoff_below_the_threshold_never_makes_high_easier():
    policy = RiskPolicy(threshold=0.2, top10_cutoff=0.1)
    assert policy.level(0.15) == "normal" and policy.level(0.2) == "high"


def test_customer_message_drafts_exist_only_for_high_risk():
    policy = RiskPolicy(threshold=0.055, top10_cutoff=0.17)
    draft = policy.customer_message("high", "o-42", "BA")
    assert draft.startswith("Hi! Your order o-42 is on its way to BA.")
    assert policy.customer_message("elevated", "o-42", "BA") == ""


def test_predict_echoes_order_id_and_adds_level_and_action(bundle, api_record):
    predictions, errors, _ = bundle.predict([{**api_record, "order_id": "o-1"}, api_record])
    assert not errors
    first, second = predictions
    assert first["order_id"] == "o-1" and "order_id" not in second
    assert first["risk_level"] in LEVELS
    assert first["recommended_action"] == ACTIONS[first["risk_level"]]


def test_order_id_is_validated(bundle, api_record):
    _, errors, _ = bundle.predict([{**api_record, "order_id": "x" * 65}])
    assert errors and errors[0]["field"] == "order_id"


def test_batch_scores_valid_rows_and_reports_the_bad_ones(bundle, api_record):
    rows = [
        {**api_record, "order_id": "a"},
        {**api_record, "order_id": "b", "customer_state": "XX"},
        {**api_record, "order_id": "c", "total_price": "abc"},
    ]
    result = score_csv(bundle, _csv(rows))
    assert result.summary["rows"] == 3
    assert result.summary["scored"] == 1 and result.summary["rejected"] == 2
    assert {e["row"] for e in result.errors} == {2, 3}
    assert {e["field"] for e in result.errors} == {"customer_state", "total_price"}
    assert result.predictions[0]["order_id"] == "a"


def test_batch_ranks_by_risk_and_drafts_messages_only_for_high(bundle, api_record):
    rows = [
        {**api_record, "order_id": "near", "customer_state": "SP"},
        {**api_record, "order_id": "far", "customer_state": "RR"},
        {**api_record, "order_id": "mid"},
    ]
    result = score_csv(bundle, _csv(rows))
    probabilities = [p["late_probability"] for p in result.predictions]
    assert probabilities == sorted(probabilities, reverse=True)
    for prediction in result.predictions:
        has_draft = bool(prediction["customer_message"])
        assert has_draft == (prediction["risk_level"] == "high")
    counts = sum(result.summary[level] for level in LEVELS)
    assert counts == result.summary["scored"] == 3
    frame = result.to_frame()
    assert list(frame.columns[:4]) == ["row", "order_id", "customer_state", "late_probability"]
    assert set(result.action_list()["risk_level"]) <= {"high", "elevated"}


def test_extra_columns_are_ignored_with_one_warning(bundle, api_record):
    result = score_csv(bundle, _csv([{**api_record, "customer_name": "Ana"}]))
    assert result.summary["scored"] == 1
    assert result.warnings[0]["row"] is None and "customer_name" in result.warnings[0]["field"]


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (b"", 400, "empty"),
        (b"n_items,total_price\n", 400, "empty"),
        (b"\xff\xfe\x00bad", 400, "not_csv"),
        (b"n_items,total_price\n1,2\n", 400, "missing_columns"),
    ],
)
def test_file_level_errors(data, status, code):
    with pytest.raises(BatchError) as caught:
        read_orders_csv(data)
    assert (caught.value.status, caught.value.code) == (status, code)


def test_more_than_the_row_limit_is_rejected(api_record):
    data = _csv([api_record] * (MAX_BATCH_ROWS + 1))
    with pytest.raises(BatchError) as caught:
        read_orders_csv(data)
    assert caught.value.status == 413 and caught.value.details["rows"] == MAX_BATCH_ROWS + 1


def test_missing_columns_are_listed(api_record):
    row = {
        k: v for k, v in api_record.items() if k not in {"seller_state", "estimated_delivery_date"}
    }
    with pytest.raises(BatchError) as caught:
        read_orders_csv(_csv([row]))
    assert caught.value.details["missing"] == [
        "seller_state",
        "estimated_delivery_date or promised_days",
    ]
