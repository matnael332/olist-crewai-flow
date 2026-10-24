"""Flask prediction API: validation, degraded mode and training/serving parity."""

import io
from pathlib import Path

import pandas as pd
import pytest

from pulse_api.app import create_app
from pulse_core.columns import read_clean_data

FIXTURE_LOOKUPS = Path(__file__).parent / "fixtures" / "olist_mini" / "lookups"

GOOD = {
    "n_items": 1,
    "total_price": 129.9,
    "total_freight": 21.5,
    "customer_state": "BA",
    "seller_state": "SP",
    "primary_category": None,  # filled from the contract in the fixture
    "payment_type": "boleto",
    "max_installments": 1,
    "purchase_ts": "2018-06-12 14:03:00",
    "estimated_delivery_date": "2018-07-05",
}


@pytest.fixture(scope="module")
def client(pipeline_out):
    app = create_app(pipeline_out, FIXTURE_LOOKUPS)
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(scope="module")
def good(pipeline_out):
    clean = read_clean_data(pipeline_out / "clean_data.csv")
    return {**GOOD, "primary_category": clean["primary_category"].iloc[0]}


def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["n_features"] == 25 and body["threshold"] > 0


def test_predict_valid_record(client, good):
    response = client.post("/predict", json={"records": [good, good]})
    assert response.status_code == 200
    body = response.get_json()
    assert len(body["predictions"]) == 2
    p = body["predictions"][0]
    assert 0 <= p["late_probability"] <= 1 and isinstance(p["top_10pct_risk"], bool)


@pytest.mark.parametrize(
    ("change", "field", "status"),
    [
        ({"customer_state": "XX"}, "customer_state", 422),
        ({"payment_type": "pix"}, "payment_type", 422),
        ({"n_items": 0}, "n_items", 422),
        ({"total_price": "cheap"}, "total_price", 422),
        ({"surprise": 1}, "surprise", 422),
        ({"purchase_ts": "12/06/2018"}, "purchase_ts", 422),
    ],
)
def test_predict_rejects_invalid_records(client, good, change, field, status):
    response = client.post("/predict", json={"records": [{**good, **change}]})
    assert response.status_code == status
    assert any(d["field"] == field for d in response.get_json()["details"])


def test_predict_rejects_bad_envelopes(client, good):
    assert client.post("/predict", data="not json").status_code == 400
    assert client.post("/predict", json={"records": []}).status_code == 400
    assert client.post("/predict", json={"records": [good] * 1001}).status_code == 400
    assert client.post("/predict", json=[good]).status_code == 400


def test_serving_matches_training_predictions(client, pipeline_out):
    """A real test-window order scored through the API equals the stored test prediction."""
    clean = read_clean_data(pipeline_out / "clean_data.csv").set_index("order_id")
    preds = pd.read_csv(pipeline_out / "predictions_test.csv", dtype={"order_id": str})
    model = client.get("/health").get_json()["model"]
    for order_id, expected in zip(preds["order_id"][:5], preds[f"p_{model}"][:5], strict=True):
        row = clean.loc[order_id]
        record = {
            "n_items": int(row.n_items),
            "n_distinct_products": int(row.n_distinct_products),
            "n_sellers": int(row.n_sellers),
            "total_price": float(row.total_price),
            "total_freight": float(row.total_freight),
            "payment_total": float(row.payment_total),
            "n_payments": int(row.n_payments),
            "max_installments": int(row.max_installments),
            "customer_state": row.customer_state,
            "seller_state": row.seller_state,
            "primary_category": row.primary_category,
            "payment_type": row.payment_type,
            "purchase_ts": row.purchase_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "promised_days": int(row.promised_days),
        }
        for name in (
            "total_weight_g",
            "total_volume_cm3",
            "max_length_cm",
            "max_height_cm",
            "max_width_cm",
            "distance_km",
        ):
            if pd.notna(row[name]):
                record[name] = float(row[name])
        body = client.post("/predict", json={"records": [record]}).get_json()
        assert body["predictions"][0]["late_probability"] == pytest.approx(expected, abs=1e-4)


def test_degraded_mode_without_artifacts(tmp_path):
    client = create_app(tmp_path, FIXTURE_LOOKUPS).test_client()
    assert client.get("/health").get_json()["status"] == "degraded"
    assert client.post("/predict", json={"records": [GOOD]}).status_code == 503


# Product endpoints: risk levels, batch scoring, daily replay ---------------------------------


def _csv(rows: list[dict]) -> str:
    return pd.DataFrame(rows).to_csv(index=False)


def test_health_reports_the_risk_levels(client):
    levels = client.get("/health").get_json()["risk_levels"]
    assert 0 < levels["elevated_from"] <= levels["high_from"] < 1


def test_predict_echoes_order_id_with_level_and_action(client, api_record):
    body = client.post("/predict", json={"records": [{**api_record, "order_id": "o-1"}]}).get_json()
    prediction = body["predictions"][0]
    assert prediction["order_id"] == "o-1"
    assert prediction["risk_level"] in {"high", "elevated", "normal"}
    assert prediction["recommended_action"]
    assert body["model"]["top10_cutoff"] > body["model"]["threshold"]


def test_batch_endpoint_scores_csv_with_partial_success(client, api_record):
    data = _csv(
        [{**api_record, "order_id": "a"}, {**api_record, "order_id": "b", "seller_state": "XX"}]
    )
    response = client.post("/predict/batch", data=data, content_type="text/csv")
    assert response.status_code == 200
    body = response.get_json()
    assert body["summary"]["scored"] == 1 and body["summary"]["rejected"] == 1
    assert body["errors"] == [
        {"row": 2, "field": "seller_state", "message": "'XX' is not an allowed value"}
    ]
    assert body["predictions"][0]["order_id"] == "a"


def test_batch_endpoint_accepts_a_multipart_file_and_returns_csv(client, api_record):
    data = _csv([{**api_record, "order_id": "a"}])
    response = client.post(
        "/predict/batch?format=csv",
        data={"file": (io.BytesIO(data.encode()), "orders.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200 and response.mimetype == "text/csv"
    assert response.get_data(as_text=True).startswith(
        "row,order_id,customer_state,late_probability"
    )


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        ("", 400, "no_file"),
        ("n_items,total_price\n1,2\n", 400, "missing_columns"),
    ],
)
def test_batch_endpoint_file_errors(client, data, status, code):
    response = client.post("/predict/batch", data=data, content_type="text/csv")
    assert response.status_code == status and response.get_json()["error"] == code


def test_batch_endpoint_with_no_valid_rows_is_422(client, api_record):
    data = _csv([{**api_record, "customer_state": "XX"}])
    response = client.post("/predict/batch", data=data, content_type="text/csv")
    assert response.status_code == 422 and response.get_json()["error"] == "no_valid_rows"


def test_batch_endpoint_rejects_bodies_over_2_mb(client):
    response = client.post(
        "/predict/batch", data=b"x" * (2 * 1024 * 1024 + 1), content_type="text/csv"
    )
    assert response.status_code == 413 and response.get_json()["error"] == "too_large"


def test_risk_list_replays_a_day_with_outcomes(client, busiest_test_day):
    url = f"/risk-list?date={busiest_test_day}&min_level=normal&include_outcome=true"
    body = client.get(url).get_json()
    assert body["date"] == str(busiest_test_day)
    assert len(body["predictions"]) == body["summary"]["scored"] > 0
    assert all(isinstance(p["actual_late"], bool) for p in body["predictions"])
    assert body["backtest"]["all"]["orders"] == body["summary"]["rows"]
    default = client.get(f"/risk-list?date={busiest_test_day}").get_json()
    assert {p["risk_level"] for p in default["predictions"]} <= {"high", "elevated"}
    assert "backtest" not in default


@pytest.mark.parametrize(
    ("query", "status", "code"),
    [
        ("date=2018-04-01", 404, "out_of_window"),
        ("date=not-a-date", 400, "invalid_date"),
        ("date=2018-07-16&min_level=urgent", 400, "invalid_level"),
    ],
)
def test_risk_list_errors(client, query, status, code):
    response = client.get(f"/risk-list?{query}")
    assert response.status_code == status and response.get_json()["error"] == code


def test_product_endpoints_degrade_without_a_model(tmp_path):
    degraded = create_app(tmp_path, FIXTURE_LOOKUPS).test_client()
    assert (
        degraded.post("/predict/batch", data="a,b\n1,2\n", content_type="text/csv").status_code
        == 503
    )
    assert degraded.get("/risk-list?date=2018-07-16").status_code == 503
