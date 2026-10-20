"""Flask prediction API: validation, degraded mode and training/serving parity."""

from pathlib import Path

import pandas as pd
import pytest

from olist_api.app import create_app
from olist_core.columns import read_clean_data

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
