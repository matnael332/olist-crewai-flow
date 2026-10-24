"""The prediction API on the toy model: health, all-or-nothing scoring and batch CSV scoring."""

import pytest

from pulse_api.app import create_app
from pulse_api.batch import BatchError, score_csv
from pulse_api.risk import RiskPolicy


@pytest.fixture(scope="module")
def client(pipeline_out):
    app = create_app(pipeline_out)
    app.testing = True
    return app.test_client()


def test_risk_policy_levels():
    policy = RiskPolicy(threshold=0.055, top10_cutoff=0.17)
    assert [policy.level(p) for p in (0.01, 0.06, 0.2)] == ["normal", "elevated", "high"]
    assert policy.describe() == {"elevated_from": 0.055, "high_from": 0.17}
    assert RiskPolicy(0.5, 0.2).high_from == 0.5


def test_health_describes_the_model(client, spec):
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["features"] == spec.model_features
    assert body["id_field"] == spec.id_column


def test_predict_scores_valid_records_and_rejects_bad_ones(client, api_record):
    ok = client.post("/predict", json={"records": [api_record]})
    assert ok.status_code == 200, ok.get_json()
    prediction = ok.get_json()["predictions"][0]
    assert prediction["id"] == "r9999" and 0 <= prediction["probability"] <= 1
    assert prediction["risk_level"] in ("high", "elevated", "normal")

    bad = client.post("/predict", json={"records": [{**api_record, "x_cat": "zzz"}]})
    assert bad.status_code == 422
    assert bad.get_json()["details"][0]["field"] == "x_cat"
    assert client.post("/predict", json={"nope": 1}).status_code == 400
    assert client.post("/predict", json={"records": []}).status_code == 400


def test_batch_csv_scores_valid_rows_and_reports_the_rest(client, bundle):
    csv = "row_id,x_num,x_cat,x_flag,x_num_sq,extra\nr1,1.0,a,0,1.0,z\nr2,abc,b,1,0.0,z\n"
    result = score_csv(bundle, csv.encode())
    assert result.summary["scored"] == 1 and result.summary["rejected"] == 1
    assert result.errors[0]["row"] == 2 and result.errors[0]["field"] == "x_num"
    assert result.warnings[0]["field"] == "extra"
    with pytest.raises(BatchError) as missing:
        score_csv(bundle, b"row_id,x_cat\nr1,a\n")
    assert missing.value.code == "missing_columns"

    response = client.post("/predict/batch?format=csv", data=csv, content_type="text/csv")
    assert response.status_code == 200
    assert (
        response.data.decode().splitlines()[0] == "row,id,probability,risk_level,recommended_action"
    )
    assert client.post("/predict/batch", data=b"", content_type="text/csv").status_code == 400
