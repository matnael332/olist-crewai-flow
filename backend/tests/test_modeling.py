import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import InconsistentVersionWarning

from pulse_core.features.build import LeakageError
from pulse_core.io import read_json, write_json_lf
from pulse_core.modeling.metrics import best_f1_threshold, precision_at_top
from pulse_core.modeling.train import read_features, select_model, train_and_evaluate


def test_selection_prefers_lean_model_within_tolerance():
    close = select_model(
        {"baseline_prior": 0.09, "logistic_regression": 0.199, "hist_gradient_boosting": 0.2}
    )
    assert close["selected"] == "logistic_regression"
    assert close["best_by_score"] == "hist_gradient_boosting"
    far = select_model(
        {"baseline_prior": 0.09, "logistic_regression": 0.15, "hist_gradient_boosting": 0.2}
    )
    assert far["selected"] == "hist_gradient_boosting"
    assert far["selection_margin"] == pytest.approx(0.05)


def test_threshold_grid_ties_go_to_lowest():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, f1 = best_f1_threshold(y, p)
    assert f1 == 1.0 and threshold == pytest.approx(0.205)
    assert precision_at_top(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.linspace(1, 0, 10)) == 1.0


def test_metrics_and_predictions_are_consistent(pipeline_out):
    metrics = read_json(pipeline_out / "metrics.json")
    meta = read_json(pipeline_out / "model_meta.json")
    spec = read_json(pipeline_out / "feature_spec.json")
    assert set(metrics["candidates"]) == {
        "baseline_prior",
        "logistic_regression",
        "hist_gradient_boosting",
    }
    chosen = metrics["selected_model"]
    assert chosen != "baseline_prior" and meta["model_name"] == chosen
    features = read_features(pipeline_out / "features.csv", spec)
    test = features[features["split"] == "test"]
    preds = pd.read_csv(pipeline_out / "predictions_test.csv", dtype={"order_id": str})
    assert preds["order_id"].tolist() == test["order_id"].tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(pipeline_out / "model.joblib")
    proba = np.round(model.predict_proba(test[spec["model_features"]])[:, 1], 6)
    np.testing.assert_allclose(proba, preds[f"p_{chosen}"].to_numpy(), atol=1e-6)
    assert meta["model_file_bytes"] < 5_000_000


def test_training_is_deterministic(tmp_path, pipeline_out):
    for name in ("a", "b"):
        target = tmp_path / name
        target.mkdir()
        for file in ("features.csv", "feature_spec.json", "dataset_contract.json"):
            shutil.copyfile(pipeline_out / file, target / file)
        train_and_evaluate(
            target / "features.csv",
            target / "feature_spec.json",
            target / "dataset_contract.json",
            target,
        )
    for file in ("metrics.json", "predictions_test.csv"):
        assert (tmp_path / "a" / file).read_bytes() == (tmp_path / "b" / file).read_bytes(), file


def test_training_refuses_forbidden_features(tmp_path, pipeline_out):
    for file in ("features.csv", "feature_spec.json", "dataset_contract.json"):
        shutil.copyfile(pipeline_out / file, tmp_path / file)
    spec = read_json(tmp_path / "feature_spec.json")
    spec["model_features"].append("delivery_days")
    write_json_lf(tmp_path / "feature_spec.json", spec)
    with pytest.raises(LeakageError):
        train_and_evaluate(
            tmp_path / "features.csv",
            tmp_path / "feature_spec.json",
            tmp_path / "dataset_contract.json",
            tmp_path,
        )
