"""Model bundle shared by the Flask API and the Streamlit Predict page (no Flask import)."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from api.validation import RecordValidator, to_features
from olist_core.cleaning.geo import load_centroids
from olist_core.io import read_json


class ModelBundle:
    """Model + contract + validator. Never raises: on any problem `error` explains why."""

    def __init__(self, artifacts_dir: Path, lookups_dir: Path) -> None:
        self.error: str | None = None
        self.model = None
        try:
            self.meta = read_json(artifacts_dir / "model_meta.json")
            self.contract = read_json(artifacts_dir / "dataset_contract.json")
            trained_with = self.meta["versions"]["scikit_learn"]
            if trained_with != sklearn.__version__:
                raise RuntimeError(
                    f"model trained with scikit-learn {trained_with}, runtime has {sklearn.__version__}"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                self.model = joblib.load(artifacts_dir / "model.joblib")
            centroids_path = Path(lookups_dir) / "zip_prefix_centroids.csv"
            centroids = load_centroids(lookups_dir) if centroids_path.exists() else None
            self.validator = RecordValidator(self.contract, centroids)
            preds = pd.read_csv(artifacts_dir / "predictions_test.csv")
            self.top10_cutoff = float(np.quantile(preds[f"p_{self.meta['model_name']}"], 0.9))
            self.test_prevalence = float(preds["y_true"].mean())
        except Exception as exc:  # noqa: BLE001 - degrade instead of crashing
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def ok(self) -> bool:
        return self.error is None

    def predict(self, records: list) -> tuple[list[dict], list[dict], list[dict]]:
        """Returns (predictions, errors, warnings)."""
        result = self.validator.validate(records)
        if result.errors:
            return [], result.errors, result.warnings
        proba = self.model.predict_proba(to_features(result.rows))[:, 1]
        threshold = self.meta["threshold"]
        predictions = [
            {
                "index": i,
                "late_probability": round(float(p), 4),
                "predicted_late": bool(p >= threshold),
                "top_10pct_risk": bool(p >= self.top10_cutoff),
            }
            for i, p in enumerate(proba)
        ]
        return predictions, [], result.warnings
