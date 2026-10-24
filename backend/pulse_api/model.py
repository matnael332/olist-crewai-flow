"""Model bundle shared by the Flask API, the ops copilot and the dashboard (no Flask import)."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from pulse_api.risk import RiskPolicy
from pulse_api.validation import RecordValidator
from pulse_core.io import read_json

SCORE_CHUNK = 1000


class ModelBundle:
    """Model + contract + validator + risk policy. Never raises: on any problem `error` says why."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.error: str | None = None
        self.model = None
        try:
            self.meta = read_json(artifacts_dir / "model_meta.json")
            self.contract = read_json(artifacts_dir / "dataset_contract.json")
            self.feature_spec = read_json(artifacts_dir / "feature_spec.json")
            trained_with = self.meta["versions"]["scikit_learn"]
            if trained_with != sklearn.__version__:
                raise RuntimeError(
                    f"model trained with scikit-learn {trained_with}, runtime has {sklearn.__version__}"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                self.model = joblib.load(artifacts_dir / "model.joblib")
            self.validator = RecordValidator(self.contract, self.feature_spec)
            self.id_field = self.validator.id_field
            preds = pd.read_csv(artifacts_dir / "predictions_test.csv")
            self.top10_cutoff = float(np.quantile(preds[f"p_{self.meta['model_name']}"], 0.9))
            self.test_prevalence = float(preds["y_true"].mean())
            self.policy = RiskPolicy(float(self.meta["threshold"]), self.top10_cutoff)
        except Exception as exc:  # noqa: BLE001 - degrade instead of crashing
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def ok(self) -> bool:
        return self.error is None

    def score_rows(self, rows: list[dict]) -> np.ndarray:
        """Probabilities for rows already validated by RecordValidator, in chunks."""
        parts = [
            self.model.predict_proba(self.validator.to_features(rows[i : i + SCORE_CHUNK]))[:, 1]
            for i in range(0, len(rows), SCORE_CHUNK)
        ]
        return np.concatenate(parts) if parts else np.array([], dtype=float)

    def describe(self, index: int, probability: float, record: dict) -> dict:
        """One prediction in the API's shape, with the risk level and the recommended action."""
        level = self.policy.level(probability)
        out: dict = {"index": index}
        if isinstance(record, dict) and self.id_field and record.get(self.id_field) is not None:
            out["id"] = str(record[self.id_field])
        out.update(
            probability=round(float(probability), 4),
            predicted_positive=bool(probability >= self.meta["threshold"]),
            top_10pct_risk=bool(probability >= self.top10_cutoff),
            risk_level=level,
            recommended_action=self.policy.action(level),
        )
        return out

    def predict(self, records: list) -> tuple[list[dict], list[dict], list[dict]]:
        """All-or-nothing scoring (POST /predict): (predictions, errors, warnings)."""
        result = self.validator.validate(records)
        if result.errors:
            return [], result.errors, result.warnings
        proba = self.score_rows(result.rows)
        predictions = [self.describe(i, float(p), records[i]) for i, p in enumerate(proba)]
        return predictions, [], result.warnings
