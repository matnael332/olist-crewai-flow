"""The compared model variations, each a full sklearn Pipeline (preprocessing included), so the
saved model.joblib can score raw feature rows directly (Flask API, Streamlit)."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

BASELINE = "baseline_prior"
LEAN = "logistic_regression"
CANDIDATE_DESCRIPTIONS = {
    BASELINE: "DummyClassifier(prior): predicts the training late rate for every order",
    LEAN: "LogisticRegression on scaled numerics + one-hot categories (threshold tuned for the imbalance)",
    "hist_gradient_boosting": "HistGradientBoostingClassifier with native categorical splits",
}


def _linear_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical),
        ]
    )


def make_candidates(seed: int, numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    tree_prep = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    dtype=np.float64,
                ),
                categorical,
            ),
        ]
    )
    categorical_idx = list(range(len(numeric), len(numeric) + len(categorical)))
    return {
        BASELINE: Pipeline(
            [
                ("prep", _linear_preprocessor(numeric, categorical)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        LEAN: Pipeline(
            [
                ("prep", _linear_preprocessor(numeric, categorical)),
                (
                    "model",
                    LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", tree_prep),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=200,
                        max_leaf_nodes=31,
                        min_samples_leaf=40,
                        l2_regularization=1.0,
                        early_stopping=False,
                        categorical_features=categorical_idx,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
