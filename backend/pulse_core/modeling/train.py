"""Train, compare and persist the model.

Time-aware protocol: candidates are compared with TimeSeriesSplit cross-validation on the train
window (ordered by the time column), the decision threshold is tuned on out-of-fold predictions
only, and the test window (later dates) is scored once. Thread count is pinned so results are
identical on a laptop and on CI runners. Everything is read from the artifacts: no dataset spec.
"""

from __future__ import annotations

import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

from pulse_core.columns import DATETIME_FORMAT
from pulse_core.io import read_json, sha256_file, write_csv_stable, write_json_lf
from pulse_core.modeling.candidates import BASELINE, CANDIDATE_DESCRIPTIONS, LEAN, make_candidates
from pulse_core.modeling.figures import make_model_figures
from pulse_core.modeling.metrics import best_f1_threshold, classification_metrics, round_probs
from pulse_core.splits import assign_split, windows_of
from pulse_core.validators.report import LeakageError

THREADS = 2
LEAN_TOLERANCE = 0.02
IMPORTANCE_ROWS = 5000
IMPORTANCE_REPEATS = 5


def read_features(features_csv: str | Path, spec: dict, target: str) -> pd.DataFrame:
    """features.csv typed from feature_spec.json (``spec``) and the target name."""
    kinds = {f["name"]: f["kind"] for f in spec["features"]}
    id_col, time_col, split_col = (spec[k] for k in ("id_column", "time_column", "split_column"))
    dtype = {id_col: str, split_col: str}
    dtype.update({name: str for name, kind in kinds.items() if kind == "categorical"})
    df = pd.read_csv(features_csv, dtype=dtype, keep_default_na=True, encoding="utf-8")
    df[time_col] = pd.to_datetime(df[time_col], format=DATETIME_FORMAT)
    for name, kind in kinds.items():
        if kind == "numeric":
            df[name] = df[name].astype("float64")
    df[target] = df[target].astype("int64")
    return df


def _cv_scores(pipe, X: pd.DataFrame, y: pd.Series, n_splits: int) -> tuple[list[dict], np.ndarray]:
    oof = np.full(len(X), np.nan)
    folds = []
    for k, (tr, va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), start=1):
        model = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        proba = round_probs(model.predict_proba(X.iloc[va])[:, 1])
        oof[va] = proba
        yv = y.iloc[va].to_numpy()
        folds.append(
            {
                "fold": k,
                "train_rows": int(len(tr)),
                "val_rows": int(len(va)),
                "val_prevalence": float(yv.mean()),
                "roc_auc": float(roc_auc_score(yv, proba)) if len(set(yv)) == 2 else None,
                "pr_auc": float(average_precision_score(yv, proba)) if len(set(yv)) == 2 else None,
            }
        )
    return folds, oof


def _mean_std(values: list) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def select_model(cv_pr_auc: dict[str, float]) -> dict:
    """Highest CV PR-AUC (rounded to 4 decimals) among non-baseline models; the lean model wins
    when it is within LEAN_TOLERANCE (relative) of the best."""
    scores = {name: round(score, 4) for name, score in cv_pr_auc.items() if name != BASELINE}
    ranked = sorted(scores, key=lambda name: (-scores[name], name))
    best = ranked[0]
    selected = best
    if LEAN in scores and best != LEAN and scores[LEAN] >= (1 - LEAN_TOLERANCE) * scores[best]:
        selected = LEAN
    runner_up = next(name for name in ranked if name != selected)
    return {
        "selected": selected,
        "best_by_score": best,
        "scores_cv_pr_auc": scores,
        "selection_margin": round(scores[selected] - scores[runner_up], 4),
        "rule": "highest mean CV PR-AUC (4 decimals) among non-baseline models; the lean "
        f"logistic regression wins when within {LEAN_TOLERANCE:.0%} of the best",
    }


def train_and_evaluate(
    features_csv: str | Path,
    spec_json: str | Path,
    contract_json: str | Path,
    out_dir: str | Path,
    seed: int,
    n_splits: int = 3,
) -> dict:
    out_dir = Path(out_dir)
    spec, contract = read_json(spec_json), read_json(contract_json)
    leaked = set(contract["forbidden_features"]) & set(spec["model_features"])
    if leaked:
        raise LeakageError(f"forbidden features in model_features: {sorted(leaked)}")
    target = contract["target"]["name"]
    id_col, time_col, split_col = (spec[k] for k in ("id_column", "time_column", "split_column"))
    windows = windows_of(contract["target"]["split_policy"])
    df = read_features(features_csv, spec, target)
    if not (assign_split(df[time_col], windows) == df[split_col]).all():
        raise ValueError("split column does not match the contract split policy")

    features = spec["model_features"]
    kinds = {f["name"]: f["kind"] for f in spec["features"]}
    numeric = [f for f in features if kinds[f] == "numeric"]
    categorical = [f for f in features if kinds[f] == "categorical"]
    train, test = df[df[split_col] == "train"], df[df[split_col] == "test"]
    excluded = df[~df[split_col].isin(["train", "test"])]
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target].to_numpy()

    candidates = make_candidates(seed, numeric, categorical)
    results, fitted, test_proba = {}, {}, {}
    with threadpool_limits(limits=THREADS):
        for name, pipe in candidates.items():
            folds, oof = _cv_scores(pipe, X_train, y_train, n_splits)
            mask = ~np.isnan(oof)
            threshold, oof_f1 = best_f1_threshold(y_train.to_numpy()[mask], oof[mask])
            fitted[name] = clone(pipe).fit(X_train, y_train)
            test_proba[name] = round_probs(fitted[name].predict_proba(X_test)[:, 1])
            roc_mean, roc_std = _mean_std([f["roc_auc"] for f in folds])
            pr_mean, pr_std = _mean_std([f["pr_auc"] for f in folds])
            results[name] = {
                "description": CANDIDATE_DESCRIPTIONS[name],
                "cv": {
                    "folds": folds,
                    "roc_auc_mean": roc_mean,
                    "roc_auc_std": roc_std,
                    "pr_auc_mean": pr_mean,
                    "pr_auc_std": pr_std,
                },
                "threshold": threshold,
                "oof_f1": oof_f1,
                "test": classification_metrics(y_test, test_proba[name], threshold),
            }
        selection = select_model({n: r["cv"]["pr_auc_mean"] or 0.0 for n, r in results.items()})
        chosen = selection["selected"]
        rng = np.random.RandomState(seed)
        rows = np.sort(
            rng.choice(len(X_test), size=min(IMPORTANCE_ROWS, len(X_test)), replace=False)
        )
        importance = permutation_importance(
            fitted[chosen],
            X_test.iloc[rows],
            y_test[rows],
            scoring="average_precision",
            n_repeats=IMPORTANCE_REPEATS,
            random_state=seed,
            n_jobs=1,
        )
    ranked = sorted(
        zip(features, importance.importances_mean, importance.importances_std, strict=True),
        key=lambda t: (-t[1], t[0]),
    )
    importances = [
        {"feature": f, "importance_mean": float(m), "importance_std": float(s)}
        for f, m, s in ranked
    ]

    model_path = out_dir / "model.joblib"
    joblib.dump(fitted[chosen], model_path, compress=3)
    preds = pd.DataFrame({id_col: test[id_col].to_numpy(), "y_true": y_test})
    for name in candidates:
        preds[f"p_{name}"] = test_proba[name]
    write_csv_stable(preds, out_dir / "predictions_test.csv", float_format="%.6f")

    split_info = {
        name: {
            "dates": list(windows[name]),
            "rows": int((df[split_col] == name).sum()),
            "prevalence": float(df.loc[df[split_col] == name, target].mean())
            if (df[split_col] == name).any()
            else None,
        }
        for name in windows
    }
    figures = make_model_figures(
        y_test, test_proba, chosen, importances, results, out_dir / "figures"
    )
    metrics = {
        "selected_model": chosen,
        "selection": selection,
        "candidates": results,
        "split": split_info,
        "cv_scheme": f"TimeSeriesSplit(n_splits={n_splits}) on the train window",
        "permutation_importance": {
            "model": chosen,
            "scoring": "average_precision",
            "rows": int(len(rows)),
            "repeats": IMPORTANCE_REPEATS,
            "features": importances,
        },
        "figures": figures,
    }
    write_json_lf(out_dir / "metrics.json", metrics)
    final = fitted[chosen].named_steps["model"]
    meta = {
        "model_name": chosen,
        "estimator": type(final).__name__,
        "description": CANDIDATE_DESCRIPTIONS[chosen],
        "model_features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "threshold": results[chosen]["threshold"],
        "selection": selection,
        "seed": seed,
        "thread_limit": THREADS,
        "split": split_info,
        "excluded_splits": {
            "rows": int(len(excluded)),
            "positive_rate": float(excluded[target].mean()) if len(excluded) else None,
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "contract_version": contract["contract_version"],
        "contract_sha256": sha256_file(contract_json),
        "feature_spec_sha256": sha256_file(spec_json),
        "model_file_bytes": model_path.stat().st_size,
    }
    write_json_lf(out_dir / "model_meta.json", meta)
    return {
        "selected_model": chosen,
        "test": results[chosen]["test"],
        "cv_pr_auc": selection["scores_cv_pr_auc"],
        "model_file_bytes": meta["model_file_bytes"],
    }
