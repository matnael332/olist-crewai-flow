"""Scoring at the snapshot: who is at risk now, and why.

The model is trained on rows whose outcome is known, so the newest months never reach it. A spec
that defines ``score_rows`` hands the analyst stage those newest rows (the decision-stage columns
of every entity still live at the snapshot); this stage scores each entity's latest row with the
selected model and writes:

- ``scores_current.csv``: one row per entity, ranked by probability, with the risk level of the
  shared policy (``pulse_core.risk``);
- ``reasons.json``: for each entity, the features that raise its score the most. A feature's
  contribution is the drop in probability when it alone is set to its typical value (the train
  median, or the most frequent category), tried for the most important features of the
  permutation importance. Deterministic, and model-agnostic.

History matters for trend and streak features, so the entity's earlier rows from the clean table
are put in front of its score rows before the features are derived.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pulse_core.columns import read_clean_data
from pulse_core.io import read_json, write_csv_stable, write_json_lf
from pulse_core.modeling.metrics import round_probs
from pulse_core.modeling.train import read_features
from pulse_core.risk import RiskPolicy
from pulse_core.spec import DatasetSpec

SCORE_ROWS = "score_rows.csv"
SCORES = "scores_current.csv"
REASONS = "reasons.json"
REASON_FEATURES = 8  # features tried per entity (the most important ones)
REASONS_PER_ENTITY = 3


def risk_policy(out_dir: Path, meta: dict) -> RiskPolicy:
    """The policy of the run: the tuned threshold and the top-10% cutoff of the test window."""
    preds = pd.read_csv(out_dir / "predictions_test.csv")
    top10 = float(np.quantile(preds[f"p_{meta['model_name']}"], 0.9))
    return RiskPolicy(float(meta["threshold"]), top10)


def _typical(train: pd.DataFrame, kinds: dict[str, str]) -> dict[str, object]:
    typical: dict[str, object] = {}
    for name, kind in kinds.items():
        values = train[name].dropna()
        if values.empty:
            continue
        if kind == "numeric":
            typical[name] = float(values.astype("float64").median())
        else:
            counts = values.astype(str).value_counts()
            typical[name] = sorted(counts[counts == counts.max()].index)[0]
    return typical


def _value(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def score_current(spec: DatasetSpec, out_dir: str | Path) -> dict | None:
    """Score the spec's score rows (None when the analyst stage wrote none)."""
    out_dir = Path(out_dir)
    if not (out_dir / SCORE_ROWS).exists():
        return None
    entity, time = spec.entity_column, spec.time_column
    if not entity:
        raise ValueError("a spec with score_rows must name its entity_column")
    meta = read_json(out_dir / "model_meta.json")
    feature_spec = read_json(out_dir / "feature_spec.json")
    metrics = read_json(out_dir / "metrics.json")
    model = joblib.load(out_dir / "model.joblib")
    features = list(meta["model_features"])
    kinds = {f["name"]: f["kind"] for f in feature_spec["features"]}

    rows = read_clean_data(out_dir / SCORE_ROWS, spec.dtypes)
    clean = read_clean_data(out_dir / "clean_data.csv", spec.dtypes)
    history = clean[clean[entity].isin(rows[entity])][list(rows.columns)]
    frame = pd.concat([history, rows], ignore_index=True)
    derived = spec.derive_features(frame)
    latest = rows.sort_values([entity, time], kind="mergesort").groupby(entity).tail(1)
    picked = latest.index + len(history)  # positions of the score rows inside ``frame``
    X = derived.loc[picked, features].reset_index(drop=True)
    for name in features:  # the dtypes the model was trained on (read_features)
        X[name] = X[name].astype("float64" if kinds[name] == "numeric" else object)
    proba = round_probs(model.predict_proba(X)[:, 1])

    policy = risk_policy(out_dir, meta)
    scores = pd.DataFrame(
        {
            entity: latest[entity].to_numpy(),
            spec.id_column: latest[spec.id_column].to_numpy(),
            time: latest[time].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
            "probability": proba,
        }
    )
    scores["risk_level"] = [policy.level(p) for p in proba]
    scores = scores.sort_values(["probability", entity], ascending=[False, True], kind="mergesort")
    scores["rank"] = np.arange(1, len(scores) + 1)
    write_csv_stable(scores.reset_index(drop=True), out_dir / SCORES, float_format="%.6f")

    target = read_json(out_dir / "dataset_contract.json")["target"]["name"]
    train = read_features(out_dir / "features.csv", feature_spec, target)
    train = train[train[feature_spec["split_column"]] == "train"]
    typical = _typical(train, {f: kinds[f] for f in features})
    ranked = metrics["permutation_importance"]["features"]
    tried = [r["feature"] for r in ranked if r["importance_mean"] > 0 and r["feature"] in typical][
        :REASON_FEATURES
    ]
    contributions = {}
    for name in tried:
        changed = X.copy()
        changed[name] = typical[name]
        contributions[name] = proba - round_probs(model.predict_proba(changed)[:, 1])

    order = {e: i for i, e in enumerate(latest[entity])}
    entities = []
    for _, row in scores.iterrows():
        i = order[row[entity]]
        found = sorted(
            ((name, round(float(contributions[name][i]), 4)) for name in tried),
            key=lambda t: (-t[1], t[0]),
        )
        reasons = [
            {
                "feature": name,
                "code": spec.reason_codes.get(name, name),
                "value": _value(X.at[i, name]),
                "typical": typical[name],
                "contribution": delta,
            }
            for name, delta in found[:REASONS_PER_ENTITY]
            if delta > 0
        ]
        entities.append({entity: row[entity], "rank": int(row["rank"]), "reasons": reasons})
    write_json_lf(
        out_dir / REASONS,
        {
            "entity_column": entity,
            "model": meta["model_name"],
            "method": (
                "contribution = probability - probability with the feature set to its typical "
                "value (train median or most frequent category)"
            ),
            "features_tried": tried,
            "typical": {name: typical[name] for name in tried},
            "entities": entities,
        },
    )
    levels = scores["risk_level"].value_counts()
    return {
        "entities": int(len(scores)),
        "score_time": str(scores[time].max()) if len(scores) else None,
        **{f"risk_{level}": int(levels.get(level, 0)) for level in ("high", "elevated", "normal")},
    }
