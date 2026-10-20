"""Metrics for an imbalanced late-delivery classifier, and a deterministic threshold search."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLD_GRID = np.round(np.arange(0.02, 0.98 + 1e-9, 0.005), 3)


def round_probs(p: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(p, dtype=float), 6)


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """F1-maximizing threshold on a fixed grid; ties go to the lowest threshold."""
    proba = round_probs(proba)
    best_t, best_f1 = float(THRESHOLD_GRID[0]), -1.0
    for t in THRESHOLD_GRID:
        f1 = f1_score(y_true, proba >= t, zero_division=0)
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def precision_at_top(y_true: np.ndarray, proba: np.ndarray, fraction: float = 0.10) -> float:
    """Precision among the top `fraction` of orders by risk (stable tie order)."""
    n = max(1, int(round(len(proba) * fraction)))
    order = np.argsort(-round_probs(proba), kind="mergesort")[:n]
    return float(np.asarray(y_true)[order].mean())


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = round_probs(proba)
    predicted = proba >= threshold
    prevalence = float(y_true.mean())
    p_at_10 = precision_at_top(y_true, proba)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "rows": int(len(y_true)),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, proba)) if both_classes else None,
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "flagged_share": float(predicted.mean()),
        "precision_at_top10pct": p_at_10,
        "lift_at_top10pct": float(p_at_10 / prevalence) if prevalence else None,
    }
