"""Evaluation figures (Figure API, Agg, deterministic metadata)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

DPI = 110
COLORS = {
    "baseline_prior": "#94a3b8",
    "logistic_regression": "#2563eb",
    "hist_gradient_boosting": "#dc2626",
}


def _save(fig: Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    return path.name


def make_model_figures(
    y_test: np.ndarray,
    test_proba: dict[str, np.ndarray],
    selected: str,
    importances: list[dict],
    results: dict,
    out: str | Path,
) -> list[str]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    two_classes = len(np.unique(y_test)) == 2

    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    for name, proba in test_proba.items():
        if two_classes:
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = results[name]["test"]["roc_auc"]
            ax.plot(fpr, tpr, color=COLORS.get(name), label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle=":", color="#cbd5e1")
    ax.set_title("ROC curves, test window")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(fontsize=8, loc="lower right")
    names.append(_save(fig, out / "roc_curves.png"))

    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    for name, proba in test_proba.items():
        if two_classes:
            precision, recall, _ = precision_recall_curve(y_test, proba)
            ap = results[name]["test"]["pr_auc"]
            ax.plot(recall, precision, color=COLORS.get(name), label=f"{name} (AP {ap:.3f})")
    ax.axhline(float(np.mean(y_test)), linestyle=":", color="#cbd5e1", label="prevalence")
    ax.set_title("Precision-recall curves, test window")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(fontsize=8, loc="upper right")
    names.append(_save(fig, out / "pr_curves.png"))

    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    if two_classes:
        frac_pos, mean_pred = calibration_curve(
            y_test, test_proba[selected], n_bins=10, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", color=COLORS.get(selected), label=selected)
    ax.plot([0, 1], [0, 1], linestyle=":", color="#cbd5e1", label="perfect calibration")
    ax.set_title(f"Calibration of {selected} (test window)")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed late rate")
    ax.legend(fontsize=8)
    names.append(_save(fig, out / "calibration_selected.png"))

    top = importances[:15][::-1]
    fig = Figure(figsize=(7, 5))
    ax = fig.subplots()
    ax.barh(
        [t["feature"] for t in top],
        [t["importance_mean"] for t in top],
        xerr=[t["importance_std"] for t in top],
        color=COLORS.get(selected, "#2563eb"),
    )
    ax.set_title(f"Permutation importance ({selected}, drop in PR-AUC)")
    ax.set_xlabel("mean decrease in average precision")
    names.append(_save(fig, out / "feature_importance_selected.png"))
    return names
