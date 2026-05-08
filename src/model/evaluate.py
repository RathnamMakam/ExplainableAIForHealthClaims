"""
Evaluation helpers shared by train.py, Phase 6 explainability, and notebooks.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Return a flat dict of evaluation metrics suitable for MLflow logging."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "auc_roc":         round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision":   round(average_precision_score(y_true, y_prob), 4),
        "f1":              round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision":       round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":          round(recall_score(y_true, y_pred, zero_division=0), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    title: str,
    out_path: Path,
    top_n: int = 20,
) -> None:
    """Bar chart of top-N feature importances, saved to out_path."""
    idx = np.argsort(importances)[::-1][:top_n]
    top_names  = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(top_n), top_values[::-1], color="#4C72B0")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance (gain)")
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.4f", label_type="edge", fontsize=7, padding=2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
