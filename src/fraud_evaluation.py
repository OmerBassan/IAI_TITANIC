"""Evaluation module for the credit-card fraud detection model.

Fraud detection lives or dies by the **minority class**, so the headline metric
here is **PR-AUC** (average precision), not accuracy or ROC-AUC. At a 0.17 %
fraud rate a model that predicts "legit" for everything scores 99.83 % accuracy
and ~0.5 ROC-AUC can look healthy while precision/recall on fraud collapse —
the precision-recall curve exposes exactly that, ROC does not.

The decision threshold is chosen by scanning the full [0, 1] range for the best
F1 on the positive (fraud) class — unlike the Titanic evaluator, which scans a
narrow band and uses macro-F1, because here the classes are wildly imbalanced
and we care about the fraud class specifically.

Plot helpers return a Matplotlib Figure (never call plt.show) so the caller
decides whether to st.pyplot(...) or fig.savefig(...).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def find_best_threshold(
    y_true: np.ndarray, proba: np.ndarray
) -> tuple[float, float]:
    """Find the threshold maximising F1 on the fraud (positive) class.

    Uses the thresholds the PR curve already evaluates internally — exact and
    far cheaper than a fixed grid scan. Returns 0.5 as a safe fallback when the
    positive class is absent (e.g. a degenerate batch).

    Args:
        y_true: Binary ground-truth labels (n,), 1 = fraud.
        proba: Predicted fraud probabilities in [0, 1] (n,).

    Returns:
        (best_threshold, best_f1) for the fraud class.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have length len(thresholds)+1; align by dropping the last.
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
    if len(f1) == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def compute_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute the full set of fraud-detection metrics.

    PR-AUC and ROC-AUC are threshold-free; precision/recall/f1/accuracy use the
    supplied threshold (all reported for the fraud / positive class).

    Args:
        y_true: Binary ground-truth labels (n,), 1 = fraud.
        proba: Predicted fraud probabilities in [0, 1] (n,).
        threshold: Decision threshold for the label-based metrics.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, pr_auc, threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    preds = (proba >= threshold).astype(int)
    try:
        roc = float(roc_auc_score(y_true, proba))
        pr = float(average_precision_score(y_true, proba))
    except ValueError:  # single class present
        roc = pr = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "threshold": float(threshold),
    }


def plot_pr_curve(y_true: np.ndarray, proba: np.ndarray) -> Figure:
    """Plot the precision-recall curve with average precision in the legend.

    This is the primary diagnostic for an imbalanced problem: it shows the
    precision available at every recall level, and the baseline (the fraud
    prevalence) makes the lift over chance explicit.
    """
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(recall, precision, color="#d62728", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.axhline(baseline, color="grey", lw=1, linestyle="--",
               label=f"Baseline ({baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, proba: np.ndarray) -> Figure:
    """Plot the ROC curve with AUC in the legend (secondary diagnostic)."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = float("nan")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, preds: np.ndarray, labels: tuple[str, str] = ("Legit", "Fraud")
) -> Figure:
    """Render a 2x2 confusion matrix with raw counts."""
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Reds")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=list(labels))
    ax.set_yticks([0, 1], labels=list(labels))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def plot_learning_curves(history: dict[str, list[float]]) -> Figure:
    """Plot train/val loss and validation PR-AUC / ROC-AUC over epochs."""
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(10, 4))

    if history.get("train_loss"):
        ax_loss.plot(range(1, len(history["train_loss"]) + 1),
                     history["train_loss"], label="Train loss")
    if history.get("val_loss"):
        ax_loss.plot(range(1, len(history["val_loss"]) + 1),
                     history["val_loss"], label="Val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss curve")
    ax_loss.legend()

    if history.get("val_pr_auc"):
        ax_metric.plot(range(1, len(history["val_pr_auc"]) + 1),
                       history["val_pr_auc"], label="Val PR-AUC")
    if history.get("val_roc_auc"):
        ax_metric.plot(range(1, len(history["val_roc_auc"]) + 1),
                       history["val_roc_auc"], label="Val ROC-AUC")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score")
    ax_metric.set_title("Validation metrics")
    ax_metric.legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Smoke test: synthetic imbalanced scores -> metrics + PR/ROC plots.
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)  # 2% positive
    p = np.clip(y * 0.5 + rng.random(5000) * 0.5, 0, 1)

    thresh, f1 = find_best_threshold(y, p)
    metrics = compute_metrics(y, p, threshold=thresh)
    print(f"[ok] best_thresh={thresh:.3f} (fraud F1={f1:.3f})")
    for k, v in metrics.items():
        print(f"       {k:10s} {v:.4f}")
    plot_pr_curve(y, p).savefig("eval_pr.png")
    plot_roc_curve(y, p).savefig("eval_roc.png")
    print("[ok] wrote eval_pr.png, eval_roc.png")
