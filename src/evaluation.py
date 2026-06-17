"""Shared evaluation module for the Titanic survival-prediction project.

Single source of truth for turning model probabilities into the graded metrics
(Accuracy, Precision, Recall, F1, ROC-AUC) and the diagnostic plots (confusion
matrix, ROC curve, learning curves). Imported by **both** ``train.py`` (report
the final validation metrics) and ``ds_app.py`` (show metrics + plots when the
uploaded CSV carries the ``Survived`` label), so the numbers a user sees in the
app are computed by the exact same code that produced the training report.

All plot helpers return a Matplotlib :class:`~matplotlib.figure.Figure` (they
never call ``plt.show``) so the caller decides what to do with it —
``st.pyplot(fig)`` in Streamlit, ``fig.savefig(...)`` in a script.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; Streamlit/scripts never need a GUI
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def find_best_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    lo: float = 0.2,
    hi: float = 0.8,
    step: float = 0.02,
) -> tuple[float, float]:
    """Scan thresholds for the one maximising macro-F1.

    Mirrors the dynamic-threshold search in ``train.py`` so train-time and
    inference-time decision rules stay identical. Macro averaging weights the
    survivor and non-survivor classes equally despite the class imbalance.

    Args:
        y_true: Binary ground-truth labels ``(n,)``.
        proba: Predicted survival probabilities in ``[0, 1]`` ``(n,)``.
        lo: Lowest threshold to try (inclusive).
        hi: Highest threshold to try (exclusive).
        step: Grid step between candidate thresholds.

    Returns:
        A ``(best_threshold, best_macro_f1)`` tuple.
    """
    best_thresh, best_macro = 0.5, 0.0
    for thresh in np.arange(lo, hi, step):
        preds = (proba >= thresh).astype(int)
        macro = f1_score(y_true, preds, average="macro", zero_division=0)
        if macro > best_macro:
            best_macro, best_thresh = float(macro), float(thresh)
    return best_thresh, best_macro


def compute_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute the full set of graded classification metrics.

    Args:
        y_true: Binary ground-truth labels ``(n,)``.
        proba: Predicted survival probabilities in ``[0, 1]`` ``(n,)``.
        threshold: Decision threshold applied to ``proba`` for the
            label-based metrics. ROC-AUC is threshold-free.

    Returns:
        A dict with ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``macro_f1``, ``roc_auc`` and the echoed ``threshold``.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    preds = (proba >= threshold).astype(int)
    # ROC-AUC is undefined with a single class present (e.g. a tiny sample);
    # report NaN rather than letting sklearn raise.
    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "roc_auc": auc,
        "threshold": float(threshold),
    }


def plot_confusion_matrix(
    y_true: np.ndarray, preds: np.ndarray, labels: tuple[str, str] = ("Died", "Survived")
) -> Figure:
    """Render a 2x2 confusion matrix with raw counts.

    Args:
        y_true: Binary ground-truth labels ``(n,)``.
        preds: Binary predicted labels ``(n,)``.
        labels: Display names for the negative and positive classes.

    Returns:
        The Matplotlib figure.
    """
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=list(labels))
    ax.set_yticks([0, 1], labels=list(labels))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    # Annotate each cell, switching text colour for contrast on dark cells.
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, proba: np.ndarray) -> Figure:
    """Plot the ROC curve with the AUC in the legend.

    Args:
        y_true: Binary ground-truth labels ``(n,)``.
        proba: Predicted survival probabilities in ``[0, 1]`` ``(n,)``.

    Returns:
        The Matplotlib figure.
    """
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


def plot_learning_curves(history: dict[str, list[float]]) -> Figure:
    """Plot train/validation learning curves over epochs.

    Args:
        history: Per-epoch metric lists. Recognised keys: ``train_loss``,
            ``val_loss`` (left panel) and ``val_accuracy``, ``val_roc_auc``
            (right panel). Missing keys are skipped so the function tolerates
            partial histories.

    Returns:
        The Matplotlib figure (two side-by-side panels: loss and val metrics).
    """
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(10, 4))

    if history.get("train_loss"):
        epochs = range(1, len(history["train_loss"]) + 1)
        ax_loss.plot(epochs, history["train_loss"], label="Train loss")
    if history.get("val_loss"):
        epochs = range(1, len(history["val_loss"]) + 1)
        ax_loss.plot(epochs, history["val_loss"], label="Val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss curve")
    ax_loss.legend()

    if history.get("val_accuracy"):
        epochs = range(1, len(history["val_accuracy"]) + 1)
        ax_metric.plot(epochs, history["val_accuracy"], label="Val accuracy")
    if history.get("val_roc_auc"):
        epochs = range(1, len(history["val_roc_auc"]) + 1)
        ax_metric.plot(epochs, history["val_roc_auc"], label="Val ROC-AUC")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score")
    ax_metric.set_title("Validation metrics")
    ax_metric.legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Brick 6 verification: synthetic scores -> metrics + all three plots.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    # Probabilities correlated with the label so AUC is meaningfully > 0.5.
    p = np.clip(y * 0.4 + rng.random(200) * 0.6, 0, 1)

    thresh, macro = find_best_threshold(y, p)
    metrics = compute_metrics(y, p, threshold=thresh)
    print(f"[ok] best_thresh={thresh:.2f} (macro_f1={macro:.3f})")
    for k, v in metrics.items():
        print(f"       {k:10s} {v:.3f}")

    preds = (p >= thresh).astype(int)
    plot_confusion_matrix(y, preds).savefig("eval_confusion.png")
    plot_roc_curve(y, p).savefig("eval_roc.png")
    hist = {
        "train_loss": [0.7, 0.6, 0.55, 0.5],
        "val_loss": [0.72, 0.63, 0.6, 0.61],
        "val_accuracy": [0.6, 0.7, 0.74, 0.75],
        "val_roc_auc": [0.65, 0.74, 0.78, 0.79],
    }
    plot_learning_curves(hist).savefig("eval_learning.png")
    print("[ok] wrote eval_confusion.png, eval_roc.png, eval_learning.png")
