"""Standalone hyperparameter search for the Titanic survival model.

Kept deliberately separate from ``train.py`` (an MLOps split): tuning is an
occasional, expensive experiment, while training is the fast, reproducible step
run on every change. This script runs an Optuna study and writes the winning
hyperparameters to ``best_params.json`` — the single source of truth for "which
brain is in the model". That file is committed to Git, so the exact
configuration behind the current weights is always known, and ``train.py`` reads
it on every run.

The objective reuses :func:`train.train` via its **standalone** ``data_path``
flow (load → split → fit preprocessing on the train split), so the search has no
dependency on ``prepare_data.py`` / ``splits.npz`` (those exist only to speed up
Colab GPU runs). Each trial trains into a throwaway temp directory; only the
final ``best_params.json`` is persisted here — the actual best weights are
produced afterwards by ``python train.py``.

Usage::

    python tuning.py                 # 20 trials, writes best_params.json
    python tuning.py --n-trials 50   # longer search
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

from train import train

# Hyperparameters this study searches. Everything else in ``TrainConfig`` keeps
# its default. Mirrors the search space in notebooks/colab_train.ipynb (cell 5).
SEARCHED_KEYS = ("lr", "dropout", "batch_size", "hidden_dim", "n_blocks")


def _objective(
    trial: optuna.Trial, data_path: str, seed: int, epochs: int, patience: int
) -> float:
    """Train one trial and return its validation ROC-AUC (the search target).

    Args:
        trial: The Optuna trial proposing a hyperparameter configuration.
        data_path: Raw training CSV passed to the standalone ``train()`` flow.
        seed: Global seed, kept fixed across trials so the only variation is the
            sampled hyperparameters (and the train/val split stays identical).
        epochs: Max epochs per trial.
        patience: Early-stopping patience per trial.

    Returns:
        The best validation ROC-AUC achieved by this trial.
    """
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
        "n_blocks": trial.suggest_int("n_blocks", 1, 3),
    }
    # Throwaway artifacts: trials only care about the returned metric, not the
    # weights. A temp dir avoids littering the repo with trial_N folders.
    with tempfile.TemporaryDirectory() as tmp:
        res = train(
            data_path=data_path,
            out_dir=tmp,
            epochs=epochs,
            patience=patience,
            seed=seed,
            **params,
        )
    return float(res["best_metrics"]["roc_auc"])


def run_study(
    n_trials: int = 20,
    data_path: str = "data/train.csv",
    seed: int = 42,
    out: str = "best_params.json",
    epochs: int = 100,
    patience: int = 10,
) -> dict[str, object]:
    """Run the Optuna study and persist the best hyperparameters.

    Args:
        n_trials: Number of trials to evaluate.
        data_path: Raw training CSV (standalone flow; no prepare_data.py needed).
        seed: Seed for the TPE sampler and each trial, for reproducibility.
        out: Destination JSON path for the best params + provenance.
        epochs: Max epochs per trial.
        patience: Early-stopping patience per trial.

    Returns:
        The payload written to ``out``.
    """
    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=seed)
    )
    study.optimize(
        lambda t: _objective(t, data_path, seed, epochs, patience),
        n_trials=n_trials,
    )

    payload = {
        "best_params": study.best_trial.params,
        "best_value_roc_auc": float(study.best_trial.value),
        "metric": "val_roc_auc",
        "n_trials": n_trials,
        "seed": seed,
        "data_path": data_path,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path = Path(out)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n[best] val ROC-AUC = {payload['best_value_roc_auc']:.4f}")
    for key, value in study.best_trial.params.items():
        print(f"       {key:12s} {value}")
    print(f"[saved] {out_path}  (commit this — it defines the model's config)")
    print("[next] run `python train.py` to train the final model with these params.")
    return payload


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the tuning run."""
    p = argparse.ArgumentParser(description="Optuna hyperparameter search.")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--data-path", default="data/train.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="best_params.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_study(
        n_trials=args.n_trials,
        data_path=args.data_path,
        seed=args.seed,
        out=args.out,
        epochs=args.epochs,
        patience=args.patience,
    )
