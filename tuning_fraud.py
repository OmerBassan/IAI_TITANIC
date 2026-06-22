"""Optuna hyperparameter search for the credit-card fraud model.

Optimises **validation PR-AUC** (the metric that matters under a 578:1 class
imbalance) over the architecture + optimisation knobs, including the
``pos_weight`` scale — the first run showed the raw 599:1 weighting was the main
thing holding PR-AUC back, so it is searched explicitly here.

The data is loaded, split and preprocessed **once** at module load (fit on the
train split only); each trial just re-trains the model, so a 30-trial search is
cheap on a GPU. The best parameters are written to ``fraud_best_params.json``,
which ``train_fraud.py`` picks up automatically on its next run.

Run::

    python tuning_fraud.py --trials 30
    python tuning_fraud.py --trials 50 --epochs 25 --device cuda
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.fraud_evaluation import compute_metrics, find_best_threshold
from src.fraud_preprocessing import (
    build_preprocessor,
    compute_pos_weight,
    load_raw,
    make_splits,
    split_xy,
)
from src.model import TabularResNet


@dataclass
class PreparedData:
    """Preprocessed splits shared across all trials (no per-trial recompute)."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    base_pos_weight: float


def prepare_data(data_path: str, val_size: float, test_size: float) -> PreparedData:
    """Load, split (train/val/test) and preprocess the dataset once.

    The test split is created for parity with training but unused here — tuning
    must never see it. The preprocessor is fit on the train split only.
    """
    raw = load_raw(data_path)
    train_df, val_df, _test_df = make_splits(raw, val_size=val_size, test_size=test_size)
    X_train_df, y_train = split_xy(train_df)
    X_val_df, y_val = split_xy(val_df)

    pre = build_preprocessor()
    X_train = pre.fit_transform(X_train_df).astype(np.float32)
    X_val = pre.transform(X_val_df).astype(np.float32)
    return PreparedData(
        X_train=X_train,
        y_train=y_train.to_numpy().astype(np.float32),
        X_val=X_val,
        y_val=y_val.to_numpy().astype(np.float32),
        base_pos_weight=compute_pos_weight(train_df["Class"]),
    )


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X), torch.as_tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


@torch.no_grad()
def _val_pr_auc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    logits_all, y_all = [], []
    for xb, yb in loader:
        logits_all.append(model(xb.to(device)).cpu())
        y_all.append(yb)
    logits = torch.cat(logits_all).numpy()
    y_true = torch.cat(y_all).numpy()
    proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return compute_metrics(y_true, proba)["pr_auc"]


def make_objective(data: PreparedData, epochs: int, patience: int, device: torch.device):
    """Build an Optuna objective bound to the prepared data."""

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        n_blocks = trial.suggest_int("n_blocks", 1, 4)
        batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        # Search pos_weight directly across the gentle..aggressive range. The raw
        # ratio is ~599; sqrt is ~24.5; this brackets both extremes generously.
        pos_weight = trial.suggest_float("pos_weight", 5.0, 200.0, log=True)

        torch.manual_seed(42)
        model = TabularResNet(
            in_features=data.X_train.shape[1],
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = _loader(data.X_train, data.y_train, batch_size, shuffle=True)
        val_loader = _loader(data.X_val, data.y_val, batch_size, shuffle=False)

        best_pr_auc = -1.0
        no_improve = 0
        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()

            pr_auc = _val_pr_auc(model, val_loader, device)
            trial.report(pr_auc, epoch)
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_pr_auc

    return objective


def run(trials: int, epochs: int, patience: int, data_path: str,
        val_size: float, test_size: float, device: str, out_path: str) -> dict:
    """Run the study and persist the best parameters."""
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                        else device if device != "auto" else "cpu")
    print(f"[setup] device={dev} | trials={trials} | epochs/trial={epochs}")

    data = prepare_data(data_path, val_size, test_size)
    print(f"[data] train={len(data.y_train)} val={len(data.y_val)} "
          f"| features={data.X_train.shape[1]} | base pos_weight={data.base_pos_weight:.1f}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(make_objective(data, epochs, patience, dev), n_trials=trials)

    print(f"\n[best] val PR-AUC = {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"       {k}: {v}")

    payload = {
        "best_params": study.best_params,
        "best_val_pr_auc": study.best_value,
        "n_trials": trials,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}  (train_fraud.py will pick these up automatically)")
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna search for the fraud model (max val PR-AUC).")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=25, help="max epochs per trial")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--data-path", default="data/creditcard.csv")
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--test-size", type=float, default=0.10)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--out-path", default="fraud_best_params.json")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    run(a.trials, a.epochs, a.patience, a.data_path, a.val_size, a.test_size,
        a.device, a.out_path)
