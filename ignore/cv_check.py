"""Cross-validation sanity check for the best Optuna trial (#15).

Confirms the single-split tuning numbers hold up across folds. Reuses the exact
training machinery from ``train.py`` (same model, loss, optimiser, early stopping,
dynamic-threshold metrics) so the per-fold numbers are comparable to a real run.

Run::

    python cv_check.py --folds 5 --epochs 200

Each fold: fit the preprocessor on that fold's train split only (no leakage),
train with early stopping on the fold's val split, record the best-epoch metrics.
Reports mean +/- std across folds.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from src.evaluation import compute_metrics, find_best_threshold
from src.model import TabularResNet
from src.preprocessing import build_preprocessor, load_raw, split_xy
from train import TrainConfig, _evaluate, _to_loader, resolve_device, set_seed


def _train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Train a single fold with early stopping; return best-epoch val metrics."""
    train_loader = _to_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = _to_loader(X_val, y_val, cfg.batch_size, shuffle=False)

    model = TabularResNet(
        in_features=X_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
    ).to(device)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    no_improve = 0
    for _ in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        val = _evaluate(model, val_loader, criterion, device)
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_metrics = val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break
    return best_metrics


def main() -> None:
    """Run stratified k-fold CV with the configured hyper-parameters."""
    p = argparse.ArgumentParser(description="K-fold CV check for the best trial.")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--patience", type=int, default=TrainConfig.patience)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--data-path", default=TrainConfig.data_path)
    p.add_argument("--device", default=TrainConfig.device, choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs, patience=args.patience, seed=args.seed, device=args.device
    )
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    raw = load_raw(args.data_path)
    X_df, y_ser = split_xy(raw)
    if y_ser is None:
        raise ValueError(f"{args.data_path} has no 'Survived' column - cannot CV.")
    y = y_ser.to_numpy()

    print(
        f"[cv] {args.folds}-fold | hidden_dim={cfg.hidden_dim} n_blocks={cfg.n_blocks} "
        f"batch={cfg.batch_size} lr={cfg.lr:.2e} dropout={cfg.dropout:.3f} "
        f"wd={cfg.weight_decay} | device={device}"
    )

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=cfg.seed)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows: list[dict[str, float]] = []
    for i, (tr, va) in enumerate(skf.split(X_df, y), start=1):
        pre = build_preprocessor()
        X_train = pre.fit_transform(X_df.iloc[tr], y[tr])
        X_val = pre.transform(X_df.iloc[va])
        m = _train_one_fold(X_train, y[tr], X_val, y[va], cfg, device)
        rows.append(m)
        print(
            f"  fold {i}: acc {m['accuracy']:.3f} | f1 {m['f1']:.3f} | "
            f"auc {m['roc_auc']:.3f} | thr {m['best_thresh']:.2f}"
        )

    print("\n[cv] mean +/- std across folds:")
    for k in keys:
        vals = np.array([r[k] for r in rows])
        print(f"  {k:10s} {vals.mean():.3f} +/- {vals.std():.3f}")


if __name__ == "__main__":
    main()
