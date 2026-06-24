"""Training script for the credit-card fraud detection model.

Pipeline: load creditcard.csv (drop duplicates) -> stratified train/val/test
split (preserving the 0.17 % fraud rate) -> fit the preprocessing pipeline
**on the training split only** (no leakage) -> train the ``TabularResNet`` with
BCEWithLogitsLoss(pos_weight) -> save the best checkpoint, selected on the
validation **PR-AUC** -> report final metrics on the held-out test split.

Why PR-AUC for model selection (not val-loss like the Titanic script): under a
578:1 imbalance the loss is dominated by the majority class and barely moves
when fraud recall changes. PR-AUC (average precision) tracks exactly the
minority-class quality we care about, so it is the right early-stopping signal.

Run::

    python train_fraud.py                 # sane defaults, CPU-friendly
    python train_fraud.py --epochs 60 --hidden-dim 128 --n-blocks 3

Artifacts land in ``artifacts_fraud/``:

* ``model.pt``            - best weights + architecture hyper-params
* ``preprocessor.joblib`` - the fitted preprocessing pipeline
* ``metadata.json``       - feature names, val + test metrics, run config
* ``history.json``        - per-epoch learning curves
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.fraud_evaluation import compute_metrics, find_best_threshold
from src.fraud_preprocessing import (
    build_preprocessor,
    compute_pos_weight,
    get_feature_names,
    load_raw,
    make_splits,
    save_preprocessor,
    split_xy,
)
from src.model import TabularResNet


@dataclass
class TrainConfig:
    """Configuration for a fraud-model training run."""

    data_path: str = "data/creditcard.csv"
    out_dir: str = "artifacts_fraud"
    epochs: int = 40
    patience: int = 8
    batch_size: int = 2048          # large batches: 283k rows, CPU-friendly throughput
    lr: float = 1e-3
    weight_decay: float = 1e-4      # stronger L2: the 599:1 run overfit (val_loss diverged)
    hidden_dim: int = 128
    n_blocks: int = 2               # 3 blocks memorised the 378 train frauds; 2 generalises
    dropout: float = 0.4
    # pos_weight for BCEWithLogitsLoss. None -> sqrt(n_neg/n_pos) (~24.5): a far
    # gentler reweighting than the raw 599:1 ratio, which inflated logits (exp
    # overflow), pinned the decision threshold at ~0.996 and hurt PR-AUC. Pass a
    # float to override (e.g. the value found by tuning_fraud.py).
    pos_weight: float | None = None
    val_size: float = 0.10
    test_size: float = 0.10
    seed: int = 42
    device: str = "auto"


_TUNABLE_KEYS = ("lr", "dropout", "batch_size", "hidden_dim", "n_blocks",
                 "weight_decay", "pos_weight")


def load_best_params(path: str | Path = "fraud_best_params.json") -> dict[str, object]:
    """Load tuned hyperparameters written by ``tuning_fraud.py``, if present.

    A missing file is not an error — training falls back to the
    :class:`TrainConfig` defaults. Only keys the study actually searches
    (:data:`_TUNABLE_KEYS`) are returned, so a stray field can never silently
    change unrelated behaviour.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[warn] could not read {path} ({exc}); using default hyperparameters.")
        return {}
    params = payload.get("best_params", payload)
    return {k: params[k] for k in _TUNABLE_KEYS if k in params}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(np.asarray(y), dtype=torch.float32),
    )
    # drop_last on the shuffled (train) loader so BatchNorm never sees a 1-row batch.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              device: torch.device, threshold: float | None = None) -> dict[str, float]:
    """Run the model over a loader and return loss + fraud metrics.

    If ``threshold`` is None, the best-F1 threshold is searched on these
    predictions (used during training/validation). At test time pass the
    threshold frozen from the validation split to avoid peeking.
    """
    model.eval()
    logits_all, y_all, loss_sum, n = [], [], 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * len(yb)
        n += len(yb)
        logits_all.append(logits.cpu())
        y_all.append(yb.cpu())

    logits = torch.cat(logits_all).numpy()
    y_true = torch.cat(y_all).numpy()
    # Clip before exp: large positive logits (common under heavy pos_weight)
    # otherwise overflow np.exp and spam RuntimeWarnings. sigmoid saturates
    # anyway, so clipping at ±30 is numerically exact to float precision.
    proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    if threshold is None:
        threshold, _ = find_best_threshold(y_true, proba)
    metrics = compute_metrics(y_true, proba, threshold=threshold)
    return {"loss": loss_sum / max(n, 1), **metrics, "best_thresh": threshold}


def train(**overrides: object) -> dict[str, object]:
    """Run end-to-end fraud training and persist the best artifacts."""
    cfg = TrainConfig(**overrides)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data: load, split, fit preprocessor on TRAIN ONLY ---
    raw = load_raw(cfg.data_path)
    train_df, val_df, test_df = make_splits(raw, val_size=cfg.val_size, test_size=cfg.test_size)
    X_train_df, y_train = split_xy(train_df)
    X_val_df, y_val = split_xy(val_df)
    X_test_df, y_test = split_xy(test_df)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)
    feature_names = get_feature_names(preprocessor)

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    train_loader = _to_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = _to_loader(X_val, y_val, cfg.batch_size, shuffle=False)
    test_loader = _to_loader(X_test, y_test, cfg.batch_size, shuffle=False)

    # --- model / optim ---
    model = TabularResNet(
        in_features=X_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
    ).to(device)
    # Gentle reweighting: sqrt of the raw imbalance ratio unless overridden.
    base_pos_weight = compute_pos_weight(train_df["Class"])
    pw_value = cfg.pos_weight if cfg.pos_weight is not None else base_pos_weight ** 0.5
    pos_weight = torch.tensor([pw_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"[setup] device={device} | train={len(y_train)} val={len(y_val)} "
          f"test={len(y_test)} | features={X_train.shape[1]} | pos_weight={pos_weight.item():.1f}")

    best_pr_auc = -1.0
    best_epoch = -1
    best_val_metrics: dict[str, float] = {}
    best_thresh = 0.5
    epochs_no_improve = 0
    ckpt_path = out_dir / "model.pt"
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_pr_auc": [], "val_roc_auc": []
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
        train_loss = running / max(len(train_loader.dataset), 1)
        val = _evaluate(model, val_loader, criterion, device)

        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} "
            f"| PR-AUC {val['pr_auc']:.4f} | ROC-AUC {val['roc_auc']:.4f} "
            f"| P {val['precision']:.3f} R {val['recall']:.3f} F1 {val['f1']:.3f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_pr_auc"].append(val["pr_auc"])
        history["val_roc_auc"].append(val["roc_auc"])

        # Best checkpoint + early stopping keyed on val PR-AUC (the metric that
        # matters under extreme imbalance), not val-loss.
        if val["pr_auc"] > best_pr_auc:
            best_pr_auc = val["pr_auc"]
            best_epoch = epoch
            best_val_metrics = val
            best_thresh = val["best_thresh"]
            epochs_no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": {
                        "in_features": X_train.shape[1],
                        "hidden_dim": cfg.hidden_dim,
                        "n_blocks": cfg.n_blocks,
                        "dropout": cfg.dropout,
                    },
                    "threshold": best_thresh,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"[early-stop] no val PR-AUC improvement for {cfg.patience} epochs.")
                break

    # --- final test evaluation with the BEST model + frozen val threshold ---
    # weights_only=False: the checkpoint stores a dict (arch + threshold), not
    # just tensors, so it must use the full unpickler (we wrote the file ourselves).
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = _evaluate(model, test_loader, criterion, device, threshold=best_thresh)

    # --- persist preprocessor + metadata ---
    pre_path = save_preprocessor(preprocessor, out_dir / "preprocessor.joblib")
    meta = {
        "config": asdict(cfg),
        "device": str(device),
        "feature_names": feature_names,
        "best_epoch": best_epoch,
        "decision_threshold": best_thresh,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\n[done] best epoch {best_epoch} | threshold {best_thresh:.4f}")
    print(f"[VAL ] PR-AUC {best_val_metrics['pr_auc']:.4f} | ROC-AUC {best_val_metrics['roc_auc']:.4f} "
          f"| P {best_val_metrics['precision']:.3f} R {best_val_metrics['recall']:.3f} "
          f"F1 {best_val_metrics['f1']:.3f}")
    print(f"[TEST] PR-AUC {test_metrics['pr_auc']:.4f} | ROC-AUC {test_metrics['roc_auc']:.4f} "
          f"| P {test_metrics['precision']:.3f} R {test_metrics['recall']:.3f} "
          f"F1 {test_metrics['f1']:.3f}")
    print(f"[saved] {ckpt_path}\n[saved] {pre_path}\n[saved] {out_dir/'metadata.json'}")

    return {"val_metrics": best_val_metrics, "test_metrics": test_metrics,
            "model_path": str(ckpt_path)}


def _parse_args() -> TrainConfig:
    # Precedence: explicit CLI flag > fraud_best_params.json > TrainConfig default.
    # Tuned values become the argparse defaults, so an un-passed flag picks them
    # up while a passed flag still overrides.
    tuned = load_best_params()
    if tuned:
        print(f"[params] loaded tuned hyperparameters from fraud_best_params.json: {tuned}")
    d = lambda key: tuned.get(key, getattr(TrainConfig, key))  # noqa: E731

    p = argparse.ArgumentParser(description="Train the credit-card fraud model.")
    p.add_argument("--data-path", default=TrainConfig.data_path)
    p.add_argument("--out-dir", default=TrainConfig.out_dir)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--patience", type=int, default=TrainConfig.patience)
    p.add_argument("--batch-size", type=int, default=d("batch_size"))
    p.add_argument("--lr", type=float, default=d("lr"))
    p.add_argument("--weight-decay", type=float, default=d("weight_decay"))
    p.add_argument("--hidden-dim", type=int, default=d("hidden_dim"))
    p.add_argument("--n-blocks", type=int, default=d("n_blocks"))
    p.add_argument("--dropout", type=float, default=d("dropout"))
    p.add_argument("--pos-weight", type=float, default=d("pos_weight"))
    p.add_argument("--val-size", type=float, default=TrainConfig.val_size)
    p.add_argument("--test-size", type=float, default=TrainConfig.test_size)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--device", default=TrainConfig.device, choices=["auto", "cuda", "cpu"])
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    train(**asdict(_parse_args()))
