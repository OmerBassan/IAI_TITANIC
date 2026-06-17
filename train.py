"""Training script for the Titanic survival-prediction model.

Pipeline: load ``train.csv`` -> stratified train/val split -> fit the
preprocessing pipeline **on the training split only** (no leakage) -> train the
``TabularResNet`` with early stopping -> save the best checkpoint (selected on
the validation ROC-AUC), the fitted preprocessor and a metadata file.

Runnable two ways:

* **CLI**::

      python train.py --epochs 200 --patience 20

* **Colab / notebook**::

      from train import train
      train(epochs=200, device="cuda")

All randomness is keyed off ``--seed`` (default 42) for reproducibility.
Artifacts land in ``checkpoints/`` by default:

* ``model.pt``        - best model weights + architecture hyper-params
* ``preprocessor.joblib`` - the fitted preprocessing pipeline
* ``metadata.json``   - feature names, metrics, hyper-params, run config
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from src.evaluation import compute_metrics, find_best_threshold
from src.model import TabularResNet
from src.preprocessing import (
    build_preprocessor,
    get_feature_names,
    load_preprocessor,
    load_raw,
    save_preprocessor,
    split_xy,
)


@dataclass
class TrainConfig:
    """Configuration for a training run.

    Attributes:
        data_path: Path to the raw training CSV.
        out_dir: Directory artifacts are written to.
        epochs: Maximum number of training epochs.
        patience: Early-stopping patience (epochs without val-AUC improvement).
        batch_size: Mini-batch size.
        lr: AdamW learning rate.
        weight_decay: AdamW decoupled weight-decay (strong, to penalise large
            weights on the small ~900-row dataset).
        hidden_dim: Residual width of the model.
        n_blocks: Number of residual blocks.
        dropout: Dropout probability.
        val_size: Fraction of data held out for validation.
        seed: Global random seed.
        device: Torch device string ("cuda", "cpu", or "auto").
        splits_path: Optional ``.npz`` of precomputed splits (see
            ``prepare_data.py``). When set, ``data_path`` is ignored and the
            train/val split + preprocessing are loaded instead of recomputed -
            useful for preprocessing locally and training in Colab.
        preprocessor_path: Path to the fitted preprocessor that produced
            ``splits_path``; required when ``splits_path`` is set.
    """

    data_path: str = "data/train.csv"
    out_dir: str = "artifacts"
    splits_path: str | None = None
    preprocessor_path: str | None = None
    epochs: int = 200
    patience: int = 20
    # Plain, sane fallbacks. The *tuned* hyperparameters live in best_params.json
    # (produced by tuning.py, committed to Git) and override these at startup —
    # see load_best_params(). weight_decay is not searched, so it stays here.
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2
    hidden_dim: int = 32
    n_blocks: int = 2
    dropout: float = 0.2
    val_size: float = 0.2
    seed: int = 42
    device: str = "auto"


_TUNABLE_KEYS = ("lr", "dropout", "batch_size", "hidden_dim", "n_blocks")


def load_best_params(path: str | Path = "best_params.json") -> dict[str, object]:
    """Load tuned hyperparameters written by ``tuning.py``, if present.

    Lets training stay decoupled from the (expensive, occasional) Optuna search:
    ``tuning.py`` writes ``best_params.json``, this reads it. Only the keys the
    study actually searches (:data:`_TUNABLE_KEYS`) are returned, so a stray
    field in the file can never silently change unrelated training behaviour.

    Args:
        path: Path to the params JSON. A missing file is not an error — it just
            means training falls back to the :class:`TrainConfig` defaults.

    Returns:
        A dict of recognised hyperparameter overrides (possibly empty).
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[warn] could not read {path} ({exc}); using default hyperparameters.")
        return {}
    # Support both a flat dict and the {"best_params": {...}} provenance wrapper.
    params = payload.get("best_params", payload)
    return {k: params[k] for k in _TUNABLE_KEYS if k in params}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducibility.

    Args:
        seed: The integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """Resolve a device string, auto-selecting CUDA when available.

    Args:
        device: "auto", "cuda" or "cpu".

    Returns:
        A ``torch.device``.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _to_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    """Wrap a feature matrix and labels in a DataLoader of float32 tensors.

    Args:
        X: Feature matrix ``(n, d)``.
        y: Binary labels ``(n,)``.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle each epoch. When shuffling (train), the
            last batch is dropped so BatchNorm never sees a batch of size 1.

    Returns:
        A configured ``DataLoader``.
    """
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


@torch.no_grad()
def _evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> dict[str, float]:
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
    proba = 1.0 / (1.0 + np.exp(-logits))

    # דינמי threshold search + metrics via the shared evaluation module, so the
    # numbers reported here match exactly what ds_app.py shows on the same data.
    best_thresh, _ = find_best_threshold(y_true, proba)
    metrics = compute_metrics(y_true, proba, threshold=best_thresh)
    return {"loss": loss_sum / max(n, 1), **metrics, "best_thresh": best_thresh}

def train(**overrides: object) -> dict[str, object]:
    """Run end-to-end training and persist the best artifacts.

    Keyword overrides map onto :class:`TrainConfig` fields, e.g.
    ``train(epochs=300, device="cuda")``.

    Returns:
        A dict with the best validation metrics and the artifact paths.
    """
    cfg = TrainConfig(**overrides)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir / "logs")

    # --- data ---
    if cfg.splits_path:
        # Precomputed locally (prepare_data.py) and shipped to Colab via Drive.
        if not cfg.preprocessor_path:
            raise ValueError("preprocessor_path is required when splits_path is set.")
        npz = np.load(cfg.splits_path)
        X_train, y_train = npz["X_train"], npz["y_train"]
        X_val, y_val = npz["X_val"], npz["y_val"]
        preprocessor = load_preprocessor(cfg.preprocessor_path)
        feature_names = get_feature_names(preprocessor)
        # Guardrail: a cached splits.npz silently goes stale when the
        # preprocessing code changes (e.g. a new feature is added) but the cache
        # is not regenerated. Compare the engineered schema frozen into the cache
        # against the one the *current* code produces, and fail loudly on drift.
        live = list(
            build_preprocessor().named_steps["engineer"].get_feature_names_out()
        )
        if "engineered_features" in npz:
            cached = [str(f) for f in npz["engineered_features"]]
            if cached != live:
                added = sorted(set(live) - set(cached))
                removed = sorted(set(cached) - set(live))
                raise ValueError(
                    "Stale cache: splits.npz was built with a different "
                    "preprocessing schema than the current code "
                    f"(added={added or 'none'}, removed={removed or 'none'}). "
                    "Regenerate with `python prepare_data.py` "
                    "(and re-upload it if on Colab)."
                )
        else:
            print("[warn] cache predates schema tracking; cannot verify it is "
                  "current. Regenerate with prepare_data.py if unsure.")
        print(f"[data] loaded precomputed splits from {cfg.splits_path} "
              f"({X_train.shape[1]} features)")
    else:
        raw = load_raw(cfg.data_path)
        X_df, y_ser = split_xy(raw)
        if y_ser is None:
            raise ValueError(f"{cfg.data_path} has no 'Survived' column - cannot train.")
        X_train_df, X_val_df, y_train, y_val = train_test_split(
            X_df,
            y_ser.to_numpy(),
            test_size=cfg.val_size,
            stratify=y_ser,
            random_state=cfg.seed,
        )
        # preprocessing: FIT ON TRAIN ONLY
        preprocessor = build_preprocessor()
        X_train = preprocessor.fit_transform(X_train_df, y_train)
        X_val = preprocessor.transform(X_val_df)
        feature_names = get_feature_names(preprocessor)

    train_loader = _to_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = _to_loader(X_val, y_val, cfg.batch_size, shuffle=False)

    # --- model / optim ---
    model = TabularResNet(
        in_features=X_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
    ).to(device)
    # pos_weight handles the mild class imbalance (~38% positive).
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # AdamW decouples weight decay from the gradient update -> cleaner, stronger
    # L2 regularisation than vanilla Adam, which matters on this tiny dataset.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    print(f"[setup] device={device} | train={len(y_train)} val={len(y_val)} "
          f"| features={X_train.shape[1]}")

    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    epochs_no_improve = 0
    ckpt_path = out_dir / "model.pt"
    # Per-epoch history -> learning curves (plotted by src.evaluation, reused
    # in the Streamlit app). Persisted to history.json alongside the weights.
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_accuracy": [], "val_roc_auc": []
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
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
            f"val_loss {val['loss']:.4f} | acc {val['accuracy']:.3f} | "
            f"f1 {val['f1']:.3f} | auc {val['roc_auc']:.3f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        history["val_roc_auc"].append(val["roc_auc"])

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val['loss'], epoch)
        writer.add_scalar("Metrics/Accuracy", val['accuracy'], epoch)
        writer.add_scalar("Metrics/ROC_AUC", val['roc_auc'], epoch)


        # best checkpoint + early stopping both keyed off validation loss:
        # we save while val_loss keeps falling and stop once it starts rising
        # (the classic overfitting signal, expected early on this small set).
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_epoch = epoch
            best_metrics = val
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
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"[early-stop] no val-loss improvement for {cfg.patience} epochs.")
                break

    # --- persist preprocessor + metadata ---
    pre_path = save_preprocessor(preprocessor, out_dir / "preprocessor.joblib")
    meta = {
        "config": asdict(cfg),
        "device": str(device),
        "feature_names": feature_names,
        "best_epoch": best_epoch,
        "val_metrics": best_metrics,
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    history_path = out_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(
        f"\n[done] best epoch {best_epoch} | "
        f"val acc {best_metrics.get('accuracy', float('nan')):.3f} | "
        f"f1 {best_metrics.get('f1', float('nan')):.3f} | "
        f"auc {best_metrics.get('roc_auc', float('nan')):.3f}"
    )
    print(f"[saved] {ckpt_path}\n[saved] {pre_path}\n[saved] {meta_path}"
          f"\n[saved] {history_path}")

    writer.close()

    return {
        "best_metrics": best_metrics,
        "model_path": str(ckpt_path),
        "preprocessor_path": str(pre_path),
        "metadata_path": str(meta_path),
    }


def _parse_args() -> TrainConfig:
    """Parse CLI arguments into a :class:`TrainConfig`.

    Returns:
        The populated config.
    """
    # Precedence: explicit CLI flag > best_params.json > TrainConfig default.
    # The tuned values become the argparse defaults, so an un-passed flag picks
    # them up while a passed flag still overrides.
    tuned = load_best_params()
    if tuned:
        print(f"[params] loaded tuned hyperparameters from best_params.json: {tuned}")
    d = lambda key: tuned.get(key, getattr(TrainConfig, key))  # noqa: E731

    p = argparse.ArgumentParser(description="Train the Titanic survival model.")
    p.add_argument("--data-path", default=TrainConfig.data_path)
    p.add_argument("--out-dir", default=TrainConfig.out_dir)
    p.add_argument("--splits-path", default=TrainConfig.splits_path)
    p.add_argument("--preprocessor-path", default=TrainConfig.preprocessor_path)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--patience", type=int, default=TrainConfig.patience)
    p.add_argument("--batch-size", type=int, default=d("batch_size"))
    p.add_argument("--lr", type=float, default=d("lr"))
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--hidden-dim", type=int, default=d("hidden_dim"))
    p.add_argument("--n-blocks", type=int, default=d("n_blocks"))
    p.add_argument("--dropout", type=float, default=d("dropout"))
    p.add_argument("--val-size", type=float, default=TrainConfig.val_size)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--device", default=TrainConfig.device, choices=["auto", "cuda", "cpu"])
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(**asdict(_parse_args()))
