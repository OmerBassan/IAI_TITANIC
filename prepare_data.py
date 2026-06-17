"""Preprocess the Titanic data **locally** and cache the result for training.

Runs the full train/val split + preprocessing (fit on the training split only,
exactly as ``train.py`` would) and writes two small artifacts:

* ``<out_dir>/splits.npz``        - X_train, y_train, X_val, y_val (dense float)
* ``<out_dir>/preprocessor.joblib`` - the fitted preprocessing pipeline

Upload both to Google Drive and train on a Colab GPU without re-running
preprocessing (or needing Kaggle credentials) there::

    from train import train
    train(splits_path="splits.npz", preprocessor_path="preprocessor.joblib",
          device="cuda")

The split is keyed off ``--seed`` so it matches ``train.py``'s default split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing import build_preprocessor, load_raw, save_preprocessor, split_xy


def prepare(
    data_path: str = "data/train.csv",
    out_dir: str = "artifacts",
    val_size: float = 0.2,
    seed: int = 42,
) -> dict[str, str]:
    """Split, fit the preprocessor on train, transform, and cache to disk.

    Args:
        data_path: Path to the raw training CSV.
        out_dir: Directory the cached artifacts are written to.
        val_size: Validation fraction.
        seed: Random seed (must match ``train.py`` for a consistent split).

    Returns:
        Dict of the written artifact paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(data_path)
    X_df, y_ser = split_xy(raw)
    if y_ser is None:
        raise ValueError(f"{data_path} has no 'Survived' column - cannot prepare.")

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_df,
        y_ser.to_numpy(),
        test_size=val_size,
        stratify=y_ser,
        random_state=seed,
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_df, y_train)  # FIT ON TRAIN ONLY
    X_val = preprocessor.transform(X_val_df)

    splits_path = out / "splits.npz"
    np.savez_compressed(
        splits_path,
        X_train=X_train.astype("float32"),
        y_train=y_train.astype("float32"),
        X_val=X_val.astype("float32"),
        y_val=y_val.astype("float32"),
    )
    pre_path = save_preprocessor(preprocessor, out / "preprocessor.joblib")

    print(f"[ok] train {X_train.shape} | val {X_val.shape} | features {X_train.shape[1]}")
    print(f"[saved] {splits_path}\n[saved] {pre_path}")
    return {"splits_path": str(splits_path), "preprocessor_path": str(pre_path)}


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description="Preprocess Titanic data locally and cache it.")
    p.add_argument("--data-path", default="data/train.csv")
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(
        data_path=args.data_path,
        out_dir=args.out_dir,
        val_size=args.val_size,
        seed=args.seed,
    )
