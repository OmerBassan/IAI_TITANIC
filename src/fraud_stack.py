"""Gradient-boosting + stacking ensemble for credit-card fraud detection.

Why this module exists
----------------------
The standalone ``TabularResNet`` (see ``train_fraud.py``) lands around
**PR-AUC ~0.84** on a single split. On this dataset that is a *strong baseline*,
not state of the art: on tabular data with 28 PCA features, gradient-boosted
trees consistently match or beat neural nets, and the real winning move is a
**diverse stacked ensemble** measured under **leakage-free cross-validation**.

This module implements exactly that:

* **Base learners** (diverse on purpose):
    - ``HistGradientBoostingClassifier`` — always available (sklearn), the spine.
    - LightGBM / XGBoost / CatBoost — auto-detected, added if installed.
    - The existing ``TabularResNet`` wrapped as a sklearn-style member, so the
      stack blends *trees + a neural net* (their errors are uncorrelated, which
      is what makes stacking pay off).
* **Out-of-fold (OOF) stacking**: each base learner produces predictions for
  rows it never trained on; a logistic-regression meta-learner is fit on those.
  The preprocessor is **re-fit inside every fold** — no statistic ever leaks
  from validation into training.
* **Honest measurement**: ``cv_report`` runs repeated stratified K-fold and
  reports **mean +/- std PR-AUC** per learner. With only ~490 frauds, a single
  split is noise; the std is the signal.
* **Clean threshold + final test**: the decision threshold is picked on
  *cross-validated* stack probabilities (never in-sample), then applied once to
  a held-out test split.

Run
---
::

    python -m src.fraud_stack --cv                 # repeated-CV leaderboard
    python -m src.fraud_stack                       # build + test + save stack
    python -m src.fraud_stack --nn --repeats 3      # include the NN member

Artifacts land in ``artifacts_stack/``:

* ``stack.joblib``    - fitted StackedFraudModel (preprocessor + bases + meta)
* ``metadata.json``   - per-learner CV scores, stack test metrics, threshold
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.fraud_evaluation import compute_metrics, find_best_threshold
from src.fraud_preprocessing import (
    RANDOM_STATE,
    TARGET,
    build_preprocessor,
    load_raw,
    split_xy,
)

# --------------------------------------------------------------------------- #
# Optional gradient-boosting back-ends. The module is fully functional with
# only sklearn (HistGradientBoosting); LightGBM/XGBoost/CatBoost are added to
# the ensemble automatically when importable. This keeps the file runnable on a
# bare environment while scaling up to the full boosting zoo when installed.
# --------------------------------------------------------------------------- #
try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except ImportError:  # pragma: no cover
    _HAS_CAT = False


# --------------------------------------------------------------------------- #
# Base-learner factory
# --------------------------------------------------------------------------- #
# Each entry maps a name -> a builder that takes the training labels (to set the
# class-imbalance weight) and returns a *fresh, unfitted* estimator. Builders
# take y so ``scale_pos_weight`` / ``class_weight`` is computed from the actual
# fold, never leaked across folds.
ModelBuilder = Callable[[np.ndarray], object]


def _imbalance_ratio(y: np.ndarray) -> float:
    """n_negative / n_positive for the training labels (>= 1)."""
    pos = max(int(np.sum(y == 1)), 1)
    neg = int(np.sum(y == 0))
    return neg / pos


def _build_histgb(y: np.ndarray) -> HistGradientBoostingClassifier:
    """sklearn histogram GBM (LightGBM-equivalent), always available.

    ``class_weight='balanced'`` is the sklearn-native way to up-weight the rare
    fraud class without resampling — analogous to ``scale_pos_weight`` in the
    external boosters. Tuned for an imbalanced, mid-size tabular problem:
    shallow-ish trees, many leaves, early-stopping on an internal validation cut.
    """
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1.0,
        max_features=0.8,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    )


def _build_lgbm(y: np.ndarray) -> object:
    return LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=64,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=_imbalance_ratio(y),
        objective="binary",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def _build_xgb(y: np.ndarray) -> object:
    return XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=_imbalance_ratio(y),
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _build_catboost(y: np.ndarray) -> object:
    return CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        scale_pos_weight=_imbalance_ratio(y),
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def base_model_builders(include_nn: bool = False, nn_kwargs: dict | None = None) -> dict[str, ModelBuilder]:
    """Return the active base-learner builders for this environment.

    HistGradientBoosting is always present; the external boosters are appended
    when their library is importable. The neural net member is opt-in (it is the
    slowest to fit but adds the most diversity to a tree-heavy ensemble).
    """
    builders: dict[str, ModelBuilder] = {"histgb": _build_histgb}
    if _HAS_LGBM:
        builders["lightgbm"] = _build_lgbm
    if _HAS_XGB:
        builders["xgboost"] = _build_xgb
    if _HAS_CAT:
        builders["catboost"] = _build_catboost
    if include_nn:
        kw = nn_kwargs or {}
        builders["resnet"] = lambda y: TorchResNetClassifier(**kw)
    return builders


def _proba1(model: object, X: np.ndarray) -> np.ndarray:
    """Positive-class probability, robust across sklearn / booster APIs."""
    proba = model.predict_proba(X)
    proba = np.asarray(proba)
    return proba[:, 1] if proba.ndim == 2 else proba.ravel()


# --------------------------------------------------------------------------- #
# Neural-net member: TabularResNet wrapped behind a fit / predict_proba API
# --------------------------------------------------------------------------- #
class TorchResNetClassifier:
    """sklearn-style wrapper around :class:`src.fraud_model.TabularResNet`.

    Lets the existing neural net act as one base learner inside the stack.
    Improvements over the standalone trainer, per the SOTA plan:

    * **Focal loss** (``gamma``) instead of a static ``pos_weight`` — it
      concentrates gradient on the hard, easily-missed frauds.
    * **Early stopping on internal-validation PR-AUC** so a noisy late epoch
      can't poison the OOF predictions.

    The wrapper is picklable: it stores the architecture spec plus a CPU
    ``state_dict``, so a fitted stack containing it round-trips through joblib.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 40,
        patience: int = 6,
        batch_size: int = 4096,
        focal_gamma: float = 2.0,
        val_fraction: float = 0.1,
        seed: int = RANDOM_STATE,
        device: str = "auto",
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.focal_gamma = focal_gamma
        self.val_fraction = val_fraction
        self.seed = seed
        self.device = device
        self._state_dict: dict | None = None
        self._in_features: int | None = None

    # -- internals ---------------------------------------------------------- #
    def _resolve_device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _new_model(self, in_features: int):
        from src.fraud_model import TabularResNet

        return TabularResNet(
            in_features=in_features,
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
        )

    @staticmethod
    def _focal_loss(logits, targets, gamma: float):
        """Binary focal loss on raw logits: (1 - p_t)^gamma * BCE."""
        import torch
        import torch.nn.functional as F

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** gamma * bce).mean()

    # -- sklearn-style API -------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchResNetClassifier":
        import torch
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self._resolve_device()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._in_features = X.shape[1]

        # Internal stratified validation cut for early stopping on PR-AUC.
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=self.val_fraction, random_state=self.seed, stratify=y
        )
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=self.batch_size, shuffle=True, drop_last=True,
        )
        Xva_t = torch.from_numpy(X_va).to(device)

        model = self._new_model(self._in_features).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        from src.fraud_evaluation import compute_metrics as _cm

        best_pr, best_state, no_improve = -1.0, None, 0
        for _ in range(self.epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = self._focal_loss(model(xb), yb, self.focal_gamma)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                logits = model(Xva_t).cpu().numpy()
            proba = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            pr = _cm(y_va, proba)["pr_auc"]
            if pr > best_pr:
                best_pr, no_improve = pr, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        self._state_dict = best_state if best_state is not None else {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._state_dict is None or self._in_features is None:
            raise RuntimeError("TorchResNetClassifier must be fit before predict_proba.")
        device = self._resolve_device()
        model = self._new_model(self._in_features).to(device)
        model.load_state_dict(self._state_dict)
        model.eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device)).cpu().numpy()
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        return np.column_stack([1.0 - p1, p1])

    # -- picklability: never pickle live torch modules, only the state dict -- #
    def __getstate__(self) -> dict:
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


# --------------------------------------------------------------------------- #
# The fitted stacked model
# --------------------------------------------------------------------------- #
@dataclass
class StackedFraudModel:
    """A fitted stacking ensemble: raw DataFrame in, fraud probability out.

    Holds the fitted preprocessor, the base learners (re-fit on the full
    training split), the logistic-regression meta-learner, the ordered base
    names, and the deployment threshold. ``predict_proba`` takes the **raw**
    credit-card columns and runs the whole pipeline.
    """

    preprocessor: object
    base_models: dict[str, object]
    meta: LogisticRegression
    base_names: list[str]
    threshold: float

    def _stack_matrix(self, X_raw) -> np.ndarray:
        X = self.preprocessor.transform(X_raw)
        return np.column_stack([_proba1(self.base_models[n], X) for n in self.base_names])

    def predict_proba(self, X_raw) -> np.ndarray:
        """Return fraud probabilities for a raw feature DataFrame."""
        return _proba1(self.meta, self._stack_matrix(X_raw))

    def predict(self, X_raw) -> np.ndarray:
        """Return 0/1 fraud labels using the frozen decision threshold."""
        return (self.predict_proba(X_raw) >= self.threshold).astype(int)


# --------------------------------------------------------------------------- #
# OOF stacking
# --------------------------------------------------------------------------- #
def _oof_base_predictions(
    builders: dict[str, ModelBuilder],
    X_raw,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build the leakage-free out-of-fold base-prediction matrix.

    For each stratified fold: fit the preprocessor on the fold's *train* rows,
    transform train + held-out, fit every base learner on train, and write its
    held-out predictions into the OOF matrix. Because the preprocessor is re-fit
    per fold, no scaling statistic leaks from a row into its own prediction.

    Returns:
        (Z, names) where ``Z`` is (n_samples, n_models) of OOF fraud
        probabilities and ``names`` are the matching column model names.
    """
    names = list(builders)
    Z = np.zeros((len(y), len(names)), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_raw = X_raw.reset_index(drop=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, y), start=1):
        pre = build_preprocessor()
        X_tr = pre.fit_transform(X_raw.iloc[tr_idx])
        X_va = pre.transform(X_raw.iloc[va_idx])
        y_tr = y[tr_idx]
        for j, name in enumerate(names):
            model = builders[name](y_tr)
            model.fit(X_tr, y_tr)
            Z[va_idx, j] = _proba1(model, X_va)
        if verbose:
            print(f"  [oof] fold {fold}/{n_splits} done")
    return Z, names


def fit_stack(
    X_raw,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = RANDOM_STATE,
    include_nn: bool = False,
    nn_kwargs: dict | None = None,
    verbose: bool = True,
) -> tuple[StackedFraudModel, dict[str, float], float]:
    """Fit the full stacking ensemble on a training split.

    Steps:
      1. Build OOF base predictions (leakage-free, preprocessor re-fit per fold).
      2. Fit the logistic-regression meta-learner on the OOF matrix.
      3. Pick the decision threshold from *cross-validated* stack probabilities
         (``cross_val_predict`` on the meta-learner), so the threshold is never
         chosen in-sample.
      4. Re-fit the preprocessor + every base learner on the full training split
         for deployment, and wrap everything in a :class:`StackedFraudModel`.

    Returns:
        (stacked_model, oof_scores, threshold) where ``oof_scores`` holds the
        cross-validated PR-AUC of each base learner and of the stack.
    """
    builders = base_model_builders(include_nn=include_nn, nn_kwargs=nn_kwargs)
    Z, names = _oof_base_predictions(builders, X_raw, y, n_splits, seed, verbose)

    # Cross-validated base-learner PR-AUC (each row scored by a model that never
    # saw it -> an honest generalization estimate over the whole train split).
    oof_scores = {name: compute_metrics(y, Z[:, j])["pr_auc"] for j, name in enumerate(names)}

    # Meta-learner. Clean stack probabilities via an inner CV so the threshold
    # and the reported stack score are not measured on data the meta-learner saw.
    meta = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clean_stack = cross_val_predict(
        meta, Z, y, cv=n_splits, method="predict_proba",
        n_jobs=None,
    )[:, 1]
    oof_scores["stack"] = compute_metrics(y, clean_stack)["pr_auc"]
    threshold, _ = find_best_threshold(y, clean_stack)

    # Deployment fit: meta on full OOF, bases + preprocessor on full train.
    meta.fit(Z, y)
    pre = build_preprocessor()
    X_full = pre.fit_transform(X_raw)
    base_models = {name: builders[name](y).fit(X_full, y) for name in names}

    stacked = StackedFraudModel(
        preprocessor=pre, base_models=base_models, meta=meta,
        base_names=names, threshold=float(threshold),
    )
    return stacked, oof_scores, float(threshold)


# --------------------------------------------------------------------------- #
# Honest measurement: repeated stratified CV leaderboard
# --------------------------------------------------------------------------- #
def cv_report(
    X_raw,
    y: np.ndarray,
    n_splits: int = 5,
    repeats: int = 3,
    include_nn: bool = False,
    nn_kwargs: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Repeated stratified K-fold PR-AUC leaderboard for every base learner.

    With ~490 frauds, a single split's PR-AUC swings by several points run to
    run, so a lone number is meaningless. This runs ``repeats`` independent
    ``n_splits``-fold CVs (different seeds) and reports **mean +/- std** PR-AUC
    per learner — the honest way to compare models on this dataset.

    The preprocessor is re-fit inside every fold (no leakage). Returns a dict of
    ``{name: {"mean": ..., "std": ..., "scores": [...]}}``.
    """
    builders = base_model_builders(include_nn=include_nn, nn_kwargs=nn_kwargs)
    names = list(builders)
    per_model: dict[str, list[float]] = {n: [] for n in names}
    X_raw = X_raw.reset_index(drop=True)

    for r in range(repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + r)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, y), start=1):
            pre = build_preprocessor()
            X_tr = pre.fit_transform(X_raw.iloc[tr_idx])
            X_va = pre.transform(X_raw.iloc[va_idx])
            y_tr, y_va = y[tr_idx], y[va_idx]
            for name in names:
                model = builders[name](y_tr)
                model.fit(X_tr, y_tr)
                pr = compute_metrics(y_va, _proba1(model, X_va))["pr_auc"]
                per_model[name].append(pr)
            print(f"[cv] repeat {r + 1}/{repeats} fold {fold}/{n_splits} "
                  + " | ".join(f"{n}={per_model[n][-1]:.4f}" for n in names))

    report = {
        n: {"mean": float(np.mean(s)), "std": float(np.std(s)), "scores": [float(v) for v in s]}
        for n, s in per_model.items()
    }
    print("\n=== Repeated-CV PR-AUC leaderboard "
          f"({repeats}x{n_splits}-fold, {len(y)} rows) ===")
    for n, st in sorted(report.items(), key=lambda kv: kv[1]["mean"], reverse=True):
        print(f"  {n:12s} PR-AUC {st['mean']:.4f} +/- {st['std']:.4f}")
    return report


# --------------------------------------------------------------------------- #
# Orchestration / CLI
# --------------------------------------------------------------------------- #
@dataclass
class StackConfig:
    data_path: str = "data/creditcard.csv"
    out_dir: str = "artifacts_stack"
    n_splits: int = 5
    repeats: int = 3
    test_size: float = 0.10
    include_nn: bool = False
    cv_only: bool = False
    nn_kwargs: dict = field(default_factory=dict)


def run(cfg: StackConfig) -> dict:
    """Build the stack: optional CV leaderboard, then fit + test + save."""
    t0 = time.time()
    raw = load_raw(cfg.data_path)
    active = list(base_model_builders(include_nn=cfg.include_nn))
    print(f"[stack] rows={len(raw)} | base learners: {active}")
    if not (_HAS_LGBM or _HAS_XGB or _HAS_CAT):
        print("[note] LightGBM/XGBoost/CatBoost not installed -> using sklearn "
              "HistGradientBoosting as the GBM spine. `pip install lightgbm "
              "xgboost catboost` to add them to the ensemble automatically.")

    X_all, y_all = split_xy(raw)
    y_all = y_all.to_numpy()

    result: dict = {"active_models": active}

    if cfg.repeats > 0:
        result["cv_report"] = cv_report(
            X_all, y_all, n_splits=cfg.n_splits, repeats=cfg.repeats,
            include_nn=cfg.include_nn, nn_kwargs=cfg.nn_kwargs,
        )
    if cfg.cv_only:
        return result

    # Held-out test split (touched exactly once, at the very end).
    train_df, test_df = _train_test_split(raw, cfg.test_size)
    X_tr, y_tr = split_xy(train_df)
    X_te, y_te = split_xy(test_df)
    y_tr, y_te = y_tr.to_numpy(), y_te.to_numpy()

    print(f"\n[stack] fitting OOF stack on {len(y_tr)} train rows...")
    stacked, oof_scores, threshold = fit_stack(
        X_tr, y_tr, n_splits=cfg.n_splits, include_nn=cfg.include_nn,
        nn_kwargs=cfg.nn_kwargs,
    )
    print("[stack] cross-validated PR-AUC: "
          + " | ".join(f"{k}={v:.4f}" for k, v in oof_scores.items()))

    test_proba = stacked.predict_proba(X_te)
    test_metrics = compute_metrics(y_te, test_proba, threshold=threshold)
    print(f"\n[TEST] PR-AUC {test_metrics['pr_auc']:.4f} | ROC-AUC {test_metrics['roc_auc']:.4f} "
          f"| P {test_metrics['precision']:.3f} R {test_metrics['recall']:.3f} "
          f"F1 {test_metrics['f1']:.3f} @ thr {threshold:.4f}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(stacked, out_dir / "stack.joblib")
    meta = {
        "active_models": active,
        "oof_cv_pr_auc": oof_scores,
        "decision_threshold": threshold,
        "test_metrics": test_metrics,
        "cv_report": result.get("cv_report"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[saved] {out_dir / 'stack.joblib'}\n[saved] {out_dir / 'metadata.json'}")

    result.update({"oof_scores": oof_scores, "test_metrics": test_metrics,
                   "threshold": threshold})
    return result


def _train_test_split(raw, test_size: float):
    """Stratified train/test split on the raw frame (preserves fraud rate)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        raw, test_size=test_size, random_state=RANDOM_STATE, stratify=raw[TARGET]
    )


def _parse_args() -> StackConfig:
    p = argparse.ArgumentParser(description="LightGBM/HistGB + stacking fraud ensemble.")
    p.add_argument("--data-path", default=StackConfig.data_path)
    p.add_argument("--out-dir", default=StackConfig.out_dir)
    p.add_argument("--n-splits", type=int, default=StackConfig.n_splits)
    p.add_argument("--repeats", type=int, default=StackConfig.repeats,
                   help="repeated-CV passes for the leaderboard (0 to skip)")
    p.add_argument("--test-size", type=float, default=StackConfig.test_size)
    p.add_argument("--nn", action="store_true", help="include the TabularResNet member")
    p.add_argument("--cv", dest="cv_only", action="store_true",
                   help="only run the repeated-CV leaderboard, skip fit/test/save")
    a = p.parse_args()
    return StackConfig(
        data_path=a.data_path, out_dir=a.out_dir, n_splits=a.n_splits,
        repeats=a.repeats, test_size=a.test_size, include_nn=a.nn, cv_only=a.cv_only,
    )


if __name__ == "__main__":
    run(_parse_args())
