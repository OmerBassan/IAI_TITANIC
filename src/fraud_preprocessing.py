"""Preprocessing pipeline for the credit-card fraud detection dataset.

Raw dataset: 284,807 transactions × 31 columns (Time, V1–V28, Amount, Class).
V1–V28 are PCA-anonymised features provided by the original authors; Time and
Amount are the only interpretable raw columns.

Engineering decisions (justified by EDA):
- Drop 1,081 exact duplicate rows found in the raw file.
- Amount  → log1p transform (right-skewed, 0–25 691) + binary small_amount
  flag (Amount < 10): 50.6 % of fraud falls here vs. 34.1 % of legit.
- Time    → cyclical sin/cos on a 24-hour period; fraud has hour-of-day
  signal (spikes at ~02:00 and ~11:00 local time).
- V1–V28 → StandardScaler; PCA already centres them but variances differ
  meaningfully (V1 ≈ 3.8, V28 ≈ 0.06).
- Class imbalance (578:1) is handled at training time via pos_weight;
  compute_pos_weight() exposes the correct scalar for BCEWithLogitsLoss.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Class"
RANDOM_STATE = 42

_V_FEATURES = [f"V{i}" for i in range(1, 29)]          # 28 PCA columns
_AMOUNT_LOG = "amount_log"
_TIME_SIN = "time_hour_sin"
_TIME_COS = "time_hour_cos"
_SMALL_AMOUNT = "small_amount"

# Features sent through StandardScaler
_NUMERIC_FEATURES = _V_FEATURES + [_AMOUNT_LOG, _TIME_SIN, _TIME_COS]  # 31
# Binary flag passed through unchanged
_BINARY_FEATURES = [_SMALL_AMOUNT]                                       # 1

# Total output features: 32
FEATURE_NAMES: list[str] = _NUMERIC_FEATURES + _BINARY_FEATURES


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature engineering for the raw credit-card fraud columns.

    No statistics are learned from training data — every transform is a
    deterministic function of the input values — so fit() is a no-op.
    This keeps the transformer safe to use on single inference rows.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "FraudFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Amount: log1p to compress the long tail, plus small-transaction flag
        df[_AMOUNT_LOG] = np.log1p(df["Amount"].clip(lower=0))
        df[_SMALL_AMOUNT] = (df["Amount"] < 10.0).astype(np.float32)

        # Time: seconds since first transaction → hour-of-day cyclical encoding
        seconds_in_day = 86_400.0
        hour_phase = 2 * np.pi * (df["Time"] % seconds_in_day) / seconds_in_day
        df[_TIME_SIN] = np.sin(hour_phase)
        df[_TIME_COS] = np.cos(hour_phase)

        return df[FEATURE_NAMES]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(FEATURE_NAMES)


def build_preprocessor() -> Pipeline:
    """Return an unfitted preprocessing pipeline.

    Pipeline: FraudFeatureEngineer → ColumnTransformer
      - StandardScaler on 31 numeric features (V1-V28, amount_log, sin, cos)
      - passthrough on 1 binary flag (small_amount)
    """
    col_transform = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUMERIC_FEATURES),
            ("bin", "passthrough", _BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("engineer", FraudFeatureEngineer()),
            ("scale", col_transform),
        ]
    )


def load_raw(path: str | Path, drop_duplicates: bool = True) -> pd.DataFrame:
    """Load the raw creditcard.csv, optionally dropping exact duplicate rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        if dropped:
            print(f"[info] Dropped {dropped} duplicate rows ({before} -> {len(df)})")
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature columns from the Class target."""
    return df.drop(columns=[TARGET]), df[TARGET].astype(np.int64)


def make_splits(
    df: pd.DataFrame,
    val_size: float = 0.10,
    test_size: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / val / test split preserving the 0.17 % fraud rate.

    Args:
        df: Full raw DataFrame (after drop_duplicates).
        val_size: Fraction of the full dataset reserved for validation.
        test_size: Fraction of the full dataset reserved for test.

    Returns:
        (train_df, val_df, test_df)
    """
    hold = val_size + test_size
    train_df, tmp_df = train_test_split(
        df, test_size=hold, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    relative_test = test_size / hold
    val_df, test_df = train_test_split(
        tmp_df, test_size=relative_test, random_state=RANDOM_STATE, stratify=tmp_df[TARGET]
    )
    return train_df, val_df, test_df


def compute_pos_weight(y_train: pd.Series) -> float:
    """Return the pos_weight scalar for torch.nn.BCEWithLogitsLoss.

    pos_weight = n_negative / n_positive  (the reciprocal of fraud rate).
    Passing this to BCEWithLogitsLoss up-weights the rare positive class
    without any data resampling.
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    return n_neg / n_pos


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Return output feature names of a *fitted* pipeline."""
    return list(preprocessor.named_steps["scale"].get_feature_names_out())


def save_preprocessor(preprocessor: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    return path


def load_preprocessor(path: str | Path) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {path}. Run train first.")
    return joblib.load(path)


if __name__ == "__main__":
    data_path = Path("data/creditcard.csv")

    raw = load_raw(data_path)
    train_df, val_df, test_df = make_splits(raw)

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    names = get_feature_names(pre)
    pos_weight = compute_pos_weight(y_train)

    print(f"[ok] Raw rows after dedup : {len(raw)}")
    print(f"[ok] Train / Val / Test   : {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"[ok] Feature matrix shape : {X_train_t.shape}  (train)")
    print(f"[ok] Output features ({len(names)}):")
    for n in names:
        print(f"       - {n}")
    print(f"[ok] Fraud in train split : {int(y_train.sum())} / {len(y_train)} ({y_train.mean()*100:.3f} %)")
    print(f"[ok] pos_weight for BCEWithLogitsLoss : {pos_weight:.1f}")
