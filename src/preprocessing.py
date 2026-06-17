"""Shared preprocessing pipeline for the Titanic survival-prediction project.

This is the single source of truth for turning the raw Kaggle ``train.csv``
columns into the numeric feature matrix the model consumes. It is imported by
**both** ``train.py`` (fit on the training split) and ``ds_app.py`` (transform
user input at inference time), so train-time and inference-time transforms are
guaranteed identical: the fitted pipeline is persisted to disk and reloaded.

The concrete decisions implemented here come straight from the EDA notebook
(``notebooks/eda.ipynb``, section 10):

* Impute ``Age`` by the median within ``(Title, Pclass)`` groups (falling back
  to a global median for unseen groups), not a single global median.
* Fill ``Embarked`` with its mode; fill missing ``Fare`` with the class median.
* Replace the ~77%-missing ``Cabin`` with a binary ``HasCabin`` flag.
* ``log1p(Fare)`` to tame the right skew, then standardise alongside ``Age``.
* Engineer ``Title``, ``FamilySize``, ``IsAlone``, and ``IsChild`` (Age < 10);
  drop the raw free-text / identifier columns (``Name``, ``Ticket``,
  ``PassengerId``, ``Cabin``).
* One-hot encode the categoricals (``Sex``, ``Embarked``, ``Title``,
  ``Pclass``); keep a single family-size representation to avoid redundancy.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # joblib ships with scikit-learn but import defensively for a clear error
    import joblib
except ImportError as exc:  # pragma: no cover - environment misconfiguration
    raise ImportError(
        "joblib is required to persist the preprocessor. "
        "Install dependencies: pip install -r requirements.txt"
    ) from exc

TARGET = "Survived"
RANDOM_STATE = 42

# Titles extracted from Name that we keep as their own category; everything
# else collapses into "Rare". Mlle/Ms are spelling variants of Miss; Mme of Mrs.
_TITLE_KEEP = {"Mr", "Mrs", "Miss", "Master"}
_TITLE_NORMALISE = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
_TITLE_PATTERN = re.compile(r",\s*([^.]+)\.")

# Columns produced by the feature-engineering step, grouped by how the
# downstream ColumnTransformer should treat them.
_NUMERIC_FEATURES = ["Age", "FareLog", "FamilySize"]
_CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Pclass"]
_BINARY_FEATURES = ["IsAlone", "HasCabin", "IsChild"]


def _extract_title(name: object) -> str:
    """Extract a normalised honorific title from a passenger ``Name`` string.

    Args:
        name: The raw ``Name`` field (e.g. ``"Braund, Mr. Owen Harris"``).

    Returns:
        One of ``{"Mr", "Mrs", "Miss", "Master", "Rare"}``.
    """
    if not isinstance(name, str):
        return "Rare"
    match = _TITLE_PATTERN.search(name)
    if not match:
        return "Rare"
    title = match.group(1).strip()
    title = _TITLE_NORMALISE.get(title, title)
    return title if title in _TITLE_KEEP else "Rare"


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateful feature engineering for the raw Titanic columns.

    Learns the imputation statistics (group-wise age medians, embarked mode,
    class-wise fare medians) on the training data during :meth:`fit`, then
    applies them in :meth:`transform`. Keeping the statistics here — rather than
    recomputing them per call — is what prevents train/inference skew and data
    leakage from the validation split.

    The transformer is tolerant of inference-time input that omits the
    ``Survived`` target or arrives as a single row from the Streamlit form.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "TitanicFeatureEngineer":
        """Learn imputation statistics from the training data.

        Args:
            X: Raw feature frame (Kaggle ``train.csv`` columns, target optional).
            y: Ignored; present for scikit-learn API compatibility.

        Returns:
            The fitted transformer.
        """
        df = X.copy()
        df["Title"] = df["Name"].apply(_extract_title) if "Name" in df else "Rare"

        # Group-wise age median, with a global fallback for unseen (Title, Pclass).
        self.age_medians_: dict[tuple[str, object], float] = (
            df.groupby(["Title", "Pclass"])["Age"].median().dropna().to_dict()
        )
        self.global_age_median_: float = float(df["Age"].median())
        self.embarked_mode_: str = (
            df["Embarked"].mode(dropna=True).iloc[0]
            if df["Embarked"].notna().any()
            else "S"
        )
        self.fare_medians_: dict[object, float] = (
            df.groupby("Pclass")["Fare"].median().dropna().to_dict()
        )
        self.global_fare_median_: float = float(df["Fare"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and imputation using learned statistics.

        Args:
            X: Raw feature frame to transform.

        Returns:
            A DataFrame with exactly the engineered feature columns
            (numeric, categorical and binary), ready for the encoder/scaler.
        """
        df = X.copy()

        # --- engineered categorical ---
        df["Title"] = df["Name"].apply(_extract_title) if "Name" in df else "Rare"

        # --- family structure ---
        sibsp = df.get("SibSp", pd.Series(0, index=df.index)).fillna(0)
        parch = df.get("Parch", pd.Series(0, index=df.index)).fillna(0)
        df["FamilySize"] = sibsp + parch + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

        # --- cabin presence as signal, raw cabin dropped ---
        df["HasCabin"] = df.get("Cabin", pd.Series(np.nan, index=df.index)).notna().astype(int)

        # --- age: median within (Title, Pclass), then global fallback ---
        def _impute_age(row: pd.Series) -> float:
            if pd.notna(row["Age"]):
                return float(row["Age"])
            return float(
                self.age_medians_.get(
                    (row["Title"], row["Pclass"]), self.global_age_median_
                )
            )

        df["Age"] = df.apply(_impute_age, axis=1)

        # --- child effect: Age < 10 (captured by "women and children first") ---
        df["IsChild"] = (df["Age"] < 10).astype(int)

        # --- embarked: mode ---
        df["Embarked"] = df.get("Embarked", self.embarked_mode_).fillna(self.embarked_mode_)

        # --- fare: class median, then log1p ---
        def _impute_fare(row: pd.Series) -> float:
            if pd.notna(row.get("Fare")):
                return float(row["Fare"])
            return float(self.fare_medians_.get(row["Pclass"], self.global_fare_median_))

        df["Fare"] = df.apply(_impute_fare, axis=1)
        df["FareLog"] = np.log1p(df["Fare"].clip(lower=0))

        return df[_NUMERIC_FEATURES + _CATEGORICAL_FEATURES + _BINARY_FEATURES]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the engineered column names (scikit-learn API).

        Args:
            input_features: Ignored; present for API compatibility.

        Returns:
            Array of engineered feature names.
        """
        return np.asarray(
            _NUMERIC_FEATURES + _CATEGORICAL_FEATURES + _BINARY_FEATURES
        )


def build_preprocessor() -> Pipeline:
    """Build the full (unfitted) preprocessing pipeline.

    The pipeline is: feature engineering → column transform (one-hot encode the
    categoricals, standardise the numerics, pass the binary flags through).

    Returns:
        An unfitted scikit-learn :class:`~sklearn.pipeline.Pipeline`.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), _NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", _BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("engineer", TitanicFeatureEngineer()),
            ("encode", column_transformer),
        ]
    )


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a raw Titanic CSV from disk.

    Args:
        path: Path to a CSV with the Kaggle ``train.csv`` schema.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file cannot be parsed as CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. "
            "Run `python fetch_data.py` or use the committed data/sample.csv."
        )
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - re-raise with actionable context
        raise ValueError(f"Could not parse {path} as CSV: {exc}") from exc


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separate features from the ``Survived`` target.

    Args:
        df: Raw DataFrame, with or without the target column.

    Returns:
        A ``(X, y)`` tuple. ``y`` is ``None`` when the target is absent
        (e.g. inference-only input).
    """
    if TARGET in df.columns:
        return df.drop(columns=[TARGET]), df[TARGET].astype(int)
    return df, None


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Return the output feature names of a *fitted* preprocessor.

    Useful for labelling coefficients / feature-importance plots in the app.

    Args:
        preprocessor: A fitted pipeline from :func:`build_preprocessor`.

    Returns:
        The list of output column names in matrix order.
    """
    return list(preprocessor.named_steps["encode"].get_feature_names_out())


def save_preprocessor(preprocessor: Pipeline, path: str | Path) -> Path:
    """Persist a fitted preprocessor to disk with joblib.

    Args:
        preprocessor: The fitted pipeline to save.
        path: Destination ``.joblib`` path; parent dirs are created.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    return path


def load_preprocessor(path: str | Path) -> Pipeline:
    """Load a fitted preprocessor previously saved with :func:`save_preprocessor`.

    Args:
        path: Path to the ``.joblib`` file.

    Returns:
        The deserialised pipeline.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {path}. Train the model first (train.py)."
        )
    return joblib.load(path)


if __name__ == "__main__":
    # Smoke test (Brick 3 verification): fit on available data and report the
    # resulting feature matrix shape + names. Prefers the full train.csv, falls
    # back to the committed sample so it runs without Kaggle credentials.
    data_path = Path("data/train.csv")
    if not data_path.exists():
        data_path = Path("data/sample.csv")

    raw = load_raw(data_path)
    X, y = split_xy(raw)

    pre = build_preprocessor()
    matrix = pre.fit_transform(X, y)
    names = get_feature_names(pre)

    print(f"[ok] Loaded {len(raw)} rows from {data_path}")
    print(f"[ok] Feature matrix: {matrix.shape} (rows x features)")
    print(f"[ok] {len(names)} output features:")
    for name in names:
        print(f"       - {name}")
    if y is not None:
        print(f"[ok] Target present: {int(y.sum())}/{len(y)} survived")
