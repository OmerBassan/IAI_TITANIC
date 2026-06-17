"""Streamlit inference UI for the Titanic survival-prediction model.

Loads the trained checkpoint + fitted preprocessor from ``artifacts/`` (produced
by ``train.py``) and offers three views, matching the assignment requirements:

* **CSV inference** — the user gives a path to a dataset; the app runs batch
  inference and, when the file carries the ``Survived`` label, shows the full
  evaluation (metrics + confusion matrix + ROC curve) and a downloadable
  predictions CSV.
* **Validation report** — reproduces the exact held-out validation split that
  ``train.py`` used (same seed / val_size from metadata) and presents its
  metrics, plots and the training learning curves.
* **Single passenger** — an interactive form with per-prediction explainability
  via the model's feature-attention gate.

All metrics and plots come from the shared ``src.evaluation`` module, so the
numbers shown here are computed by the exact same code that produced the training
report. Every disk read / inference call is wrapped in try/except so bad input
surfaces as an ``st.error`` instead of crashing the app.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.model_selection import train_test_split

from src.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_roc_curve,
)
from src.model import TabularResNet
from src.preprocessing import load_raw, split_xy

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

# Flat artifacts under artifacts/ (written by `python train.py`).
ARTIFACT_DIR = Path("artifacts")


@st.cache_resource
def load_artifacts() -> tuple[object, TabularResNet, dict]:
    """Load the fitted preprocessor, trained model and run metadata from disk.

    Returns:
        A ``(preprocessor, model, metadata)`` tuple. The model is in eval mode.

    Raises:
        FileNotFoundError: If an expected artifact is missing.
    """
    preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.joblib")
    ckpt = torch.load(ARTIFACT_DIR / "model.pt", map_location="cpu")
    model = TabularResNet(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    return preprocessor, model, metadata


try:
    preprocessor, model, metadata = load_artifacts()
except Exception as exc:  # noqa: BLE001 - any load failure must not crash the page
    st.error(
        "Failed to load model artifacts. Ensure `artifacts/` contains model.pt, "
        f"preprocessor.joblib and metadata.json (run `python train.py`). Error: {exc}"
    )
    st.stop()

# Decision threshold tuned at train time (macro-F1), so app predictions match the
# training report. Falls back to 0.5 if an older metadata file lacks it.
THRESHOLD: float = float(
    metadata.get("val_metrics", {}).get("best_thresh", 0.5)
)
FEATURE_NAMES: list[str] = metadata.get("feature_names", [])


def run_inference(raw_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Transform raw passenger rows and return survival probabilities + labels.

    Args:
        raw_features: Raw Kaggle-schema rows with the ``Survived`` target already
            removed (see :func:`src.preprocessing.split_xy`).

    Returns:
        A ``(proba, preds)`` tuple of 1-D arrays: survival probabilities in
        ``[0, 1]`` and binary predictions at the tuned threshold.
    """
    x_mat = preprocessor.transform(raw_features)
    x_tensor = torch.tensor(np.asarray(x_mat), dtype=torch.float32)
    with torch.no_grad():
        proba = model.predict_proba(x_tensor).numpy().ravel()
    preds = (proba >= THRESHOLD).astype(int)
    return proba, preds


def show_evaluation(y_true: np.ndarray, proba: np.ndarray, preds: np.ndarray) -> None:
    """Render the metric tiles + confusion matrix + ROC curve for labelled data.

    Args:
        y_true: Ground-truth binary labels.
        proba: Predicted survival probabilities.
        preds: Binary predictions at the tuned threshold.
    """
    metrics = compute_metrics(y_true, proba, threshold=THRESHOLD)
    cols = st.columns(5)
    for col, key in zip(cols, ("accuracy", "precision", "recall", "f1", "roc_auc")):
        col.metric(key.replace("_", " ").title(), f"{metrics[key]:.3f}")

    plot_a, plot_b = st.columns(2)
    with plot_a:
        st.pyplot(plot_confusion_matrix(y_true, preds))
    with plot_b:
        st.pyplot(plot_roc_curve(y_true, proba))


st.title("🚢 Titanic Survival Predictor")
tab_csv, tab_val, tab_single = st.tabs(
    ["📁 CSV inference", "📊 Validation report", "👤 Single passenger"]
)

# ---------------------------------------------------------------------------
# Tab A — batch inference from a CSV path (assignment's core requirement)
# ---------------------------------------------------------------------------
with tab_csv:
    st.markdown(
        "Provide a path to a CSV with the Kaggle Titanic schema. The model runs "
        "inference on every row; if the file has a `Survived` column, the full "
        "evaluation (metrics + plots) is shown."
    )
    csv_path = st.text_input(
        "Dataset CSV path",
        value="data/train.csv",
        help="No Kaggle credentials? Use the committed data/sample.csv.",
    )
    if st.button("Run inference", key="run_csv"):
        try:
            raw = load_raw(csv_path)
            X, y = split_xy(raw)
            proba, preds = run_inference(X)
        except Exception as exc:  # noqa: BLE001 - surface bad path / schema cleanly
            st.error(f"Could not run inference on `{csv_path}`: {exc}")
        else:
            st.success(f"Ran inference on {len(raw)} rows.")
            out = pd.DataFrame(
                {
                    "PassengerId": raw.get("PassengerId", pd.RangeIndex(len(raw))),
                    "survival_proba": proba.round(4),
                    "prediction": preds,
                }
            )
            if y is not None:
                st.subheader("Evaluation on labelled data")
                show_evaluation(y.to_numpy(), proba, preds)
                out.insert(1, "actual", y.to_numpy())
            else:
                st.info("No `Survived` column — showing predictions only.")

            st.subheader("Predictions")
            st.dataframe(out, use_container_width=True, height=300)
            st.download_button(
                "Download predictions CSV",
                out.to_csv(index=False).encode("utf-8"),
                file_name="titanic_predictions.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Tab B — held-out validation report (reproduces train.py's split)
# ---------------------------------------------------------------------------
with tab_val:
    st.markdown(
        "Reproduces the exact held-out validation split from training "
        "(same seed and split fraction) and reports its metrics, plots and the "
        "training learning curves."
    )
    cfg = metadata.get("config", {})
    data_path = cfg.get("data_path") or "data/train.csv"
    val_size = float(cfg.get("val_size", 0.2))
    seed = int(cfg.get("seed", 42))

    try:
        raw = load_raw(data_path)
        X, y = split_xy(raw)
        if y is None:
            raise ValueError(f"{data_path} has no 'Survived' column.")
        # Same call as train.py — deterministic given seed, so this is the very
        # same validation split the saved metrics were computed on.
        _, X_val, _, y_val = train_test_split(
            X, y.to_numpy(), test_size=val_size, stratify=y, random_state=seed
        )
        proba, preds = run_inference(X_val)
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"Could not regenerate the validation split from `{data_path}` ({exc}). "
            "Showing the metrics saved at training time instead."
        )
        st.json(metadata.get("val_metrics", {}))
    else:
        st.subheader(f"Held-out validation set ({len(y_val)} passengers)")
        show_evaluation(y_val, proba, preds)

    # Learning curves from the saved per-epoch history.
    try:
        history = json.loads(
            (ARTIFACT_DIR / "history.json").read_text(encoding="utf-8")
        )
        st.subheader("Training learning curves")
        st.pyplot(plot_learning_curves(history))
    except Exception as exc:  # noqa: BLE001
        st.info(f"Learning curves unavailable: {exc}")

# ---------------------------------------------------------------------------
# Tab C — single-passenger interactive prediction + attention explainability
# ---------------------------------------------------------------------------
with tab_single:
    st.markdown("Enter passenger details to predict survival probability.")
    with st.form("passenger_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Personal Info")
            name = st.text_input("Full Name (with title)", "Braund, Mr. Owen Harris")
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
        with col2:
            st.subheader("Ticket Details")
            pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
            ticket = st.text_input("Ticket Number", "A/5 21171")
            fare = st.number_input("Fare (£)", min_value=0.0, value=7.25)
            cabin = st.text_input("Cabin (blank if unknown)", "")
            embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
        with col3:
            st.subheader("Family Onboard")
            sibsp = st.number_input("Siblings / Spouses", min_value=0, max_value=10, value=0)
            parch = st.number_input("Parents / Children", min_value=0, max_value=10, value=0)
        submitted = st.form_submit_button("Predict Survival")

    if submitted:
        try:
            row = pd.DataFrame([{
                "PassengerId": 999,
                "Pclass": pclass,
                "Name": name,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Ticket": ticket,
                "Fare": fare,
                "Cabin": cabin if cabin else float("nan"),
                "Embarked": embarked,
            }])
            x_tensor = torch.tensor(
                np.asarray(preprocessor.transform(row)), dtype=torch.float32
            )
            with torch.no_grad():
                proba = model.predict_proba(x_tensor).item()
                attention = model.attention_weights.squeeze().numpy()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")
        else:
            st.divider()
            col_res, col_exp = st.columns([1, 2])
            with col_res:
                if proba >= THRESHOLD:
                    st.success("### 🟢 Prediction: SURVIVED")
                else:
                    st.error("### 🔴 Prediction: DID NOT SURVIVE")
                st.info(f"Survival Probability: **{proba:.1%}**")
            with col_exp:
                st.subheader("🧠 Decision Explainability (Feature Attention)")
                st.markdown(
                    "How the model weighted each feature for *this* passenger:"
                )
                if FEATURE_NAMES and len(attention) == len(FEATURE_NAMES):
                    imp = pd.DataFrame(
                        {"Feature": FEATURE_NAMES, "Attention Weight": attention}
                    )
                    imp = imp[imp["Attention Weight"] > 0.01].sort_values(
                        "Attention Weight"
                    )
                    st.bar_chart(imp.set_index("Feature"), height=300)
                else:
                    st.info("Attention weights unavailable for this model.")
