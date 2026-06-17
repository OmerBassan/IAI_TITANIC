# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is an **early-stage** Titanic survival-prediction project. Most of it is not yet built. The only working code is `fetch_data.py`; everything else described below is the planned design from `workflow.md`. Treat `workflow.md` (written in Hebrew) as the authoritative project plan and check it before adding new components.

The graded goal emphasizes **creativity and use of AI**, not just a working model — design choices should reflect that (see "Architecture intent" below).

## Commands

```bash
pip install -r requirements.txt   # full stack: torch, pandas, scikit-learn, kaggle, streamlit, jupyter
python fetch_data.py              # downloads data/train.csv from Kaggle (idempotent — skips if present)
```

There is no test suite, linter, or `train.py`/`ds_app.py` entry point yet (those are later bricks).

## Data

`fetch_data.py` uses the **Kaggle API** to download **only `train.csv`** from the `titanic` competition into `data/train.csv` (per the assignment — `test.csv`/`gender_submission.csv` are intentionally not used; the train/val split is created from `train.csv` alone). Auth requires either `~/.kaggle/kaggle.json`, `KAGGLE_USERNAME`/`KAGGLE_KEY`, or a `KAGGLE_API_TOKEN` (the `KGAT_…` token format works), **and** accepting the competition rules at https://www.kaggle.com/competitions/titanic/rules (otherwise the API returns 403). The fetch is idempotent and `data/` is created on demand. `data/sample.csv` is a committed 40-row balanced sample so the repo runs without credentials; full CSVs are gitignored.

## Architecture intent (planned, per workflow.md)

The plan calls for a clean separation of concerns once built:

- **Model definition** lives separately from the **training script** (`train.py`). The plan favors a non-trivial tabular architecture (e.g. ResNet-style tabular net with skip connections, or simple tabular attention) over a plain MLP, to demonstrate PyTorch depth.
- **Training loop** should include early stopping, automatic best-checkpoint saving (selected on the validation set), and per-epoch loss/metric logging.
- **Preprocessing pipeline** (missing-value handling, categorical encoding, normalization) built in scikit-learn or PyTorch.
- **Evaluation** reports Accuracy, Precision, Recall, F1, and ROC-AUC, plus learning-curve and ROC plots.
- **Streamlit UI** (`ds_app.py`) loads the trained checkpoint and offers interactive prediction, ideally with feature-importance / prediction explanation and optional CSV batch upload.

When implementing any of these, follow the file names and responsibilities above so the structure stays consistent with the plan.

## Conventions

The plan specifies full type hints and Google-style docstrings on all functions/classes, and wrapping I/O-sensitive operations (model loading, file reads, UI input) in try/except so the app never crashes on bad input. Existing code (`fetch_data.py`) already mixes Hebrew comments with English output strings — this is fine; match the surrounding style of whatever file you edit.  
  
  
  
  
The plan (brick by brick)

  Brick 0 — Foundation & hygiene

  Project layout decision, requirements.txt, .gitignore (ignore data/*.csv except a small sample, **pycache**, venv, checkpoints). Goal: clean pip

  install in a fresh venv. Verify: env installs cleanly.

  Brick 1 — Reproducible data fetch

  Rework fetch_[data.py](http://data.py) to satisfy "from Kaggle, train.csv only." Commit a small sample CSV to data/ for the repo. Verify: fetch yields data/train.csv;

  sample committed.

  Brick 2 — EDA notebook (Jupyter)

  Distributions, missingness, correlations, survival rates by feature, with written insights inline (depth + clarity is a graded criterion). Verify:   

  notebook runs top-to-bottom.

  Brick 3 — Preprocessing module (shared)

  One reusable module used by both [train.py](http://train.py) and ds_[app.py](http://app.py) so train-time and inference-time transforms are identical. Missing-value handling,

  categorical encoding, scaling, feature engineering (Title, FamilySize, etc.). Fitted preprocessor persisted to disk alongside weights. Verify:       

  consistent feature matrix.

  Brick 4 — PyTorch model definition

  [model.py](http://model.py), separate from training. Verify: forward pass on a dummy batch.

  Brick 5 — [train.py](http://train.py)

  Load → train/val split → fit preprocessing on train → training loop with early stopping + seed → save best checkpoint + preprocessor + metadata.     

  Per-epoch logging. Verify: produces artifacts + prints val metrics.

  Brick 6 — Evaluation module (shared)

  Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curve, learning curves. Reused by Streamlit. Verify: metrics + plots generate.       

  Brick 7 — ds_[app.py](http://app.py) (Streamlit)

  Load model+preprocessor from disk, user supplies CSV path, run inference, show metrics+plots (when labels present). Creativity hooks: feature        

  importance / per-prediction explanation, batch prediction download. Verify: app launches and runs on sample CSV.

  Brick 8 — Robustness & polish

  try/except on I/O & UI inputs, full type hints, Google-style docstrings, remove dead code, fixed seeds.

  Brick 9 — README + reproducibility check + screenshots

  Fresh-venv run-through, README (setup/install/run both apps/screenshots/design writeup).