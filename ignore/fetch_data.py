"""Fetch the Titanic ``train.csv`` from Kaggle into the local ``data/`` directory.

Only ``train.csv`` from the Kaggle *Titanic* competition is downloaded, per the
assignment. ``test.csv`` and ``gender_submission.csv`` are intentionally ignored;
the train/validation split is created later from ``train.csv`` alone.

Authentication:
    The Kaggle API needs credentials. Provide them in one of two ways:
      1. Place ``kaggle.json`` at ``~/.kaggle/kaggle.json`` (Kaggle > Account >
         "Create New API Token"), or
      2. Set the ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` environment variables.

    You must also accept the competition rules once at
    https://www.kaggle.com/competitions/titanic/rules — otherwise the API
    returns a 403 even with valid credentials.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

COMPETITION = "titanic"
TARGET_FILE = "train.csv"
DATA_DIR = Path("data")


def fetch_dataset(data_dir: Path = DATA_DIR) -> Path:
    """Download ``train.csv`` from the Kaggle Titanic competition.

    Args:
        data_dir: Directory the CSV is written to. Created if missing.

    Returns:
        Path to the downloaded ``train.csv``.

    Raises:
        SystemExit: If the Kaggle package is missing, credentials are not
            configured, or the download fails. The message explains the fix.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    target_path = data_dir / TARGET_FILE

    if target_path.exists():
        print(f"[skip] Dataset already present: {target_path}")
        return target_path

    # Imported lazily so a missing dependency yields a clear message instead of
    # an ImportError at module load.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit(
            "The 'kaggle' package is not installed.\n"
            "Install dependencies first: pip install -r requirements.txt"
        )

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - surface any auth failure plainly
        sys.exit(
            f"Kaggle authentication failed: {exc}\n"
            "Set up credentials via ~/.kaggle/kaggle.json or the "
            "KAGGLE_USERNAME / KAGGLE_KEY environment variables."
        )

    print(f"[download] {TARGET_FILE} from Kaggle competition '{COMPETITION}'...")
    try:
        api.competition_download_file(
            COMPETITION, TARGET_FILE, path=str(data_dir), quiet=False
        )
    except Exception as exc:  # noqa: BLE001 - network / 403 / API errors
        sys.exit(
            f"Download failed: {exc}\n"
            "Make sure you have accepted the competition rules at\n"
            "https://www.kaggle.com/competitions/titanic/rules"
        )

    # Kaggle sometimes delivers the file as a .zip; unpack if so.
    zipped = data_dir / f"{TARGET_FILE}.zip"
    if zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            zf.extract(TARGET_FILE, path=data_dir)
        zipped.unlink()

    if not target_path.exists():
        sys.exit(
            f"Expected {target_path} after download but it is missing. "
            "Inspect the contents of the data/ directory."
        )

    print(f"[done] Saved dataset to: {target_path}")
    return target_path


if __name__ == "__main__":
    print("Starting Kaggle data fetch...")
    fetch_dataset()
