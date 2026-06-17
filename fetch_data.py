"""Download ``train.csv`` from the Kaggle Titanic competition.

Idempotent — skips the download if ``data/train.csv`` already exists.

Kaggle credentials are resolved in this order:

1. A ``.env`` file in the project root (``KAGGLE_USERNAME`` + ``KAGGLE_KEY``,
   or ``KAGGLE_API_TOKEN``).  Copy ``.env.example`` and fill it in.
2. Environment variables already set in the shell.
3. ``~/.kaggle/kaggle.json`` (the classic Kaggle credential file).

Auth also requires accepting the competition rules at
https://www.kaggle.com/competitions/titanic/rules (otherwise the API returns 403).

Usage::

    python fetch_data.py
    python fetch_data.py --out-dir data
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _load_dotenv(root: Path) -> None:
    """Load ``.env`` from *root* if python-dotenv is available.

    Args:
        root: Directory to look for a ``.env`` file in.
    """
    env_path = root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import]
        load_dotenv(env_path, override=False)  # shell vars take priority
        print(f"[env] loaded credentials from {env_path}")
    except ImportError:
        # dotenv not installed — env vars / kaggle.json still work fine
        print("[env] python-dotenv not installed; .env file ignored.")


def fetch(out_dir: str = "data") -> Path:
    """Download ``train.csv`` into *out_dir*, skipping if already present.

    Args:
        out_dir: Directory to place ``train.csv`` in (created if absent).

    Returns:
        Path to the downloaded (or pre-existing) ``train.csv``.

    Raises:
        SystemExit: On Kaggle API errors (403 = rules not accepted, 401 = bad creds).
    """
    project_root = Path(__file__).resolve().parent
    _load_dotenv(project_root)

    dest = Path(out_dir) / "train.csv"
    if dest.exists():
        print(f"[skip] {dest} already exists — delete it to re-download.")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: PLC0415 — import after env is populated
    except ImportError as exc:
        raise SystemExit(
            "kaggle package not found. Run: pip install kaggle"
        ) from exc

    print("[fetch] downloading train.csv from Kaggle competition 'titanic' …")
    try:
        kaggle.api.authenticate()
        kaggle.api.competition_download_file(
            competition="titanic",
            file_name="train.csv",
            path=str(dest.parent),
            force=False,
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "403" in msg:
            raise SystemExit(
                "Kaggle returned 403 — have you accepted the competition rules at "
                "https://www.kaggle.com/competitions/titanic/rules ?"
            ) from exc
        if "401" in msg or "credential" in msg.lower():
            raise SystemExit(
                "Kaggle credentials missing or invalid.\n"
                "  Option A: create .env with KAGGLE_USERNAME and KAGGLE_KEY\n"
                "  Option B: place kaggle.json in ~/.kaggle/\n"
                "  Option C: set KAGGLE_USERNAME / KAGGLE_KEY env vars"
            ) from exc
        raise SystemExit(f"Kaggle download failed: {exc}") from exc

    # The kaggle client may append .zip if it downloaded a zip archive.
    zipped = dest.with_suffix(".csv.zip")
    if zipped.exists() and not dest.exists():
        import zipfile
        with zipfile.ZipFile(zipped) as zf:
            zf.extract("train.csv", dest.parent)
        zipped.unlink()

    if not dest.exists():
        raise SystemExit(
            f"Download appeared to succeed but {dest} was not created. "
            "Check the kaggle client output above."
        )

    print(f"[ok] saved {dest} ({dest.stat().st_size:,} bytes)")
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download Titanic train.csv from Kaggle.")
    p.add_argument("--out-dir", default="data", help="Directory to save train.csv in.")
    args = p.parse_args()
    fetch(out_dir=args.out_dir)
