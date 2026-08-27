"""Path-resolution behavior for VisQAI's shared data, checkpoint, and production
storage.

This module owns the runtime behavior for resolving dated checkpoint
directories and current data files, while the shared storage-root values
themselves are defined in :mod:`visqai.constants`. Keeping the values separate
from the resolution logic allows the roots to be centrally configured without
coupling unrelated consumers to this module.

Checkpoint and production artifacts use timestamp-based directory layouts
rather than caller-selected names. :func:`dated_run_dir` creates the canonical
`<root>/<YYYY-MM-DD>/<HH-MM-SS>` path for a new run, while
:func:`latest_checkpoint_dir` locates the most recently modified existing
checkpoint run.

Data loading is likewise centralized through :func:`latest_data_file` and
:func:`load_table`. The former resolves the current master export only when a
caller has not supplied an explicit path; the latter abstracts the CSV/Excel
format change so callers do not need to select a pandas reader themselves.

The functions in this module do not create directories or eagerly resolve
optional storage locations. This is intentional: callers operating with
explicit paths should remain usable on machines where the shared Dropbox
storage is unavailable, including CI and deployment environments.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from visqai.constants import DATA_LATEST_DIR


def dated_run_dir(root: str | Path) -> Path:
    """Return the canonical timestamped directory for a new run.

    Checkpoints and production packages are organized by creation time rather
    than by caller-selected descriptive names. This provides a consistent
    layout and avoids collisions, stale names, and duplicated naming
    conventions across training and packaging workflows.

    The returned directory is not created on disk. The caller is responsible
    for creating it when needed.

    Args:
        root: Root directory under which the date/time hierarchy should be
            constructed.

    Returns:
        Path: The run directory in the form
        `<root>/<YYYY-MM-DD>/<HH-MM-SS>`.
    """
    now = datetime.now()
    return Path(root) / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")


def latest_checkpoint_dir(root: str | Path) -> Path:
    """Return the most recently modified checkpoint run directory.

    Searches the dated checkpoint hierarchy produced by
    :func:`dated_run_dir`, first selecting the most recently modified
    `<YYYY-MM-DD>` directory and then the most recently modified
    `<HH-MM-SS>` directory within it.

    This function is intended for fallback resolution when a caller has not
    supplied an explicit model directory. It does not search recursively
    across all possible run directories; the two-level date/time structure is
    part of the expected storage layout.

    Args:
        root: Root directory containing dated checkpoint run directories.

    Returns:
        Path: The most recently modified `<date>/<time>` checkpoint
        directory.

    Raises:
        FileNotFoundError: If `root` does not exist, contains no dated
            directories, or the most recent dated directory contains no
            checkpoint run directories.
    """
    root = Path(root)
    date_dirs = [d for d in root.iterdir() if d.is_dir()] if root.exists() else []
    if not date_dirs:
        raise FileNotFoundError(f"No dated checkpoint directories found in {root}")
    latest_date = max(date_dirs, key=os.path.getmtime)
    time_dirs = [d for d in latest_date.iterdir() if d.is_dir()]
    if not time_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {latest_date}")
    return max(time_dirs, key=os.path.getmtime)


def latest_data_file(dir_: str | Path = DATA_LATEST_DIR) -> Path:
    """Return the most recently modified data file in a directory.

    The directory is expected to contain the current master data export.
    When multiple files are present, the file with the most recent
    modification time is selected, providing a deterministic fallback
    convention consistent with checkpoint discovery.

    This function is intended to be called only when the caller has not
    supplied an explicit data path. It is deliberately not used as an eager
    command-line argument default so that environments without the shared
    Dropbox storage remain usable when an explicit path is provided.

    Args:
        dir_: Directory containing the candidate data files. Defaults to
            :data:`visqai.constants.DATA_LATEST_DIR`.

    Returns:
        Path: The most recently modified file in `dir_`.

    Raises:
        FileNotFoundError: If `dir_` does not exist or contains no files.
    """
    dir_ = Path(dir_)
    if not dir_.exists():
        raise FileNotFoundError(f"Data directory not found: {dir_}")
    candidates = [p for p in dir_.iterdir() if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No files found in {dir_}")
    return max(candidates, key=os.path.getmtime)


def load_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a tabular data file using the appropriate pandas reader.

    Supports both Excel workbooks and CSV files so callers do not need to
    know which export format they have received. Excel files with `.xlsx`
    or `.xls` suffixes are read with :func:`pandas.read_excel`; all other
    suffixes are treated as CSV and read with :func:`pandas.read_csv`.

    Additional keyword arguments are forwarded unchanged to the selected
    pandas reader, allowing callers to specify options such as
    `index_col=False` without coupling themselves to the underlying file
    format.

    Args:
        path: Path to the tabular data file.
        **kwargs: Additional keyword arguments forwarded to the selected
            pandas reader.

    Returns:
        pd.DataFrame: The contents of the input data file.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the file cannot be parsed by the selected pandas
            reader.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, **kwargs)
    return pd.read_csv(path, **kwargs)
