"""Shared input-validation helpers for VisQAI's public API.

This module centralizes common boundary checks used across the features,
training, inference, packaging, and evaluation layers. The helpers validate
types, required DataFrame columns, paths, numeric constraints, allowed values,
and non-empty inputs before invalid arguments reach deeper implementation
layers where they could otherwise produce indirect or misleading errors.

The validation functions are intentionally small and side-effect free. They
either return `None` after confirming that an input satisfies the requested
constraint or, where useful for path validation, return the normalized
:class:`pathlib.Path` being validated.

Validation is primarily intended for public API and workflow boundaries.
Frequently executed internal paths such as training loops and model forward
methods deliberately omit these checks because their inputs have already been
validated upstream and repeating the checks would add unnecessary per-call
overhead.

The module does not attempt to validate domain-specific semantics that belong
to the consuming subsystem; callers remain responsible for checks specific to
their own algorithms or data contracts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def require_dataframe(df, name: str = "df") -> None:
    """Require that a value is a pandas DataFrame.

    Args:
        df: Value to validate.
        name: Human-readable name used in the error message.

    Raises:
        TypeError: If `df` is not a :class:`pandas.DataFrame`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")


def require_nonempty_dataframe(df, name: str = "df") -> None:
    """Validate that an object is a non-empty pandas DataFrame.

    Args:
        df: Object to validate.
        name: Human-readable name used in error messages.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If `df` is an empty DataFrame.
    """
    require_dataframe(df, name)
    if df.empty:
        raise ValueError(f"{name} is empty")


def require_columns(df, columns, name: str = "df") -> None:
    """Validate that a DataFrame contains all required columns.

    Args:
        df: DataFrame to validate.
        columns: Iterable of column names that must be present.
        name: Human-readable name used in error messages.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If one or more required columns are missing.
    """
    require_dataframe(df, name)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required column(s): {missing}")


def require_path_exists(path, name: str = "path", kind: str | None = None) -> Path:
    """Validate that a path exists and optionally has the requested type.

    Args:
        path: Path-like value to validate.
        name: Human-readable name used in error messages.
        kind: Optional path type constraint. Must be `"file"`, `"dir"`,
            or `None` to validate existence without restricting the type.

    Returns:
        The validated path as a :class:`pathlib.Path`.

    Raises:
        ValueError: If `path` is missing, empty, or does not match the
            requested `kind`.
        FileNotFoundError: If `path` does not exist.
    """
    if path is None or (isinstance(path, str) and not path):
        raise ValueError(f"{name} must be given")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")
    if kind == "file" and not p.is_file():
        raise ValueError(f"{name} is not a file: {p}")
    if kind == "dir" and not p.is_dir():
        raise ValueError(f"{name} is not a directory: {p}")
    return p


def require_positive(value, name: str = "value") -> None:
    """Validate that a value is a strictly positive numeric value.

    Args:
        value: Value to validate.
        name: Human-readable name used in the error message.

    Raises:
        ValueError: If `value` is a boolean, is not an integer or floating-
            point number, or is less than or equal to zero.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")


def require_positive_int(value, name: str = "value") -> None:
    """Validate that a value is a strictly positive integer.

    Args:
        value: Value to validate.
        name: Human-readable name used in the error message.

    Raises:
        ValueError: If `value` is a boolean, is not an integer, or is less
            than or equal to zero.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def require_non_negative(value, name: str = "value") -> None:
    """Validate that a value is a non-negative numeric value.

    Args:
        value: Value to validate.
        name: Human-readable name used in the error message.

    Raises:
        ValueError: If `value` is a boolean, is not an integer or floating-
            point number, or is less than zero.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")


def require_in(value, choices, name: str = "value") -> None:
    """Validate that a value is contained in an allowed set of choices.

    Args:
        value: Value to validate.
        choices: Iterable of allowed values.
        name: Human-readable name used in the error message.

    Raises:
        ValueError: If `value` is not present in `choices`.
    """
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")


def require_type(value, types, name: str = "value") -> None:
    """Validate that a value is an instance of one or more expected types.

    Args:
        value: Value to validate.
        types: Expected type or tuple of accepted types.
        name: Human-readable name used in the error message.

    Raises:
        TypeError: If `value` is not an instance of any of the specified
            `types`.
    """
    if not isinstance(value, types):
        expected = (
            types.__name__ if isinstance(types, type) else " or ".join(t.__name__ for t in types)
        )
        raise TypeError(f"{name} must be {expected}, got {type(value).__name__}")


def require_nonempty(seq, name: str = "value") -> None:
    """Validate that a sequence-like value is not empty.

    Args:
        seq: Sequence or collection to validate.
        name: Human-readable name used in the error message.

    Raises:
        ValueError: If `seq` is `None` or has zero length.
    """
    if seq is None or len(seq) == 0:
        raise ValueError(f"{name} must not be empty")
