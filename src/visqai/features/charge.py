"""Provide the model's whole-antibody net-charge feature.

This module exposes a single raw physical charge measurement,
`Whole_Antibody_Charge_at_Buffer_pH`, under the internal feature name
`whole_charge`. The legacy derived charge features and charge-source
joining or imputation machinery are intentionally excluded from the final
feature representation.

The module recognizes multiple raw CSV header variants for the same physical
measurement and normalizes them to an internal column before feature
generation. Missing or unavailable charge measurements are represented by a
neutral default of `0.0`, consistent with the missing-value convention used
by the broader feature-building pipeline.

Attributes:
    WHOLE_CHARGE_RAW_COLS: Raw CSV column names recognized as whole-antibody
        net charge, ordered by lookup priority.
    CHARGE_FEATURE_COLS: Names of the numeric charge features exposed to the
        model.

Functions:
    normalize_charge_columns: Normalize recognized raw charge column names.
    featurize_charge: Generate the model-ready whole-charge feature.
"""

from __future__ import annotations

import pandas as pd

from visqai.validation import require_dataframe

WHOLE_CHARGE_RAW_COLS = ("Whole_Antibody_Charge_at_Buffer_pH", "Whole_Charge")
CHARGE_FEATURE_COLS = ["whole_charge"]


def normalize_charge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ "Normalize the raw whole-charge column name.

    The first recognized raw charge column in
    :data:`WHOLE_CHARGE_RAW_COLS` is renamed to `"_raw_whole_charge"`.
    Columns whose names begin with `"Unnamed"` are removed as common
    spreadsheet-index artifacts.

    The operation is performed on a copy of the input DataFrame and is
    idempotent with respect to the normalized column name. If no recognized
    raw charge column is present, the DataFrame is returned without adding
    one.

    Args:
        df: Input DataFrame containing zero or more recognized raw whole-charge
            columns.

    Returns:
        pd.DataFrame: Copy of `df` with the recognized raw charge column
        normalized to `"_raw_whole_charge"` when present.

    Raises:
        TypeError: If `df` does not satisfy the DataFrame validation
            performed by :func:`visqai.validation.require_dataframe`.
    """
    require_dataframe(df, "df")
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    for cand in WHOLE_CHARGE_RAW_COLS:
        if cand in df.columns:
            df = df.rename(columns={cand: "_raw_whole_charge"})
            break
    return df


def featurize_charge(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Append the whole-antibody net-charge model feature.

    The normalized `"_raw_whole_charge"` column is converted to numeric
    values and stored as `"whole_charge"`. Values that cannot be converted
    to numeric values, as well as missing charge data when the raw column is
    absent, are represented as `0.0`.

    The returned feature-name list identifies the numeric column that should
    be included in the downstream standardized numeric feature group.

    Args:
        df: Input DataFrame, optionally containing the normalized
            `"_raw_whole_charge"` column.

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple containing:

            - A copy of `df` with the `"whole_charge"` feature appended.
            - A list containing the generated numeric feature name,
              `"whole_charge"`.

    Raises:
        TypeError: If `df` does not satisfy the DataFrame validation
            performed by :func:`visqai.validation.require_dataframe`.
    """
    require_dataframe(df, "df")
    df = df.copy()
    if "_raw_whole_charge" in df.columns:
        whole_charge = pd.to_numeric(df["_raw_whole_charge"], errors="coerce").fillna(0.0)
    else:
        whole_charge = pd.Series(0.0, index=df.index)
    df["whole_charge"] = whole_charge.values
    return df, list(CHARGE_FEATURE_COLS)
