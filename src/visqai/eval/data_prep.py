"""
data_prep.py
============
prepare_df: int->float coercion + ID->str, with an optional bad-row filter.

Unified from two near-duplicates: ibal_parity_test.py's prepare_df(df) and
learning_curve_ibal.py's prepare_df(df, drop_bad_rows=False) (a strict
superset -- the default reproduces ibal_parity_test.py's exact behavior).
"""

from __future__ import annotations

import pandas as pd

from visqai.eval.constants import VISC_COLS


def prepare_df(df: pd.DataFrame, drop_bad_rows: bool = False) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        if col != "ID":
            df[col] = df[col].astype(float)
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)
    if drop_bad_rows:
        visc_mask = pd.Series(True, index=df.index)
        for vc in VISC_COLS:
            if vc in df.columns:
                visc_mask &= df[vc].notna() & (df[vc] > 0)
        crit = [c for c in ["MW", "Protein_conc", "kP"] if c in df.columns]
        num_mask = df[crit].notna().all(axis=1) if crit else pd.Series(True, index=df.index)
        df = df[visc_mask & num_mask].reset_index(drop=True)
    return df
