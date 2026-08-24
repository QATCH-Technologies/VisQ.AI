"""
charge.py
=========
Net protein charge feature: exactly one raw physical measurement --
Whole_Antibody_Charge_at_Buffer_pH (some CSVs label it "Whole_Charge") --
exposed to the model as `whole_charge`.

Every derived charge feature that previously lived here (net_charge,
abs_charge, near_pI and its concentration interactions, theo_pI, pI_gap,
charge_screened, the legacy-Charge/Fv-charge-source join machinery, the
missingness-imputation path) has been removed: the final model keeps only
Whole Charge, everything else the earlier ablation work explored is gone
from both the model and the repository.
"""

from __future__ import annotations

import pandas as pd

# Raw CSV headers recognized as the whole-antibody net charge at formulation
# pH, in priority order.
WHOLE_CHARGE_RAW_COLS = ("Whole_Antibody_Charge_at_Buffer_pH", "Whole_Charge")

CHARGE_FEATURE_COLS = ["whole_charge"]


def normalize_charge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever raw whole-charge header is present to the internal
    name `featurize_charge` expects. Idempotent and safe if absent (older
    CSVs / rows with no charge data)."""
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    for cand in WHOLE_CHARGE_RAW_COLS:
        if cand in df.columns:
            df = df.rename(columns={cand: "_raw_whole_charge"})
            break
    return df


def featurize_charge(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Append `whole_charge`. 0.0 where absent (no protein present, or an
    older CSV that never measured it) -- the same "missing degrades to the
    neutral default" convention every other raw column in build_feature_frame
    already follows.

    Returns (df_out, cols) where `cols` are the new numeric columns to add to
    the StandardScaler group.
    """
    df = df.copy()
    if "_raw_whole_charge" in df.columns:
        whole_charge = pd.to_numeric(df["_raw_whole_charge"], errors="coerce").fillna(0.0)
    else:
        whole_charge = pd.Series(0.0, index=df.index)
    df["whole_charge"] = whole_charge.values
    return df, list(CHARGE_FEATURE_COLS)
