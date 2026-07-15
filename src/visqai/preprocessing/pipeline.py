"""
pipeline.py
===========
The single row-level feature-engineering pipeline shared by training
(visqai.training.data.load_and_preprocess) and inference
(visqai.inference.predictor.ViscosityPredictorCNP._preprocess).

Previously this logic existed twice, independently: once inline in
train_o_net_v4_rung1.py::load_and_preprocess, once inline in
inference_o_net.py::ViscosityPredictorCNP._preprocess. Diffing the two
surfaced two real divergences, both resolved here in favor of the trainer's
(more current, charge-aware) version:

1. THE BUG: inference's version never called charge.normalize_charge_columns /
   charge.featurize_charge at all, so `net_charge` and friends were always
   absent -> silently zero-filled by the generic "missing expected feature"
   fallback, AND the CCI/regime lookup used the stale |pH-PI_mean| proxy
   instead of the real net-charge formula. Fixed here: charge featurization
   runs before the physics-priors step (priors.calculate_row_priors reads
   `net_charge` off the row).
2. proline molecular weight: train used 115.13, inference used 115.1 (a
   discrepancy inference's own comment flagged but never resolved). The
   fitted preprocessor was trained against 115.13, so that is the correct
   value to standardize on.

Also NOT carried forward: inference's "NEW RHEOLOGICAL PHYSICS FEATURES"
block (Phi_Protein, Phi_Stabilizer, KD_Asymptote, Exp_Crowding,
Ionic_Strength_Proxy). These were computed at inference time but never added
to `num_cols` at train time, so the fitted ColumnTransformer.transform()
silently ignored them -- they were dead computation with zero effect on any
prediction. Dropped rather than ported, since porting dead code forward just
carries the confusion into the new package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from visqai.features.categorical import (
    CHEM_CATEGORICALS,
    featurize_chemical_categoricals,
    describe_property_space,
)
from visqai.features.charge import (
    normalize_charge_columns,
    featurize_charge,
    CHARGE_FEATURE_COLS_BASE,
    ADD_SCREENED_CHARGE,
)
from visqai.physics.priors import calculate_row_priors, PRIOR_COLS, CONC_SPLIT_COLS

# Base numeric columns expected on the raw input (matches the fitted
# preprocessor's ColumnTransformer num_cols, before engineered/prior columns).
BASE_NUMERIC_COLS = [
    "kP",
    "MW",
    "PI_mean",
    "PI_range",
    "Protein_conc",
    "Temperature",
    "Buffer_pH",
    "Buffer_conc",
    "Salt_conc",
    "Stabilizer_conc",
    "Surfactant_conc",
    "Excipient_conc",
    "C_Class",
    "HCI",
]

# Only Protein_type stays one-hot; the six chemical categoricals are
# featurized into physicochemical property vectors instead (Rung-1).
BASE_CATEGORICAL_COLS = ["Protein_type"]

ENGINEERED_COLS = [
    "log_conc",
    "conc_sq",
    "conc_x_kP",
    "conc_x_HCI",
    "Crowding_Index",
    "Stabilizer_Squared",
    "Total_Solute_Mass",
    "Effective_Protein_Fraction",
]

SHEAR_MAP = {
    "Viscosity_100": 100.0,
    "Viscosity_1000": 1000.0,
    "Viscosity_10000": 10000.0,
    "Viscosity_100000": 100000.0,
    "Viscosity_15000000": 1.5e7,
}

MW_MAP = {
    "sucrose": 342.3,
    "trehalose": 342.3,
    "arginine": 174.2,
    "proline": 115.13,
    "lysine": 149.19,
    "nacl": 58.44,
    "default_sugar": 342.3,
}


def _get_mw(chemical_series: pd.Series, default_mw: float = 342.3) -> pd.Series:
    return (
        chemical_series.astype(str)
        .str.lower()
        .map(lambda x: next((mw for name, mw in MW_MAP.items() if name in x), default_mw))
    )


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Run the full row-level feature-engineering pipeline: numeric defaults,
    chemical-categorical featurization (Rung-1), charge featurization
    (Rung-2), unit normalization, engineered physics columns, and per-row
    prior/regime features.

    Returns
    -------
    (df_out, num_cols, cat_cols)
        df_out   : df with every engineered column appended.
        num_cols : full ordered list of numeric column names to feed a
                   StandardScaler / ColumnTransformer "num" transformer.
        cat_cols : categorical column names for the "cat" (one-hot) transformer.
    """
    df = df.copy()

    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].apply(lambda x: x.value if hasattr(x, "value") else x)

    num_cols = list(BASE_NUMERIC_COLS)
    cat_cols = list(BASE_CATEGORICAL_COLS)
    chem_cat_cols = list(CHEM_CATEGORICALS)

    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0.0)

    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().replace("nan", "unknown")
        else:
            df[c] = "unknown"

    # --- Rung-1: chemical categoricals -> physicochemical property vectors ---
    for c in chem_cat_cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .fillna("none")
                .astype(str)
                .str.lower()
                .replace("nan", "none")
                .replace("unknown", "none")
            )
        else:
            df[c] = "none"
    df, prop_cols = featurize_chemical_categoricals(df)

    # --- Rung-2: charge/pI -> feature block (THE FIX: this must run before
    # the physics-priors step below, since calculate_row_priors reads
    # net_charge off each row) ---
    df = normalize_charge_columns(df)
    df, charge_cols = featurize_charge(df)

    # --- Unit normalization to mg/mL ---
    if "Stabilizer_type" in df.columns:
        stabilizer_mw = _get_mw(df["Stabilizer_type"], default_mw=342.3)
    else:
        stabilizer_mw = 342.3
    df["Stabilizer_mg_mL"] = df["Stabilizer_conc"] * stabilizer_mw

    if "Salt_type" in df.columns:
        salt_mw = _get_mw(df["Salt_type"], default_mw=58.44)
    else:
        salt_mw = 58.44
    df["Salt_mg_mL"] = (df["Salt_conc"] * salt_mw) / 1000.0

    if "Excipient_type" in df.columns:
        excipient_mw = _get_mw(df["Excipient_type"], default_mw=150.0)
    else:
        excipient_mw = 150.0
    df["Excipient_mg_mL"] = (df["Excipient_conc"] * excipient_mw) / 1000.0

    df["Surfactant_mg_mL"] = df["Surfactant_conc"] * 10.0

    # --- Engineered physics columns ---
    df["log_conc"] = np.log1p(df["Protein_conc"])
    df["conc_sq"] = df["Protein_conc"] ** 2
    df["conc_x_kP"] = df["Protein_conc"] * df["kP"]
    df["conc_x_HCI"] = df["Protein_conc"] * df["HCI"]

    df["Crowding_Index"] = df["Protein_conc"] * df["Stabilizer_mg_mL"]
    df["Stabilizer_Squared"] = df["Stabilizer_mg_mL"] ** 2

    df["Total_Solute_Mass"] = (
        df["Protein_conc"] + df["Stabilizer_mg_mL"] + df["Excipient_mg_mL"] + df["Salt_mg_mL"] + df["Surfactant_mg_mL"]
    )
    df["Effective_Protein_Fraction"] = df["Protein_conc"] / df["Total_Solute_Mass"].replace(0, 1e-6)

    # --- Priors and regime (charge-aware; net_charge is available now) ---
    features_df = df.apply(calculate_row_priors, axis=1, result_type="expand")
    df = pd.concat([df, features_df], axis=1)

    num_cols = (
        num_cols
        + PRIOR_COLS
        + CONC_SPLIT_COLS
        + ENGINEERED_COLS
        + prop_cols
        + charge_cols
    )

    return df, num_cols, cat_cols


def _all_property_columns() -> set[str]:
    tidy = describe_property_space()
    return set(tidy.columns) - {"categorical", "category"}


def _all_charge_columns() -> set[str]:
    cols = set(CHARGE_FEATURE_COLS_BASE)
    if ADD_SCREENED_CHARGE:
        cols.add("charge_screened")
    return cols


def protected_feature_indices(num_cols: list[str]) -> list[int]:
    """Indices (into num_cols) of features that must never be masked out of
    the decoder query during training: concentration/pH/identity (the
    strongest viscosity drivers), plus every continuous physicochemical
    property and charge column (masking those to zero is physically
    incoherent -- zero is a specific, meaningful physical state for these,
    not a neutral null)."""
    protected_names = {
        "Protein_conc",
        "log_conc",
        "conc_sq",
        "conc_x_kP",
        "conc_x_HCI",
        "Buffer_pH",
        "MW",
        "PI_mean",
        "Temperature",
    }
    protected_names |= _all_property_columns()
    protected_names |= _all_charge_columns()
    return [i for i, c in enumerate(num_cols) if c in protected_names]
