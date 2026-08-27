"""Shared row-level feature-engineering pipeline for VisQAI model training and
inference.

This module is the single source of truth for transforming raw formulation
rows into the feature frame consumed by the fitted preprocessing and modeling
pipeline. It is used by both training and inference paths to ensure that
feature definitions, defaults, unit conversions, engineered physics features,
chemical-property encodings, charge handling, and row-level priors remain
identical between model fitting and prediction.

The pipeline includes:

* Normalization of enum-like object values.
* Defaulting and validation of the base numeric feature space.
* Normalization of categorical formulation fields.
* Rung-1 chemical-categorical featurization into physicochemical property
  vectors.
* Normalization and featurization of the raw whole-antibody charge
  measurement.
* Conversion of formulation concentrations into mass-per-volume features.
* Derived concentration, protein-property, crowding, and solute-fraction
  features.
* Row-level prior and formulation-regime features.
* Identification of features that must remain visible when decoder inputs are
  masked during training.
* Shared row-level cleanup for evaluation and data-preparation workflows.

The module deliberately does not contain the legacy Rung-2 derived charge
features or the former inference-only rheological feature block. The fitted
model uses the raw whole-charge measurement represented by
`visqai.features.charge.featurize_charge` and only features included in the
training preprocessor are retained.

The proline molecular weight is standardized at `115.13` to match the value
used when fitting the preprocessing pipeline. Salt mass concentration is
clipped at :data:`SALT_MG_ML_CAP` before downstream engineered features are
constructed, preventing unseen leave-one-salt-out values from producing
unbounded extrapolation in derived solute features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from visqai.constants import VISC_COLS
from visqai.validation import require_dataframe, require_type
from visqai.features.categorical import (
    CHEM_CATEGORICALS,
    featurize_chemical_categoricals,
    describe_property_space,
)
from visqai.features.charge import (
    normalize_charge_columns,
    featurize_charge,
    CHARGE_FEATURE_COLS,
)
from visqai.features.priors import (
    calculate_row_priors,
    PRIOR_COLS,
    CONC_SPLIT_COLS,
    CONC_THRESHOLDS,
    CONC_HIGH_FRAC_CAP,
)

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


BASE_CATEGORICAL_COLS = ["Protein_type"]

ENGINEERED_COLS = [
    "log_conc",
    "conc_sq",
    "conc_x_kP",
    "conc_sq_x_kP",
    "conc_x_HCI",
    "Crowding_Index",
    "Stabilizer_Squared",
    "Total_Solute_Mass",
    "Effective_Protein_Fraction",
]

MW_MAP = {
    "sucrose": 342.3,
    "trehalose": 342.3,
    "arginine": 174.2,
    "proline": 115.13,
    "lysine": 149.19,
    "nacl": 58.44,
    "default_sugar": 342.3,
}


# Generous physical cap on Salt_mg_mL based on the NaCl concentration
# threshold and the same headroom used for high-concentration features.
# Applied before derived solute features are calculated to limit extrapolation
# when salt is absent from training data but present in a held-out fold.
SALT_MG_ML_CAP: float = (CONC_THRESHOLDS["nacl"] * MW_MAP["nacl"] / 1000.0) * CONC_HIGH_FRAC_CAP


def _get_mw(chemical_series: pd.Series, default_mw: float = 342.3) -> pd.Series:
    """Map chemical names to molecular weights using substring matching.

    Chemical names are normalized to lowercase strings and matched against
    entries in :data:`MW_MAP`. The first matching molecular weight is used;
    values that do not match any known chemical receive `default_mw`.

    Args:
        chemical_series: Series containing chemical or formulation component
            names.
        default_mw: Molecular weight to use when no known chemical name is
            found. Defaults to `342.3`.

    Returns:
        A pandas Series of molecular weights aligned with
        `chemical_series.index`.
    """
    return (
        chemical_series.dropna()
        .astype(str)
        .str.lower()
        .map(lambda x: next((mw for name, mw in MW_MAP.items() if name in str(x)), default_mw))
    )


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the complete model-ready row-level feature frame.

    Applies the feature-engineering pipeline shared by training and inference.
    Missing base numeric columns are initialized to zero, categorical values
    are normalized, chemical categorical variables are converted to
    physicochemical property features, the whole-charge measurement is
    normalized and featurized, concentration units are standardized, and
    engineered physics and formulation-prior features are calculated.

    Charge featurization intentionally occurs before row-level priors are
    calculated because the prior calculation consumes the resulting
    `net_charge` value when determining charge-dependent formulation
    behavior. The returned feature lists therefore describe the complete
    numeric and categorical feature space expected by the fitted model.

    Args:
        df: Raw formulation DataFrame. The input is copied before any
            transformations are applied, so the caller's DataFrame is not
            modified in place.

    Returns:
        A tuple containing:

        * The transformed feature DataFrame.
        * The ordered list of numeric feature column names.
        * The ordered list of categorical feature column names.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If an upstream feature-engineering or validation operation
            rejects the supplied data.
    """
    require_dataframe(df, "df")
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

    df = normalize_charge_columns(df)
    df, charge_cols = featurize_charge(df)

    if "Stabilizer_type" in df.columns:
        stabilizer_mw = _get_mw(df["Stabilizer_type"], default_mw=342.3)
    else:
        stabilizer_mw = 342.3
    df["Stabilizer_mg_mL"] = df["Stabilizer_conc"] * stabilizer_mw

    if "Salt_type" in df.columns:
        salt_mw = _get_mw(df["Salt_type"], default_mw=58.44)
    else:
        salt_mw = 58.44
    df["Salt_mg_mL"] = np.clip((df["Salt_conc"] * salt_mw) / 1000.0, 0.0, SALT_MG_ML_CAP)

    if "Excipient_type" in df.columns:
        excipient_mw = _get_mw(df["Excipient_type"], default_mw=150.0)
    else:
        excipient_mw = 150.0
    df["Excipient_mg_mL"] = (df["Excipient_conc"] * excipient_mw) / 1000.0

    df["Surfactant_mg_mL"] = df["Surfactant_conc"] * 10.0

    df["log_conc"] = np.log1p(df["Protein_conc"])
    df["conc_sq"] = df["Protein_conc"] ** 2
    df["conc_x_kP"] = df["Protein_conc"] * df["kP"]

    # Quadratic concentration term modulated by kP. Captures protein-specific
    # curvature in the concentration-viscosity relationship that is not represented
    # by the linear conc_x_kP or protein-agnostic conc_sq terms, improving
    # generalization for proteins with atypical kP values.
    df["conc_sq_x_kP"] = df["conc_sq"] * df["kP"]
    df["conc_x_HCI"] = df["Protein_conc"] * df["HCI"]

    # A molar-concentration proxy (Protein_conc / MW) was evaluated as an
    # alternative crowding feature but degraded held-out performance. It was
    # therefore rejected; reintroduce only with fresh controlled validation.
    df["Crowding_Index"] = df["Protein_conc"] * df["Stabilizer_mg_mL"]
    df["Stabilizer_Squared"] = df["Stabilizer_mg_mL"] ** 2

    df["Total_Solute_Mass"] = (
        df["Protein_conc"]
        + df["Stabilizer_mg_mL"]
        + df["Excipient_mg_mL"]
        + df["Salt_mg_mL"]
        + df["Surfactant_mg_mL"]
    )
    df["Effective_Protein_Fraction"] = df["Protein_conc"] / df["Total_Solute_Mass"].replace(0, 1e-6)

    features_df = df.apply(calculate_row_priors, axis=1, result_type="expand")
    df = pd.concat([df, features_df], axis=1)

    num_cols = num_cols + PRIOR_COLS + CONC_SPLIT_COLS + ENGINEERED_COLS + prop_cols + charge_cols

    return df, num_cols, cat_cols


def _all_property_columns() -> set[str]:
    """Return all continuous chemical-property feature names.

    Retrieves the physicochemical property-space description produced by the
    categorical feature module and removes its categorical identifier columns.
    The resulting set is used to identify property features that must remain
    available during decoder-query masking.

    Returns:
        Set of physicochemical property feature column names.
    """
    tidy = describe_property_space()
    return set(tidy.columns) - {"categorical", "category"}


def _all_charge_columns() -> set[str]:
    """Return all charge-derived feature names exposed by the charge pipeline.

    Returns:
        Set of feature names defined by
        :data:`visqai.features.charge.CHARGE_FEATURE_COLS`.
    """
    return set(CHARGE_FEATURE_COLS)


def protected_feature_indices(num_cols: list[str]) -> list[int]:
    """Return numeric-feature indices that must not be decoder-masked.

    Protected features represent strong viscosity drivers or continuous
    physicochemical measurements for which replacing the observed value with
    zero would correspond to a physically meaningful state rather than a
    neutral missing-value representation. The protected set includes
    concentration-derived terms, protein identity and environmental variables,
    all continuous chemical-property features, and all charge features.

    Indices are returned relative to the supplied `num_cols` ordering, so
    they can be applied directly to a numeric feature vector or matrix whose
    columns follow that same ordering.

    Args:
        num_cols: Ordered list of numeric feature names used by the model
            preprocessor.

    Returns:
        List of zero-based indices in `num_cols` corresponding to protected
        features.

    Raises:
        TypeError: If `num_cols` is not a list.
    """
    require_type(num_cols, list, "num_cols")
    protected_names = {
        "Protein_conc",
        "log_conc",
        "conc_sq",
        "conc_x_kP",
        "conc_sq_x_kP",
        "conc_x_HCI",
        "Buffer_pH",
        "MW",
        "PI_mean",
        "Temperature",
    }
    protected_names |= _all_property_columns()
    protected_names |= _all_charge_columns()
    return [i for i, c in enumerate(num_cols) if c in protected_names]


def prepare_df(df: pd.DataFrame, drop_bad_rows: bool = False) -> pd.DataFrame:
    """Normalize raw evaluation or training rows before feature engineering.

    Copies the input DataFrame, converts integer-valued columns to floating
    point except for `ID`, normalizes `ID` to strings, and optionally
    removes rows that lack valid viscosity measurements or required physical
    inputs.

    When `drop_bad_rows` is enabled, a row is retained only if every
    viscosity column present in :data:`visqai.constants.VISC_COLS` is non-null
    and strictly positive, and every available required input among `MW`,
    `Protein_conc`, and `kP` is non-null. The resulting DataFrame is
    reindexed from zero.

    Args:
        df: Raw formulation or evaluation DataFrame.
        drop_bad_rows: Whether to remove rows with invalid viscosity outputs
            or missing required physical inputs. Defaults to `False`.

    Returns:
        A cleaned copy of `df` with normalized numeric and identifier
        columns. If `drop_bad_rows` is `True`, invalid rows are removed.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    require_dataframe(df, "df")
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
