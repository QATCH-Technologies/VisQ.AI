"""Physics-derived formulation priors and concentration-regime features used by
the VisQAI row-level feature-engineering pipeline.

This module centralizes the concentration thresholds, formulation prior lookup
table, charge-coupling-index (CCI) calculation, protein-specific regime
classification, and per-row prior/concentration feature construction that
were previously duplicated across training and inference implementations.

The CCI is based on the distance between formulation pH and the protein's
mean isoelectric point (PI_mean). It is highest when formulation pH is close
to PI_mean and is scaled by `C_Class`. The resulting CCI is mapped to a
protein-type-dependent `Near-pI`, `Mixed`, or `Far` regime, which selects
the appropriate formulation-interaction prior from :data:`PRIOR_TABLE`.

Concentration-split features represent ingredient concentration relative to
their corresponding threshold rather than in raw concentration units. Values
above the threshold are capped by :data:`CONC_HIGH_FRAC_CAP` to limit
extrapolation when an ingredient is absent from a training fold but present
in a held-out fold.

The prior table reflects a marginal-effect audit performed on the charge-aware
regime definition. Arginine, proline, NaCl, and Tween-20 priors were reduced
to zero where the available within-regime evidence did not justify the
previous hand-tuned effects. Lysine and stabilizer effects were retained
because their effects were statistically significant, correctly signed, and
consistent across regimes. Chemical-property vectors remain responsible for
representing the underlying physicochemical properties of formulation
ingredients rather than relying on unsupported categorical prior magnitudes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONC_THRESHOLDS = {
    "arginine": 150.0,
    "lysine": 100.0,
    "proline": 200.0,
    "nacl": 150.0,
    "tween-20": 0.01,
    "tween-80": 0.01,
    "stabilizer": 0.2,
    "trehalose": 0.2,
}

# Ingredient priors were simplified where the data did
# not support stable, regime-specific effects. Arginine, proline, NaCl, and
# Tween-20 were set to zero because their observed effects were weak,
# inconsistent across regimes, or poorly supported. Their physicochemical
# properties are already represented by the chemical feature vectors.
#
# Lysine and stabilizer priors were retained because their effects were
# statistically significant, correctly signed, and consistent across regimes.
PRIOR_TABLE = {
    "mab_igg1": {
        "Near-pI": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -1,
        },
        "Far": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -1,
        },
    },
    "mab_igg4": {
        "Near-pI": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -1,
        },
        "Far": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -1,
        },
    },
    "fc-fusion": {
        "Near-pI": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -2,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -2,
        },
    },
    "bispecific": {
        "Near-pI": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -2,
        },
    },
    "adc": {
        "Near-pI": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": -2,
        },
    },
    "bsa": {
        "Near-pI": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
    },
    "polyclonal": {
        "Near-pI": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
    },
    "default": {
        "Near-pI": {
            "arginine": 0,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
    },
}

PRIOR_COLS = [
    "prior_arginine",
    "prior_lysine",
    "prior_proline",
    "prior_nacl",
    "prior_stabilizer",
    "prior_tween-20",
    "prior_tween-80",
]

CONC_SPLIT_COLS = [c for k in CONC_THRESHOLDS for c in (f"{k}_low", f"{k}_high")]

# Ceiling on the `_high` fraction-of-threshold feature. Limits the
# contribution of concentrations above their threshold and prevents extreme
# values from producing unsupported extrapolation, consistent with the
# capping strategy used for other normalized physical features.
CONC_HIGH_FRAC_CAP: float = 3.0


def calculate_cci(c_class, ph, pi, tau: float = 1.5) -> float:
    """Calculate the charge-coupling index for a formulation.

    The CCI is a pH-distance proxy for the degree to which a formulation is
    expected to exhibit charge-related coupling. It is proportional to
    `C_Class` and decays exponentially as the absolute difference between
    formulation pH and the protein's mean isoelectric point increases.

    The calculation is:

    `CCI = C_Class * exp(-abs(pH - PI_mean) / tau)`

    Consequently, the index reaches its maximum value of approximately
    `C_Class` when pH equals PI_mean and decreases toward zero as the pH
    moves farther from the isoelectric point.

    Args:
        c_class: Protein charge-class or charge-coupling scaling factor.
        ph: Formulation buffer pH.
        pi: Protein mean isoelectric point.
        tau: Exponential decay constant controlling how rapidly CCI decreases
            with pH-to-pI distance. Defaults to `1.5`.

    Returns:
        The scalar charge-coupling index as a Python `float`.
    """
    delta_ph = abs(ph - pi)
    return float(c_class) * float(np.exp(-delta_ph / tau))


def calculate_regime(cci: float, p_type: str) -> str:
    """Classify a formulation into a pH/charge interaction regime.

    Regime boundaries depend on the protein type because different protein
    classes use different CCI ranges to define Near-pI, Mixed, and Far
    behavior. Protein-type matching is case-insensitive and uses substring
    matching so related type labels can share the same thresholds.

    The recognized threshold families are:

    * `mab_igg1`
    * `mab_igg4`
    * `fc-fusion` and `trispecific`
    * `bispecific` and `adc`
    * `bsa` and `polyclonal`
    * A default threshold family for all other protein types.

    Args:
        cci: Charge-coupling index calculated from formulation pH, protein pI,
            and charge class.
        p_type: Protein class/type identifier used to select the appropriate
            CCI thresholds.

    Returns:
        `"Near-pI"`, `"Mixed"`, or `"Far"` according to the applicable
        protein-type-specific thresholds.
    """
    p_type = str(p_type).lower()
    if "mab_igg1" in p_type:
        return "Near-pI" if cci >= 0.90 else ("Mixed" if cci >= 0.50 else "Far")
    if "mab_igg4" in p_type:
        return "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.40 else "Far")
    if any(x in p_type for x in ["fc-fusion", "trispecific"]):
        return "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
    if any(x in p_type for x in ["bispecific", "adc"]):
        return "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.45 else "Far")
    if any(x in p_type for x in ["bsa", "polyclonal"]):
        return "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
    return "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")


def calculate_row_priors(row) -> dict:
    """Calculate prior and concentration-split features for one formulation.

    Determines the formulation's charge-coupling regime from CCI, selects the
    corresponding protein-class prior table, and populates ingredient-specific
    prior features when recognized formulation ingredients are present.

    The function scans salt, stabilizer, excipient, and surfactant type and
    concentration columns. Recognized ingredients activate their corresponding
    prior feature, while concentration-split features are generated as
    normalized fractions of the ingredient-specific threshold in
    :data:`CONC_THRESHOLDS`.

    Concentrations at or below the threshold populate the `*_low` feature.
    Concentrations above the threshold populate `*_high` using only the
    amount above the threshold. The high-concentration fraction is capped at
    :data:`CONC_HIGH_FRAC_CAP` to prevent extreme concentrations from producing
    values outside the range represented by the training pipeline.

    Missing pH and pI values default to `7.0`. Missing, non-numeric, or
    non-positive ingredient concentrations are treated as absent. Unknown or
    `none` ingredient labels do not activate prior or concentration features.

    Args:
        row: Row-like object supporting `get` and containing formulation
            fields such as `C_Class`, `Buffer_pH`, `PI_mean`,
            `Protein_class_type`, ingredient type columns, and ingredient
            concentration columns.

    Returns:
        Dictionary containing all columns in :data:`PRIOR_COLS` and
        :data:`CONC_SPLIT_COLS`. Features for absent or unrecognized
        ingredients are returned as `0.0`.
    """
    c_class = row.get("C_Class", 1.0)
    ph = row.get("Buffer_pH", 7.0)
    pi = row.get("PI_mean", 7.0)
    if pd.isna(ph):
        ph = 7.0
    if pd.isna(pi):
        pi = 7.0

    cci = calculate_cci(c_class, ph, pi)

    p_type = str(row.get("Protein_class_type", "default")).lower()
    regime = calculate_regime(cci, p_type)

    lookup_key = "default"
    for key in PRIOR_TABLE.keys():
        if key != "default" and key in p_type:
            lookup_key = key
            break
    table = PRIOR_TABLE[lookup_key]
    regime_dict = table.get(regime, table["Far"])

    priors = {k: 0.0 for k in PRIOR_COLS}
    concs = {k: 0.0 for k in CONC_SPLIT_COLS}

    scan_cols = [
        ("Salt_type", "Salt_conc"),
        ("Stabilizer_type", "Stabilizer_conc"),
        ("Excipient_type", "Excipient_conc"),
        ("Surfactant_type", "Surfactant_conc"),
    ]

    for type_col, conc_col in scan_cols:
        ing_name = str(row.get(type_col, "none")).lower()
        try:
            ing_conc = float(row.get(conc_col, 0.0))
        except (TypeError, ValueError):
            ing_conc = 0.0
        if pd.isna(ing_conc):
            ing_conc = 0.0
        if ing_name in ["none", "unknown", "nan"] or ing_conc <= 0:
            continue

        if "arginine" in ing_name or "arg" in ing_name:
            priors["prior_arginine"] = regime_dict.get("arginine", 0)
        elif "lysine" in ing_name or "lys" in ing_name:
            priors["prior_lysine"] = regime_dict.get("lysine", 0)
        elif "proline" in ing_name:
            priors["prior_proline"] = regime_dict.get("proline", 0)
        elif "nacl" in ing_name:
            priors["prior_nacl"] = regime_dict.get("nacl", 0)
        elif type_col == "Stabilizer_type":
            priors["prior_stabilizer"] = regime_dict.get("stabilizer", 0)
        elif "tween" in ing_name or "polysorbate" in ing_name:
            t_key = "tween-20" if "20" in ing_name else "tween-80"
            priors[f"prior_{t_key}"] = regime_dict.get(t_key, 0)

        for target_ing, threshold in CONC_THRESHOLDS.items():
            match = (target_ing in ing_name) or (target_ing == "arginine" and "arg" in ing_name)
            if match:
                # Normalize concentration relative to its threshold rather than using raw
                # concentration. This keeps the feature on a consistent scale and limits
                # extrapolation when an ingredient is absent from a training fold but present
                # in a held-out fold, preventing large standardized values from unsupported
                # concentration units.
                concs[f"{target_ing}_low"] = min(ing_conc, threshold) / threshold
                concs[f"{target_ing}_high"] = min(
                    max(ing_conc - threshold, 0) / threshold, CONC_HIGH_FRAC_CAP
                )

    return {**priors, **concs}
