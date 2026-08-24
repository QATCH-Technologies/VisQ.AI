"""
priors.py
=========
Physics-prior lookup tables and the charge-coupling-index (CCI) / regime
classification used to select a formulation's excipient-interaction prior.

CONC_THRESHOLDS and PRIOR_TABLE were previously duplicated verbatim between
ml/cnp_mk2/inference_o_net.py and ml/cnp_mk2/train_o_net_v4_rung1.py (diffed
byte-identical before merging here).

calculate_cci uses the |pH - PI_mean| distance proxy: peaks when the
formulation pH sits at the protein's isoelectric point.
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

# --- 2026-07-15 marginal-effect audit (see docstring below the table) -----
# arginine / proline / nacl / tween-20 were dropped to 0 in every cell: a
# multivariate OLS of log10-viscosity on Protein_conc + Protein_type +
# co-occurring ingredient concentration, using the SAME charge-aware
# regime/CCI this table is keyed by, found:
#   - nacl: the table claims a uniform -1 in every regime, but conditioned
#     on the real regime the sign FLIPS -- significantly negative Near-pI
#     (p<0.001) vs significantly POSITIVE in Far, which is 66% of the data
#     (coef +0.0006-0.0016, p=0.001-0.004). A flat per-regime integer can't
#     represent a sign that depends on concentration/regime this way, and
#     the wrong-signed Far-regime majority is the dominant failure mode
#     (see charge.py's charge_screened fix for the other half of this --
#     the OOD covariate shift when nacl rows are LOGO-held-out).
#   - arginine / proline / tween-20: pooled sign is directionally negative
#     (matches the table) but never reaches significance within any single
#     regime cell at the support these categories have (5-42 rows per
#     regime), including the cell the table assigns its LARGEST magnitude
#     to (mab_igg1/igg4/bispecific/adc Near-pI, arginine=-2: n_present=5,
#     p=0.69, coef effectively 0). There isn't enough signal to justify a
#     hand-tuned, regime-differentiated integer; the physicochemical
#     property vectors (categorical.py) already carry these ingredients'
#     real descriptors (charge, MW, logP, HLB, ...) for the network to
#     learn from directly, without a contradicted prior fighting it.
# lysine and stabilizer were left as-is: both show a highly significant,
# regime-consistent, correctly-signed effect in the same audit (lysine
# coef -0.019..-0.024, p<=0.002; stabilizer coef +0.126..+0.128, p<1e-45).
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

# Ceiling on the "_high" fraction-of-threshold (concentration above the
# CONC_THRESHOLDS cutoff, in units of the threshold itself). Bounds how far
# an extreme concentration can push this feature, same rationale as
# charge.py's ION_STRENGTH_CAP_M.
CONC_HIGH_FRAC_CAP: float = 3.0


def calculate_cci(c_class, ph, pi, tau: float = 1.5) -> float:
    """Charge-coupling index: peaks (i.e. -> C_Class) when the formulation pH
    sits at the protein's isoelectric point (the |pH - PI_mean| distance
    proxy)."""
    delta_ph = abs(ph - pi)
    return float(c_class) * float(np.exp(-delta_ph / tau))


def calculate_regime(cci: float, p_type: str) -> str:
    """Map a CCI value to a Near-pI/Mixed/Far regime, with per-protein-class
    thresholds (matches process_row_features / _calculate_physics_features)."""
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
    """Per-row prior/concentration-split features."""
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
                # Fraction-of-threshold, not raw concentration (2026-07-15
                # proline trace): under a leave-one-ingredient-out fold these
                # two columns are IDENTICALLY ZERO in every training row (the
                # held-out ingredient never appears in train), so the fitted
                # StandardScaler has zero/near-zero variance for them: any
                # nonzero held-out value then blows up into a huge z-score
                # the network never learned to handle. Isolating this
                # (zeroing only proline_low/high against a real trained
                # model) reproduced the entire -0.38 ablation_delta gap on
                # its own -- the property vector was not at fault. Dividing
                # by `threshold` bounds the raw magnitude of that jump to
                # O(1) instead of O(concentration units), the same fix
                # applied to charge.py's ionic-strength proxy for nacl.
                concs[f"{target_ing}_low"] = min(ing_conc, threshold) / threshold
                concs[f"{target_ing}_high"] = min(max(ing_conc - threshold, 0) / threshold, CONC_HIGH_FRAC_CAP)

    return {**priors, **concs}
