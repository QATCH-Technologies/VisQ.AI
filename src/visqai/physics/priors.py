"""
priors.py
=========
Physics-prior lookup tables and the charge-coupling-index (CCI) / regime
classification used to select a formulation's excipient-interaction prior.

CONC_THRESHOLDS and PRIOR_TABLE were previously duplicated verbatim between
ml/cnp_mk2/inference_o_net.py and ml/cnp_mk2/train_o_net_v4_rung1.py (diffed
byte-identical before merging here).

calculate_cci is extracted from the TRAINER's process_row_features, not from
inference_o_net.py's _calculate_cci — the two had diverged: the trainer's
version branches on the real net_charge (computed by
visqai.features.charge.featurize_charge) when available, while inference's
version still used the older |pH - PI_mean| distance proxy unconditionally.
Since train_o_net_v4_rung1.py is the current, charge-aware trainer and its
fitted preprocessor already expects charge columns, the trainer's version is
the correct one to standardize on — this fixes the train/inference skew where
the near-pI regime lookup silently used stale physics at inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from visqai.features.charge import NEAR_PI_SIGMA

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

PRIOR_TABLE = {
    "mab_igg1": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Far": {
            "arginine": 0,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
    },
    "mab_igg4": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Far": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
    },
    "fc-fusion": {
        "Near-pI": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": 0,
            "nacl": 0,
            "proline": -2,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -2,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
    },
    "bispecific": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
    },
    "adc": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
    },
    "bsa": {
        "Near-pI": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
    },
    "polyclonal": {
        "Near-pI": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
    },
    "default": {
        "Near-pI": {
            "arginine": -1,
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
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
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


def calculate_cci(c_class, ph, pi, net_charge=None, near_pi_sigma: float = NEAR_PI_SIGMA, tau: float = 1.5) -> float:
    """Charge-coupling index: peaks (i.e. -> C_Class) when the protein sits at
    its isoelectric point. Uses the real net charge when available (the
    charge-aware branch that was missing from inference_o_net.py's copy), and
    falls back to the legacy |pH - PI_mean| distance proxy otherwise (older
    data / no charge column)."""
    if net_charge is not None and not pd.isna(net_charge):
        return float(c_class) * float(np.exp(-(float(net_charge) ** 2) / (2.0 * (near_pi_sigma**2))))
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
    """
    Per-row prior/concentration-split features. Matches process_row_features
    (train_o_net_v4_rung1.py) / _calculate_physics_features
    (inference_o_net.py) exactly, EXCEPT this version always takes the
    charge-aware CCI branch when `net_charge` is present on the row — which
    requires visqai.preprocessing.pipeline.build_feature_frame to have run
    charge featurization before calling this, closing the train/inference gap.
    """
    c_class = row.get("C_Class", 1.0)
    ph = row.get("Buffer_pH", 7.0)
    pi = row.get("PI_mean", 7.0)
    if pd.isna(ph):
        ph = 7.0
    if pd.isna(pi):
        pi = 7.0

    net_charge = row.get("net_charge", None)
    cci = calculate_cci(c_class, ph, pi, net_charge=net_charge)

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
                concs[f"{target_ing}_low"] = min(ing_conc, threshold)
                concs[f"{target_ing}_high"] = max(ing_conc - threshold, 0)

    return {**priors, **concs}
