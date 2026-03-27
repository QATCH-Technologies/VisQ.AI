"""
data_pipeline.py
================
Feature engineering, preprocessing, and sample construction for the
CBM-CNP viscosity prediction pipeline.

Public API
----------
load_and_preprocess(csv_path, save_dir=None)
    Load raw CSV, engineer physics features, normalize concept proxies,
    fit sklearn preprocessor + physics scaler, and return sample dicts.

Constants
---------
CONC_THRESHOLDS
    Per-ingredient concentration thresholds for the low/high split features.
PRIOR_TABLE
    Protein-class x pH-regime lookup table for prior feature scores.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch

from cb_cnp.constants import CONCEPT_DEFS, N_CONCEPTS_SUPERVISED, V_BAR_REGISTRY


# ============================================================
# Ingredient-level concentration thresholds
# ============================================================

CONC_THRESHOLDS: dict[str, float] = {
    "arginine": 150.0,
    "lysine": 100.0,
    "proline": 200.0,
    "nacl": 150.0,
    "tween-20": 0.01,
    "tween-80": 0.01,
    "stabilizer": 0.2,
    "trehalose": 0.2,
}


# ============================================================
# Protein-class x pH-regime prior table
# ============================================================

PRIOR_TABLE: dict[str, dict[str, dict[str, int]]] = {
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


# ============================================================
# Internal helpers
# ============================================================

_MW_MAP: dict[str, float] = {
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
        .map(lambda x: next((mw for name, mw in _MW_MAP.items() if name in x), default_mw))
    )


import numpy as np
import pandas as pd


def _calculate_protein_phi(df: pd.DataFrame) -> pd.Series:
    """
    Calculates Phi_Protein dynamically using MW, pI, and HCI.
    Avoids hard-coded types by using physical scaling laws.
    """
    # 1. Base Partial Specific Volume (the 'dry' volume)
    # Most proteins fall between 0.70 and 0.75; 0.73 is the physical mean.
    V_BAR_DRY = 0.73

    # 2. Hydration Adjustment (Water bound to the surface)
    # Standard hydration (delta) is ~0.3g water / 1g protein.
    # We scale this by Hydrophobicity (HCI). Higher HCI = Lower Hydration.
    # We normalize HCI (assuming it's roughly 0 to 1, or center it).
    hydration_delta = 0.35 * (1.1 - df["HCI"].clip(0, 1))

    # 3. Electrostatic Expansion (The 'Charge' effect)
    # Proteins expand effectively when highly charged (far from pI).
    # If pH is not in your DF, we use C_Class as a proxy for charge density.
    charge_expansion = df["C_Class"].abs() * 0.05

    # 4. Combined Effective Specific Volume (mL/g)
    # V_eff = V_dry + V_hydrated + V_electrostatic
    v_eff = V_BAR_DRY + hydration_delta + charge_expansion

    # 5. Convert to Phi (Volume Fraction)
    # Conc (mg/mL) * v_eff (mL/g) / 1000 mg/g = Dimensionless Ratio
    return (df["Protein_conc"] * v_eff) / 1000.0


def _engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Safely adds physics-inspired engineered columns to df in-place.
    Protects against NaN, Inf, and Zero-division.
    """
    print("Normalizing units and calculating Physics Features...")

    def safe_div(num, den, epsilon=1e-8):
        return num / den.replace(0, np.nan).fillna(epsilon)

    st_mw = _get_mw(df["Stabilizer_type"], default_mw=342.3).replace(0, 342.3).fillna(342.3)
    df["Stabilizer_mg_mL"] = df["Stabilizer_conc"] * st_mw
    sa_mw = _get_mw(df["Salt_type"], default_mw=58.44).replace(0, 58.44).fillna(58.44)
    df["Salt_mg_mL"] = (df["Salt_conc"] * sa_mw) / 1000.0

    ex_mw = _get_mw(df["Excipient_type"], default_mw=150.0).replace(0, 150.0).fillna(150.0)
    df["Excipient_mg_mL"] = (df["Excipient_conc"] * ex_mw) / 1000.0
    df["Surfactant_mg_mL"] = df["Surfactant_conc"] * 10.0

    df["log_conc"] = np.log1p(df["Protein_conc"].clip(lower=0))
    df["conc_sq"] = df["Protein_conc"] ** 2
    df["conc_x_kP"] = df["Protein_conc"] * df["kP"].fillna(0)
    df["conc_x_HCI"] = df["Protein_conc"] * df["HCI"].fillna(0)
    df["Crowding_Index"] = df["Protein_conc"] * df["Stabilizer_mg_mL"]
    df["Stabilizer_Squared"] = df["Stabilizer_mg_mL"] ** 2

    df["Total_Solute_Mass"] = (
        df["Protein_conc"]
        + df["Stabilizer_mg_mL"]
        + df["Excipient_mg_mL"]
        + df["Salt_mg_mL"]
        + df["Surfactant_mg_mL"]
    ).clip(lower=1e-6)

    def get_vbar_series(type_series, default_key):
        return type_series.map(V_BAR_REGISTRY).fillna(V_BAR_REGISTRY[default_key])

    vbar_stab = get_vbar_series(df["Stabilizer_type"], "Stabilizer_default")
    vbar_salt = get_vbar_series(df["Salt_type"], "Salt_default")
    vbar_excip = get_vbar_series(df["Excipient_type"], "Excipient_default")
    vbar_surf = get_vbar_series(df["Surfactant_type"], "Surfactant_default")

    df["Phi_Protein"] = _calculate_protein_phi(df)
    df["Phi_Stabilizer"] = (df["Stabilizer_mg_mL"] * vbar_stab) / 1000.0
    df["Phi_Salt"] = (df["Salt_mg_mL"] * vbar_salt) / 1000.0
    df["Phi_Excipient"] = (df["Excipient_mg_mL"] * vbar_excip) / 1000.0
    df["Phi_Surfactant"] = (df["Surfactant_mg_mL"] * vbar_surf) / 1000.0
    df["Phi_Total"] = (
        df["Phi_Protein"]
        + df["Phi_Stabilizer"]
        + df["Phi_Salt"]
        + df["Phi_Excipient"]
        + df["Phi_Surfactant"]
    ).fillna(0)

    PHI_MAX = 0.65
    safe_phi = df["Phi_Total"].clip(lower=0, upper=PHI_MAX - 0.001)
    df["KD_Asymptote"] = np.log1p((1.0 - (safe_phi / PHI_MAX)) ** -2.0)
    df["Exp_Crowding"] = np.log1p(np.exp(safe_phi * 2.5))
    df["Exp_Crowding"] = np.exp(safe_phi * 2.5)
    df["Excipient_Molarity"] = safe_div(df["Excipient_mg_mL"], ex_mw)
    prot_mw = df["MW"].replace(0, np.nan).fillna(df["MW"].median() if not df["MW"].empty else 150)
    df["Protein_Molarity"] = safe_div(df["Protein_conc"], prot_mw)

    df["Excipient_Molar_Ratio"] = safe_div(df["Excipient_Molarity"], df["Protein_Molarity"])
    df["Surfactant_Loading"] = safe_div(df["Surfactant_mg_mL"], df["Protein_conc"])
    df["Total_Ionic_Strength"] = np.sqrt(
        ((df["Salt_conc"] + df["Excipient_conc"]) / 1000.0).clip(lower=0)
    )
    df["charge_x_ionic"] = df["C_Class"].fillna(0) * df["Total_Ionic_Strength"]
    df["Effective_Protein_Fraction"] = df["Protein_conc"] / df["Total_Solute_Mass"].replace(0, 1e-6)
    engineered_cols = [
        "log_conc",
        "conc_sq",
        "conc_x_kP",
        "conc_x_HCI",
        "Crowding_Index",
        "Stabilizer_Squared",
        "Total_Solute_Mass",
        "Effective_Protein_Fraction",
        "KD_Asymptote",
        "Exp_Crowding",
        "Phi_Protein",
        "Phi_Stabilizer",
        "Phi_Total",
        "charge_x_ionic",
    ]
    # engineered_cols = [
    #     "log_conc",
    #     "conc_sq",
    #     "conc_x_kP",
    #     "conc_x_HCI",
    #     "Crowding_Index",
    #     "Stabilizer_Squared",
    #     "Total_Solute_Mass",
    #     "Effective_Protein_Fraction",
    #     "KD_Asymptote",
    #     "Exp_Crowding",
    #     "Phi_Protein",
    #     "Phi_Stabilizer",
    #     "Phi_Total",
    #     "charge_x_ionic",
    #     "Excipient_Molarity",
    #     "Protein_Molarity",
    #     "Excipient_Molar_Ratio",
    #     "Total_Ionic_Strength",
    #     "Surfactant_Loading",
    # ]

    # Replace any infinite values with 0 (or a large number) and NaNs with 0
    df[engineered_cols] = df[engineered_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df, list(dict.fromkeys(engineered_cols))  # Remove duplicates


def _process_row_features(row: pd.Series) -> dict[str, float]:
    """
    Compute pH-regime prior scores and low/high concentration split features
    for a single formulation row.

    Returns a flat dict of feature name -> value for all prior and conc-split cols.
    """
    new_prior_cols = [
        "prior_arginine",
        "prior_lysine",
        "prior_proline",
        "prior_nacl",
        "prior_stabilizer",
        "prior_tween-20",
        "prior_tween-80",
    ]
    new_conc_cols = []
    for k in CONC_THRESHOLDS:
        new_conc_cols.extend([f"{k}_low", f"{k}_high"])

    c_class = row.get("C_Class", 1.0)
    ph = row.get("Buffer_pH", 7.0)
    pi = row.get("PI_mean", 7.0)
    if pd.isna(ph):
        ph = 7.0
    if pd.isna(pi):
        pi = 7.0

    delta_ph = abs(ph - pi)
    tau = 1.5
    cci = c_class * np.exp(-delta_ph / tau)

    p_type = str(row.get("Protein_class_type", "default")).lower()
    if "mab_igg1" in p_type:
        regime = "Near-pI" if cci >= 0.90 else ("Mixed" if cci >= 0.50 else "Far")
    elif "mab_igg4" in p_type:
        regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.40 else "Far")
    elif any(x in p_type for x in ["fc-fusion", "trispecific"]):
        regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
    elif any(x in p_type for x in ["bispecific", "adc"]):
        regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.45 else "Far")
    elif any(x in p_type for x in ["bsa", "polyclonal"]):
        regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
    else:
        regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")

    lookup_key = next((k for k in PRIOR_TABLE if k != "default" and k in p_type), "default")
    table = PRIOR_TABLE[lookup_key]
    regime_dict = table.get(regime, table["Far"])

    priors = {k: 0.0 for k in new_prior_cols}
    concs = {k: 0.0 for k in new_conc_cols}

    scan_cols = [
        ("Salt_type", "Salt_conc"),
        ("Stabilizer_type", "Stabilizer_conc"),
        ("Excipient_type", "Excipient_conc"),
        ("Surfactant_type", "Surfactant_conc"),
    ]
    for type_col, conc_col in scan_cols:
        ing_name = str(row.get(type_col, "none")).lower()
        ing_conc = float(row.get(conc_col, 0.0))
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


def _normalize_concept_proxies(
    df: pd.DataFrame,
    save_dir: str | None,
) -> np.ndarray:
    """
    Extract, sign-correct, z-score, and activation-normalize concept proxy columns.

    v3 normalization:
      - tanh concepts: tanh(z / 2) -> [-1, 1]
      - sigmoid concepts: sigmoid(z) -> [0, 1]

    Saves normalisation statistics to save_dir if provided.

    Returns
    -------
    concept_normalized : np.ndarray [N, N_CONCEPTS_SUPERVISED]
    """
    proxy_cols = [cd[1] for cd in CONCEPT_DEFS]
    proxy_signs = np.array([cd[2] for cd in CONCEPT_DEFS], dtype=float)
    proxy_activations = [cd[3] for cd in CONCEPT_DEFS]

    concept_raw = np.zeros((len(df), N_CONCEPTS_SUPERVISED), dtype=np.float64)
    for j, col in enumerate(proxy_cols):
        if col in df.columns:
            concept_raw[:, j] = df[col].fillna(0.0).values.astype(float)

    concept_raw_signed = concept_raw * proxy_signs
    c_mean = concept_raw_signed.mean(axis=0)
    c_std = concept_raw_signed.std(axis=0) + 1e-8
    z_scored = (concept_raw_signed - c_mean) / c_std

    concept_normalized = np.zeros_like(z_scored)
    for j, act_type in enumerate(proxy_activations):
        if act_type == "sigmoid":
            concept_normalized[:, j] = 1.0 / (1.0 + np.exp(-z_scored[:, j]))
        else:
            concept_normalized[:, j] = np.tanh(z_scored[:, j] / 2.0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "concept_proxy_mean.npy"), c_mean)
        np.save(os.path.join(save_dir, "concept_proxy_std.npy"), c_std)
        np.save(os.path.join(save_dir, "concept_proxy_signs.npy"), proxy_signs)
        np.save(
            os.path.join(save_dir, "concept_proxy_activations.npy"),
            np.array(proxy_activations),
        )
        print(f"Concept proxy scaler saved to {save_dir}/concept_proxy_*.npy")

    return concept_normalized


def _build_samples(
    df: pd.DataFrame,
    X_matrix: np.ndarray,
    physics_scaler: object,
    concept_normalized: np.ndarray,
) -> list[dict]:
    """
    Construct per-row sample dicts containing interpolated viscosity curves,
    scaled static features, and concept proxy targets.

    Each sample dict contains:
        "static"          : Tensor [static_dim]
        "points"          : Tensor [N_pts, 2]  (scaled log-shear, scaled log-visc)
        "group"           : str  (Protein_type)
        "id"              : str  (row ID)
        "concept_targets" : Tensor [N_CONCEPTS_SUPERVISED]
    """
    shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1000.0,
        "Viscosity_10000": 10000.0,
        "Viscosity_100000": 100000.0,
        "Viscosity_15000000": 1.5e7,
    }
    key_shears = [100.0, 1000.0, 10000.0, 100000.0, 15000000.0]
    key_logs = np.log10(key_shears)

    samples = []
    for i in range(len(df)):
        raw_x, raw_y = [], []
        for col, shear_val in shear_map.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = df.iloc[i][col]
                if v <= 0:
                    v = 1e-6
                raw_x.append(np.log10(shear_val))
                raw_y.append(np.log10(v))

        if len(raw_x) < 3:
            continue

        si = np.argsort(raw_x)
        x_arr, y_arr = np.array(raw_x)[si], np.array(raw_y)[si]
        interpolator = PchipInterpolator(x_arr, y_arr)

        interval_endpoints = np.unique(np.concatenate([x_arr, key_logs]))
        interval_endpoints = interval_endpoints[
            (interval_endpoints >= x_arr.min()) & (interval_endpoints <= x_arr.max())
        ]
        interval_endpoints.sort()

        dense_x_list = []
        for j in range(len(interval_endpoints) - 1):
            interval_pts = np.linspace(interval_endpoints[j], interval_endpoints[j + 1], 10)
            dense_x_list.append(
                interval_pts[:-1] if j < len(interval_endpoints) - 2 else interval_pts
            )

        dense_x = np.concatenate(dense_x_list) if dense_x_list else x_arr
        dense_y = interpolator(dense_x)

        pts = [
            physics_scaler.transform(np.array([[dx, dy]]))[0] for dx, dy in zip(dense_x, dense_y)
        ]

        if pts:
            pts_np = np.stack(pts)
            samples.append(
                {
                    "static": torch.tensor(X_matrix[i], dtype=torch.float32),
                    "points": torch.tensor(pts_np, dtype=torch.float32),
                    "group": df.iloc[i]["Protein_type"],
                    "id": df.iloc[i]["ID"],
                    "concept_targets": torch.tensor(concept_normalized[i], dtype=torch.float32),
                }
            )

    return samples


# ============================================================
# Public entry point
# ============================================================


def load_and_preprocess(
    csv_path: str,
    save_dir: str | None = None,
) -> tuple[list[dict], int]:
    """
    Load raw formulation CSV and return model-ready sample dicts.

    Steps
    -----
    1. Load CSV; fill missing numeric/categorical columns.
    2. Engineer physics features (unit conversions, interaction terms,
       volume fractions, electrostatics).
    3. Compute pH-regime prior scores and concentration-split features.
    4. Extract and normalize concept proxy targets.
    5. Fit sklearn ColumnTransformer (StandardScaler + OneHotEncoder).
    6. Fit physics scaler on (log-shear, log-viscosity) pairs.
    7. Build per-sample dicts with PCHIP-interpolated viscosity curves.

    Parameters
    ----------
    csv_path : str
        Path to the raw formulation CSV.
    save_dir : str | None
        If provided, saves preprocessor.pkl, physics_scaler.pkl, and
        concept_proxy_*.npy to this directory.

    Returns
    -------
    samples : list[dict]
        One dict per valid row (rows with < 3 viscosity measurements are dropped).
    static_dim : int
        Dimensionality of the static feature vector after preprocessing.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df.to_csv("pembro_data.csv", index=False)

    cat_cols = [
        "Protein_type",
        "Protein_class_type",
        "Buffer_type",
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
        "Excipient_type",
    ]
    num_cols = [
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

    # Feature engineering
    df, engineered_cols = _engineer_features(df)

    # pH-regime prior and concentration-split features
    new_prior_cols = [
        "prior_arginine",
        "prior_lysine",
        "prior_proline",
        "prior_nacl",
        "prior_stabilizer",
        "prior_tween-20",
        "prior_tween-80",
    ]
    new_conc_cols = []
    for k in CONC_THRESHOLDS:
        new_conc_cols.extend([f"{k}_low", f"{k}_high"])

    print("Calculating Regimes and Concentration Splits...")
    features_df = df.apply(_process_row_features, axis=1, result_type="expand")
    df = pd.concat([df, features_df], axis=1)

    num_cols = num_cols + new_prior_cols + new_conc_cols + engineered_cols

    # Concept proxy normalization
    concept_normalized = _normalize_concept_proxies(df, save_dir)

    # Static feature preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    X_matrix = preprocessor.fit_transform(df)
    if np.isnan(X_matrix).any():
        print("WARNING: NaNs found in X_matrix after preprocessing! Replacing with 0.")
        X_matrix = np.nan_to_num(X_matrix)

    # Physics scaler fitted on all (log-shear, log-viscosity) pairs
    shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1000.0,
        "Viscosity_10000": 10000.0,
        "Viscosity_100000": 100000.0,
        "Viscosity_15000000": 1.5e7,
    }
    all_shear, all_visc = [], []
    for i in range(len(df)):
        for col, shear_val in shear_map.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = df.iloc[i][col]
                if v <= 0:
                    v = 1e-6
                all_shear.append(np.log10(shear_val))
                all_visc.append(np.log10(v))

    physics_scaler = StandardScaler()
    physics_scaler.fit(np.column_stack([all_shear, all_visc]))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.pkl"))
        joblib.dump(physics_scaler, os.path.join(save_dir, "physics_scaler.pkl"))

    samples = _build_samples(df, X_matrix, physics_scaler, concept_normalized)
    return samples, X_matrix.shape[1]
