"""
data.py
=======
Data loading and preprocessing pipeline for TransformerNP.

Public API
----------
load_and_preprocess(csv_path, save_dir=None)
    -> (samples: list[dict], static_ctx_dim: int, static_qry_dim: int)

Each sample dict contains:
    "static"          torch.float32 [static_ctx_dim]  — full features (context encoder)
    "static_qry"      torch.float32 [static_qry_dim]  — reduced features (query encoder)
                      Excludes Protein_type and Protein_class_type so the query
                      cannot bypass cross-attention by reading protein identity directly.
    "points"          torch.float32 [N_dense, 2]  — (scaled_log_shear, scaled_log_visc)
    "group"           str            — Protein_type value
    "id"              any            — row ID from the CSV
    "concept_targets" torch.float32 [N_CONCEPTS_SUPERVISED]  — unused by TNP

Split-static rationale  [TNP-ATTN-6]
--------------------------------------
The context encoder receives the full static vector (including one-hot protein
identity) so it can produce protein-discriminative latent tokens.
The query encoder and decoder receive a reduced static vector that deliberately
omits Protein_type and Protein_class_type.  Without protein identity in the
query, the model cannot predict viscosity accurately from the decoder input
alone — it must attend selectively to the right context tokens to infer which
protein-class behaviour applies.  This makes selective cross-attention
necessary and learnable.
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tnp.constants import (
    CONC_THRESHOLDS,
    CONCEPT_DEFS,
    N_CONCEPTS_SUPERVISED,
    PRIOR_TABLE,
)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Full categorical columns — used in the CONTEXT encoder.
# Includes protein identity so context tokens are protein-discriminative.
_CAT_COLS = [
    "Protein_type",
    "Protein_class_type",
    "Buffer_type",
    "Salt_type",
    "Stabilizer_type",
    "Surfactant_type",
    "Excipient_type",
]

# Reduced categorical columns — used in the QUERY encoder and decoder.
# Protein_type and Protein_class_type are intentionally excluded so the
# model must use cross-attention (not the query static) to infer protein
# identity.  [TNP-ATTN-6]
_QRY_CAT_COLS = [
    "Buffer_type",
    "Salt_type",
    "Stabilizer_type",
    "Surfactant_type",
    "Excipient_type",
]

_NUM_COLS = [
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

_SHEAR_MAP = {
    "Viscosity_100": 100.0,
    "Viscosity_1000": 1_000.0,
    "Viscosity_10000": 10_000.0,
    "Viscosity_100000": 100_000.0,
    "Viscosity_15000000": 1.5e7,
}

_KEY_SHEARS = [100.0, 1_000.0, 10_000.0, 100_000.0, 15_000_000.0]

_MW_MAP = {
    "sucrose": 342.3,
    "trehalose": 342.3,
    "arginine": 174.2,
    "proline": 115.13,
    "lysine": 149.19,
    "nacl": 58.44,
    "default_sugar": 342.3,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_mw(series: pd.Series, default: float = 342.3) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .map(lambda x: next((mw for name, mw in _MW_MAP.items() if name in x), default))
    )


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Stabilizer_mg_mL"] = df["Stabilizer_conc"] * _get_mw(
        df["Stabilizer_type"], 342.3
    )
    df["Salt_mg_mL"] = df["Salt_conc"] * _get_mw(df["Salt_type"], 58.44) / 1000.0
    df["Excipient_mg_mL"] = (
        df["Excipient_conc"] * _get_mw(df["Excipient_type"], 150.0) / 1000.0
    )
    df["Surfactant_mg_mL"] = df["Surfactant_conc"] * 10.0

    df["log_conc"] = np.log1p(df["Protein_conc"])
    df["conc_sq"] = df["Protein_conc"] ** 2
    df["conc_x_kP"] = df["Protein_conc"] * df["kP"]
    df["conc_x_HCI"] = df["Protein_conc"] * df["HCI"]
    df["Crowding_Index"] = df["Protein_conc"] * df["Stabilizer_mg_mL"]
    df["Stabilizer_Squared"] = df["Stabilizer_mg_mL"] ** 2
    df["Total_Solute_Mass"] = (
        df["Protein_conc"]
        + df["Stabilizer_mg_mL"]
        + df["Excipient_mg_mL"]
        + df["Salt_mg_mL"]
        + df["Surfactant_mg_mL"]
    )

    VBP, VBS, VBSa, VBE = 0.73e-3, 0.62e-3, 0.30e-3, 0.70e-3
    df["Phi_Protein"] = df["Protein_conc"] * VBP
    df["Phi_Stabilizer"] = df["Stabilizer_mg_mL"] * VBS
    df["Phi_Salt"] = df["Salt_mg_mL"] * VBSa
    df["Phi_Excipient"] = df["Excipient_mg_mL"] * VBE
    df["Phi_Total"] = (
        df["Phi_Protein"] + df["Phi_Stabilizer"] + df["Phi_Salt"] + df["Phi_Excipient"]
    )

    df["Effective_Protein_Fraction"] = df["Protein_conc"] / df[
        "Total_Solute_Mass"
    ].replace(0, 1e-6)

    PHI_MAX = 0.65
    safe_phi = df["Phi_Total"].clip(upper=PHI_MAX - 0.01)
    df["KD_Asymptote"] = (1.0 - (safe_phi / PHI_MAX)) ** -2.0
    df["Exp_Crowding"] = np.exp(safe_phi * 2.5)
    df["Ionic_Strength_Proxy"] = np.sqrt(df["Salt_conc"] / 1000.0)

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
    ]
    return df, engineered_cols


def _process_row_features(row: pd.Series) -> dict:
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
        new_conc_cols += [f"{k}_low", f"{k}_high"]

    c_class = row.get("C_Class", 1.0)
    ph = row.get("Buffer_pH", 7.0)
    pi = row.get("PI_mean", 7.0)
    ph = 7.0 if pd.isna(ph) else ph
    pi = 7.0 if pd.isna(pi) else pi
    cci = c_class * np.exp(-abs(ph - pi) / 1.5)

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

    lookup_key = "default"
    for key in PRIOR_TABLE:
        if key != "default" and key in p_type:
            lookup_key = key
            break
    regime_dict = PRIOR_TABLE[lookup_key].get(regime, PRIOR_TABLE[lookup_key]["Far"])

    priors = {k: 0.0 for k in new_prior_cols}
    concs = {k: 0.0 for k in new_conc_cols}

    for type_col, conc_col in [
        ("Salt_type", "Salt_conc"),
        ("Stabilizer_type", "Stabilizer_conc"),
        ("Excipient_type", "Excipient_conc"),
        ("Surfactant_type", "Surfactant_conc"),
    ]:
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
            match = (target_ing in ing_name) or (
                target_ing == "arginine" and "arg" in ing_name
            )
            if match:
                concs[f"{target_ing}_low"] = min(ing_conc, threshold)
                concs[f"{target_ing}_high"] = max(ing_conc - threshold, 0)

    return {**priors, **concs}


def _build_physics_scaler(df: pd.DataFrame) -> StandardScaler:
    all_shear, all_visc = [], []
    for i in range(len(df)):
        for col, shear_val in _SHEAR_MAP.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = max(df.iloc[i][col], 1e-6)
                all_shear.append(np.log10(shear_val))
                all_visc.append(np.log10(v))
    scaler = StandardScaler()
    scaler.fit(np.column_stack([all_shear, all_visc]))
    return scaler


def _build_sample_list(
    df: pd.DataFrame,
    X_ctx: np.ndarray,
    X_qry: np.ndarray,
    concept_normalized: np.ndarray,
    physics_scaler: StandardScaler,
) -> list:
    """
    Build sample dicts with both full (context) and reduced (query) static vectors.

    Each dict has:
        "static"     [static_ctx_dim] — full features for context encoding
        "static_qry" [static_qry_dim] — reduced features for query (no protein identity)
        "points"     [N_dense, 2]
        "group"      str
        "id"         any
    """
    key_logs = np.log10(_KEY_SHEARS)
    samples = []

    for i in range(len(df)):
        raw_x, raw_y = [], []
        for col, shear_val in _SHEAR_MAP.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = max(df.iloc[i][col], 1e-6)
                raw_x.append(np.log10(shear_val))
                raw_y.append(np.log10(v))
        if len(raw_x) < 3:
            continue

        si = np.argsort(raw_x)
        x_arr = np.array(raw_x)[si]
        y_arr = np.array(raw_y)[si]
        interp = PchipInterpolator(x_arr, y_arr)

        endpoints = np.unique(np.concatenate([x_arr, key_logs]))
        endpoints = endpoints[(endpoints >= x_arr.min()) & (endpoints <= x_arr.max())]
        endpoints.sort()

        dense_x_list = []
        for j in range(len(endpoints) - 1):
            pts = np.linspace(endpoints[j], endpoints[j + 1], 10)
            dense_x_list.append(pts[:-1] if j < len(endpoints) - 2 else pts)
        dense_x = np.concatenate(dense_x_list) if dense_x_list else x_arr
        dense_y = interp(dense_x)

        pts = [
            physics_scaler.transform(np.array([[dx, dy]]))[0]
            for dx, dy in zip(dense_x, dense_y)
        ]
        if pts:
            pts_np = np.stack(pts)
            samples.append(
                {
                    "static": torch.tensor(X_ctx[i], dtype=torch.float32),
                    "static_qry": torch.tensor(X_qry[i], dtype=torch.float32),
                    "points": torch.tensor(pts_np, dtype=torch.float32),
                    "group": df.iloc[i]["Protein_type"],
                    "id": df.iloc[i]["ID"],
                    "concept_targets": torch.tensor(
                        concept_normalized[i], dtype=torch.float32
                    ),
                }
            )

    return samples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_preprocess(csv_path: str, save_dir: str | None = None):
    """
    Load the formulation CSV, engineer features, fit scalers, build sample list.

    Returns:
        samples         : list[dict]  — one dict per formulation row
        static_ctx_dim  : int  — dimensionality of "static" (full, for context)
        static_qry_dim  : int  — dimensionality of "static_qry" (reduced, for query)

    Artefacts saved to save_dir (if provided):
        preprocessor.pkl        — full-feature sklearn ColumnTransformer (context)
        query_preprocessor.pkl  — reduced-feature ColumnTransformer (query)
        physics_scaler.pkl      — StandardScaler for (log_shear, log_visc)
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df.to_csv("pembro_data.csv", index=False)

    num_cols = list(_NUM_COLS)

    for c in num_cols:
        df[c] = df[c].fillna(0.0) if c in df.columns else 0.0
    for c in _CAT_COLS:
        df[c] = (
            df[c].astype(str).str.lower().replace("nan", "unknown")
            if c in df.columns
            else "unknown"
        )

    df, engineered_cols = _engineer_features(df)

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
        new_conc_cols += [f"{k}_low", f"{k}_high"]

    print("Calculating Regimes and Concentration Splits...")
    features_df = df.apply(_process_row_features, axis=1, result_type="expand")
    df = pd.concat([df, features_df], axis=1)

    num_cols.extend(new_prior_cols)
    num_cols.extend(new_conc_cols)
    num_cols.extend(engineered_cols)

    # Concept proxy normalisation (CBM cross-compatibility)
    proxy_cols = [cd[1] for cd in CONCEPT_DEFS]
    proxy_signs = np.array([cd[2] for cd in CONCEPT_DEFS], dtype=float)
    concept_raw = np.zeros((len(df), N_CONCEPTS_SUPERVISED), dtype=np.float64)
    for j, col in enumerate(proxy_cols):
        if col in df.columns:
            concept_raw[:, j] = df[col].fillna(0.0).values.astype(float)
    concept_raw_signed = concept_raw * proxy_signs
    c_mean = concept_raw_signed.mean(axis=0)
    c_std = concept_raw_signed.std(axis=0) + 1e-8
    concept_normalized = np.tanh((concept_raw_signed - c_mean) / c_std / 2.0)

    # ---- Context preprocessor — full features including protein identity ----
    ctx_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _CAT_COLS,
            ),
        ]
    )
    X_ctx = ctx_preprocessor.fit_transform(df)
    if np.isnan(X_ctx).any():
        print("WARNING: NaNs in X_ctx! Replacing with 0.")
        X_ctx = np.nan_to_num(X_ctx)

    # ---- Query preprocessor — reduced: no Protein_type / Protein_class_type ----
    # [TNP-ATTN-6] Protein identity is withheld from the query so the model must
    # infer it via cross-attention over context tokens rather than reading it directly.
    qry_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _QRY_CAT_COLS,
            ),
        ]
    )
    X_qry = qry_preprocessor.fit_transform(df)
    if np.isnan(X_qry).any():
        print("WARNING: NaNs in X_qry! Replacing with 0.")
        X_qry = np.nan_to_num(X_qry)

    print(
        f"  static_ctx_dim={X_ctx.shape[1]}  "
        f"static_qry_dim={X_qry.shape[1]}  "
        f"(removed {X_ctx.shape[1] - X_qry.shape[1]} protein-identity one-hot dims from query)"
    )

    physics_scaler = _build_physics_scaler(df)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(ctx_preprocessor, os.path.join(save_dir, "preprocessor.pkl"))
        joblib.dump(qry_preprocessor, os.path.join(save_dir, "query_preprocessor.pkl"))
        joblib.dump(physics_scaler, os.path.join(save_dir, "physics_scaler.pkl"))

    samples = _build_sample_list(df, X_ctx, X_qry, concept_normalized, physics_scaler)

    return samples, X_ctx.shape[1], X_qry.shape[1]
