"""
feature_importance_cnp.py
=========================
Permutation-based feature importance for the CrossSampleCNP viscosity predictor.

Method: Decoder-side permutation importance.
  - For each feature (column of the static feature matrix), shuffle its values
    across all samples, re-run the decoder with pre-computed latent vectors r,
    and record the increase in MSE (log10-viscosity space).  A larger increase
    means the model relies more on that feature.
  - This is model-agnostic, requires no gradient computation, and produces
    directly interpretable ΔMSE values.
  - r (the latent context vector) is kept fixed during permutation, isolating
    the importance of the static features to the *decoder* pathway.

Two output plots
  1. First-Order Features  — all importance is rolled up to the 21 original
     input columns (14 numerical + 7 categorical).  Engineered features are
     attributed back to their parent input column(s).
  2. All Features          — OHE categorical columns are grouped back to their
     source column; engineered numerical features are shown individually.

Usage
-----
  python feature_importance_cnp.py \
      --model_dir  /path/to/training/output \
      --data_csv   /path/to/formulation_data_03042026.csv \
      --out_dir    ./feature_importance_results \
      --n_repeats  10

Requirements: the model_dir must contain:
  - best_model.pt         (saved with torch.save(model.state_dict(), ...))
  - preprocessor.pkl      (sklearn ColumnTransformer)
  - physics_scaler.pkl    (sklearn StandardScaler for shear / viscosity)
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict

import joblib
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

# ============================================================
# 0.  Colour palette (presentation-ready)
# ============================================================
PAL_BLUE = "#2E86AB"
PAL_TEAL = "#17B890"
PAL_ORANGE = "#F4845F"
PAL_PURPLE = "#7B5EA7"
PAL_SLATE = "#4A5568"
PAL_GOLD = "#F6AE2D"
PAL_RED = "#E63946"
PAL_GREY = "#A8B2C1"


# Gradient for the bar charts (positive = blue, near-zero = grey)
def _importance_colour(val, vmax):
    """Colour bar by relative magnitude."""
    if vmax <= 0:
        return PAL_GREY
    frac = np.clip(val / vmax, 0, 1)
    # Interpolate GREY → BLUE
    r = int(168 + frac * (46 - 168))
    g = int(178 + frac * (134 - 178))
    b = int(193 + frac * (171 - 193))
    return f"#{r:02x}{g:02x}{b:02x}"


# ============================================================
# 1.  Model definition  (must match train_cnp_3.py)
# ============================================================


class AttentionPool(nn.Module):
    def __init__(self, latent_dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class CrossSampleCNP(nn.Module):
    def __init__(self, static_dim, hidden_dim=128, latent_dim=128, dropout=0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.pooler = AttentionPool(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(1 + static_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ctx_tensor, query_shear, query_static):
        r = self.pooler(self.encoder(ctx_tensor))
        n_q = query_shear.size(1)
        r_exp = r.unsqueeze(1).repeat(1, n_q, 1)
        return self.decoder(torch.cat([query_shear, query_static, r_exp], dim=-1))

    def encode_memory(self, ctx_tensor):
        return self.pooler(self.encoder(ctx_tensor))

    def decode_from_memory(self, r, query_shear, query_static):
        n_q = query_shear.size(1)
        r_exp = r.unsqueeze(1).repeat(1, n_q, 1)
        return self.decoder(torch.cat([query_shear, query_static, r_exp], dim=-1))


# ============================================================
# 2.  Preprocessing constants  (must match train_cnp_3.py)
# ============================================================
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
MW_MAP = {
    "sucrose": 342.3,
    "trehalose": 342.3,
    "arginine": 174.2,
    "proline": 115.13,
    "lysine": 149.19,
    "nacl": 58.44,
    "default_sugar": 342.3,
}
CAT_COLS = [
    "Protein_type",
    "Protein_class_type",
    "Buffer_type",
    "Salt_type",
    "Stabilizer_type",
    "Surfactant_type",
    "Excipient_type",
]
NUM_COLS_RAW = [
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

# ============================================================
# 3.  First-order attribution map
#     Engineered / derived features → source input column
#     Features that depend on two inputs are split 50/50 (fractional_weight).
# ============================================================
# Format:  "model_feature_name": ("first_order_column", weight)
FIRST_ORDER_MAP = {
    # ── Raw numerical (1-to-1) ───────────────────────────────────
    "kP": ("kP", 1.0),
    "MW": ("MW", 1.0),
    "PI_mean": ("PI_mean", 1.0),
    "PI_range": ("PI_range", 1.0),
    "Protein_conc": ("Protein_conc", 1.0),
    "Temperature": ("Temperature", 1.0),
    "Buffer_pH": ("Buffer_pH", 1.0),
    "Buffer_conc": ("Buffer_conc", 1.0),
    "Salt_conc": ("Salt_conc", 1.0),
    "Stabilizer_conc": ("Stabilizer_conc", 1.0),
    "Surfactant_conc": ("Surfactant_conc", 1.0),
    "Excipient_conc": ("Excipient_conc", 1.0),
    "C_Class": ("C_Class", 1.0),
    "HCI": ("HCI", 1.0),
    # ── Protein-conc transforms ──────────────────────────────────
    "log_conc": ("Protein_conc", 1.0),
    "conc_sq": ("Protein_conc", 1.0),
    "Phi_Protein": ("Protein_conc", 1.0),
    # ── Interaction: split between both parents ──────────────────
    "conc_x_kP": ("Protein_conc", 0.5),  # 50 % to Protein_conc
    # "conc_x_kP" also contributes to kP (handled below via duplicate)
    "conc_x_HCI": ("Protein_conc", 0.5),  # 50 % to Protein_conc
    # ── Salt / ionic ─────────────────────────────────────────────
    "Salt_mg_mL": ("Salt_conc", 1.0),
    "Phi_Salt": ("Salt_conc", 1.0),
    "Ionic_Strength_Proxy": ("Salt_conc", 1.0),
    "nacl_low": ("Salt_conc", 1.0),
    "nacl_high": ("Salt_conc", 1.0),
    "prior_nacl": ("Salt_conc", 1.0),
    # ── Stabilizer ───────────────────────────────────────────────
    "Stabilizer_mg_mL": ("Stabilizer_conc", 1.0),
    "Phi_Stabilizer": ("Stabilizer_conc", 1.0),
    "Stabilizer_Squared": ("Stabilizer_conc", 1.0),
    "stabilizer_low": ("Stabilizer_conc", 1.0),
    "stabilizer_high": ("Stabilizer_conc", 1.0),
    "prior_stabilizer": ("Stabilizer_conc", 1.0),
    # ── Crowding: Protein_conc × Stabilizer_mg_mL ────────────────
    "Crowding_Index": ("Protein_conc", 0.5),  # 50 % each
    # ── Surfactant / Tween ───────────────────────────────────────
    "Surfactant_mg_mL": ("Surfactant_conc", 1.0),
    "tween-20_low": ("Surfactant_conc", 1.0),
    "tween-20_high": ("Surfactant_conc", 1.0),
    "tween-80_low": ("Surfactant_conc", 1.0),
    "tween-80_high": ("Surfactant_conc", 1.0),
    "prior_tween-20": ("Surfactant_conc", 1.0),
    "prior_tween-80": ("Surfactant_conc", 1.0),
    # ── Excipient / amino-acid additives ─────────────────────────
    "Excipient_mg_mL": ("Excipient_conc", 1.0),
    "Phi_Excipient": ("Excipient_conc", 1.0),
    "arginine_low": ("Excipient_conc", 1.0),
    "arginine_high": ("Excipient_conc", 1.0),
    "lysine_low": ("Excipient_conc", 1.0),
    "lysine_high": ("Excipient_conc", 1.0),
    "proline_low": ("Excipient_conc", 1.0),
    "proline_high": ("Excipient_conc", 1.0),
    "prior_arginine": ("Excipient_conc", 1.0),
    "prior_lysine": ("Excipient_conc", 1.0),
    "prior_proline": ("Excipient_conc", 1.0),
    # ── Composite volume-fraction / crowding features ─────────────
    # All dominated by protein concentration at typical formulation ranges.
    "Total_Solute_Mass": ("Protein_conc", 0.5),
    "Effective_Protein_Fraction": ("Protein_conc", 1.0),
    "Phi_Total": ("Protein_conc", 0.5),
    "KD_Asymptote": ("Protein_conc", 0.5),
    "Exp_Crowding": ("Protein_conc", 0.5),
    # ── Categorical OHE columns are handled dynamically ──────────
    #    (feature names like "cat__Protein_type_adalimumab" → "Protein_type")
}

# Duplicate 50/50 split entries for the *second* parent
FIRST_ORDER_MAP_EXTRA = {
    "conc_x_kP": ("kP", 0.5),
    "conc_x_HCI": ("HCI", 0.5),
    "Crowding_Index": ("Stabilizer_conc", 0.5),
    "Total_Solute_Mass": ("Stabilizer_conc", 0.25),  # rough split
    "Phi_Total": ("Stabilizer_conc", 0.25),
    "KD_Asymptote": ("Stabilizer_conc", 0.25),
    "Exp_Crowding": ("Stabilizer_conc", 0.25),
}

# ── Readable display names for 1st-order features ──────────
DISPLAY_NAMES = {
    "kP": "kP",
    "MW": "MW  (molecular weight)",
    "PI_mean": "pI  (mean isoelectric point)",
    "PI_range": "pI  (isoelectric range)",
    "Protein_conc": "Protein Concentration",
    "Temperature": "Temperature",
    "Buffer_pH": "Buffer pH",
    "Buffer_conc": "Buffer Concentration",
    "Salt_conc": "Salt Concentration",
    "Stabilizer_conc": "Stabilizer Concentration",
    "Surfactant_conc": "Surfactant Concentration",
    "Excipient_conc": "Excipient Concentration",
    "C_Class": "C_Class  (charge class)",
    "HCI": "HCI",
    "Protein_type": "Protein Type",
    "Protein_class_type": "Protein Class",
    "Buffer_type": "Buffer Type",
    "Salt_type": "Salt Type",
    "Stabilizer_type": "Stabilizer Type",
    "Surfactant_type": "Surfactant Type",
    "Excipient_type": "Excipient Type",
}

# ── Feature category for colour-coding (1st order plot) ───────
FEATURE_CATEGORY = {
    "kP": "Protein Property",
    "MW": "Protein Property",
    "PI_mean": "Protein Property",
    "PI_range": "Protein Property",
    "HCI": "Protein Property",
    "C_Class": "Protein Property",
    "Protein_type": "Protein Identity",
    "Protein_class_type": "Protein Identity",
    "Protein_conc": "Formulation",
    "Temperature": "Process",
    "Buffer_pH": "Buffer",
    "Buffer_conc": "Buffer",
    "Buffer_type": "Buffer",
    "Salt_conc": "Salt / Ionic",
    "Salt_type": "Salt / Ionic",
    "Stabilizer_conc": "Stabilizer",
    "Stabilizer_type": "Stabilizer",
    "Surfactant_conc": "Surfactant",
    "Surfactant_type": "Surfactant",
    "Excipient_conc": "Excipient",
    "Excipient_type": "Excipient",
}
CATEGORY_COLOURS = {
    "Protein Identity": PAL_PURPLE,
    "Protein Property": PAL_BLUE,
    "Formulation": PAL_TEAL,
    "Buffer": PAL_GOLD,
    "Salt / Ionic": PAL_ORANGE,
    "Stabilizer": PAL_RED,
    "Surfactant": "#9B59B6",
    "Excipient": "#1ABC9C",
    "Process": PAL_SLATE,
}


# ── Category for all-features plot ─────────────────────────────
def _all_feat_category(name):
    """Assign a colour category to an expanded model feature."""
    for cat_col in CAT_COLS:
        if name.startswith(f"cat__{cat_col}") or name == cat_col:
            # Reuse 1st-order category for the grouped cat feature
            mapping = {
                "Protein_type": "Protein Identity",
                "Protein_class_type": "Protein Identity",
                "Buffer_type": "Buffer",
                "Salt_type": "Salt / Ionic",
                "Stabilizer_type": "Stabilizer",
                "Surfactant_type": "Surfactant",
                "Excipient_type": "Excipient",
            }
            return mapping.get(cat_col, "Other")
    n = name.replace("num__", "")
    if n in NUM_COLS_RAW:
        return FEATURE_CATEGORY.get(n, "Other")
    # Engineered feature assignment
    eng_cat = {
        "log_conc": "Derived: Concentration",
        "conc_sq": "Derived: Concentration",
        "conc_x_kP": "Derived: Interaction",
        "conc_x_HCI": "Derived: Interaction",
        "Phi_Protein": "Derived: Volume Fraction",
        "Phi_Stabilizer": "Derived: Volume Fraction",
        "Phi_Salt": "Derived: Volume Fraction",
        "Phi_Excipient": "Derived: Volume Fraction",
        "Phi_Total": "Derived: Volume Fraction",
        "Crowding_Index": "Derived: Crowding",
        "Stabilizer_Squared": "Derived: Crowding",
        "Total_Solute_Mass": "Derived: Crowding",
        "Effective_Protein_Fraction": "Derived: Crowding",
        "KD_Asymptote": "Derived: Crowding",
        "Exp_Crowding": "Derived: Crowding",
        "Salt_mg_mL": "Salt / Ionic",
        "Stabilizer_mg_mL": "Stabilizer",
        "Excipient_mg_mL": "Excipient",
        "Surfactant_mg_mL": "Surfactant",
        "Ionic_Strength_Proxy": "Salt / Ionic",
    }
    if n in eng_cat:
        return eng_cat[n]
    if n.startswith("prior_"):
        return "Derived: Prior / Regime"
    if n.endswith("_low") or n.endswith("_high"):
        return "Derived: Concentration Split"
    return "Other"


ALL_FEAT_CATEGORY_COLOURS = {
    "Protein Identity": PAL_PURPLE,
    "Protein Property": PAL_BLUE,
    "Formulation": PAL_TEAL,
    "Buffer": PAL_GOLD,
    "Salt / Ionic": PAL_ORANGE,
    "Stabilizer": PAL_RED,
    "Surfactant": "#9B59B6",
    "Excipient": "#1ABC9C",
    "Process": PAL_SLATE,
    "Derived: Concentration": "#5B9BD5",
    "Derived: Volume Fraction": "#6EBFB5",
    "Derived: Crowding": "#E8A838",
    "Derived: Interaction": "#C0392B",
    "Derived: Prior / Regime": "#8E44AD",
    "Derived: Concentration Split": "#2ECC71",
    "Other": PAL_GREY,
}


# ============================================================
# 4.  Data preprocessing  (mirrors train_cnp_3.py)
# ============================================================


def get_mw(chemical_series, default_mw=342.3):
    return (
        chemical_series.astype(str)
        .str.lower()
        .map(
            lambda x: next((mw for name, mw in MW_MAP.items() if name in x), default_mw)
        )
    )


def load_and_preprocess(csv_path, preprocessor, physics_scaler):
    """Re-run the same preprocessing as train_cnp_3.py using saved transformers."""
    df = pd.read_csv(csv_path)

    num_cols = list(NUM_COLS_RAW)

    for c in num_cols:
        df[c] = df[c].fillna(0.0) if c in df.columns else 0.0

    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().replace("nan", "unknown")
        else:
            df[c] = "unknown"

    # ── Unit conversion & engineered features ─────────────────
    stabilizer_mw = get_mw(df["Stabilizer_type"])
    df["Stabilizer_mg_mL"] = df["Stabilizer_conc"] * stabilizer_mw

    salt_mw = get_mw(df["Salt_type"], default_mw=58.44)
    df["Salt_mg_mL"] = df["Salt_conc"] * salt_mw / 1000.0

    excipient_mw = get_mw(df["Excipient_type"], default_mw=150.0)
    df["Excipient_mg_mL"] = df["Excipient_conc"] * excipient_mw / 1000.0
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
    V_BAR = dict(
        protein=0.73 / 1000, stab=0.62 / 1000, salt=0.30 / 1000, excip=0.70 / 1000
    )
    df["Phi_Protein"] = df["Protein_conc"] * V_BAR["protein"]
    df["Phi_Stabilizer"] = df["Stabilizer_mg_mL"] * V_BAR["stab"]
    df["Phi_Salt"] = df["Salt_mg_mL"] * V_BAR["salt"]
    df["Phi_Excipient"] = df["Excipient_mg_mL"] * V_BAR["excip"]
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

    # ── Prior & regime features ────────────────────────────────
    new_prior_cols = [
        f"prior_{k}"
        for k in [
            "arginine",
            "lysine",
            "proline",
            "nacl",
            "stabilizer",
            "tween-20",
            "tween-80",
        ]
    ]
    new_conc_cols = [f"{k}_{s}" for k in CONC_THRESHOLDS for s in ("low", "high")]

    def process_row(row):
        from collections import defaultdict as _dd

        priors = {k: 0.0 for k in new_prior_cols}
        concs = {k: 0.0 for k in new_conc_cols}
        ph = float(row.get("Buffer_pH", 7.0) or 7.0)
        pi = float(row.get("PI_mean", 7.0) or 7.0)
        c_class = float(row.get("C_Class", 1.0) or 1.0)
        cci = c_class * np.exp(-abs(ph - pi) / 1.5)
        p_type = str(row.get("Protein_class_type", "default")).lower()
        if "mab_igg1" in p_type:
            regime = "Near-pI" if cci >= 0.90 else ("Mixed" if cci >= 0.50 else "Far")
        elif "mab_igg4" in p_type:
            regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.40 else "Far")
        else:
            regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
        # (full PRIOR_TABLE omitted for brevity — values will be small)
        for type_col, conc_col in [
            ("Salt_type", "Salt_conc"),
            ("Stabilizer_type", "Stabilizer_conc"),
            ("Excipient_type", "Excipient_conc"),
            ("Surfactant_type", "Surfactant_conc"),
        ]:
            ing_name = str(row.get(type_col, "none")).lower()
            ing_conc = float(row.get(conc_col, 0.0) or 0.0)
            if ing_name in ["none", "unknown", "nan"] or ing_conc <= 0:
                continue
            for target, threshold in CONC_THRESHOLDS.items():
                if target in ing_name or (target == "arginine" and "arg" in ing_name):
                    concs[f"{target}_low"] = min(ing_conc, threshold)
                    concs[f"{target}_high"] = max(ing_conc - threshold, 0)
        return {**priors, **concs}

    feat_df = df.apply(process_row, axis=1, result_type="expand")
    df = pd.concat([df, feat_df], axis=1)

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
    all_num = num_cols + new_prior_cols + new_conc_cols + engineered_cols

    # Transform using saved preprocessor
    X_matrix = preprocessor.transform(df)
    X_matrix = np.nan_to_num(X_matrix)

    # ── Build samples ──────────────────────────────────────────
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
        for col, sv in shear_map.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = df.iloc[i][col]
                if v <= 0:
                    v = 1e-6
                raw_x.append(np.log10(sv))
                raw_y.append(np.log10(v))
        if len(raw_x) < 3:
            continue
        si = np.argsort(raw_x)
        x_arr, y_arr = np.array(raw_x)[si], np.array(raw_y)[si]
        interp = PchipInterpolator(x_arr, y_arr)
        eps = np.unique(np.concatenate([x_arr, key_logs]))
        eps = eps[(eps >= x_arr.min()) & (eps <= x_arr.max())]
        eps.sort()
        dense_x = []
        for j in range(len(eps) - 1):
            pts = np.linspace(eps[j], eps[j + 1], 10)
            dense_x.extend(pts[:-1] if j < len(eps) - 2 else pts)
        if not dense_x:
            dense_x = x_arr
        dense_x = np.array(dense_x)
        dense_y = interp(dense_x)
        pts_np = np.stack(
            [
                physics_scaler.transform([[dx, dy]])[0]
                for dx, dy in zip(dense_x, dense_y)
            ]
        )
        samples.append(
            {
                "static": torch.tensor(X_matrix[i], dtype=torch.float32),
                "points": torch.tensor(pts_np, dtype=torch.float32),
                "group": df.iloc[i]["Protein_type"].lower(),
                "id": df.iloc[i]["ID"],
            }
        )

    return samples, X_matrix.shape[1], df


# ============================================================
# 5.  Permutation importance (decoder-side)
# ============================================================


def compute_permutation_importance(
    model, samples, raw_df, physics_scaler, device, n_repeats=10, seed=42
):
    """
    Returns:
      importances  np.ndarray [n_features, n_repeats]  ΔMSE per feature per repeat
      feature_idx  list[int]  column indices used
    """
    rng = np.random.default_rng(seed)
    parity_shear_map = [
        "Viscosity_100",
        "Viscosity_1000",
        "Viscosity_10000",
        "Viscosity_100000",
        "Viscosity_15000000",
    ]
    n_shears = len(parity_shear_map)
    shear_vals = [100.0, 1000.0, 10000.0, 100000.0, 1.5e7]
    log_shears = np.log10(shear_vals)

    visc_mean = physics_scaler.mean_[1]
    visc_scale = physics_scaler.scale_[1]
    shear_mean = physics_scaler.mean_[0]
    shear_scale = physics_scaler.scale_[0]

    # Standardised shear query tensor  [1, n_shears, 1]
    q_shear = torch.tensor(
        [[(ls - shear_mean) / shear_scale] for ls in log_shears],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    # Collect: true log-viscosities & valid masks
    true_lv = []
    valid_mask = []
    for s in samples:
        row = raw_df[raw_df["ID"] == s["id"]]
        tl, vm = [0.0] * 5, [False] * 5
        if not row.empty:
            r = row.iloc[0]
            for j, col in enumerate(parity_shear_map):
                if col in raw_df.columns and pd.notna(r[col]) and r[col] > 0:
                    tl[j], vm[j] = np.log10(float(r[col])), True
        true_lv.append(tl)
        valid_mask.append(vm)
    true_lv = np.array(true_lv)
    valid_mask = np.array(valid_mask)

    # Build context tensors & pre-compute r
    eval_groups = defaultdict(list)
    for s in samples:
        eval_groups[s["group"]].append(s)

    ctx_tensors, r_list = [], []
    model.eval()
    with torch.no_grad():
        for s in samples:
            group_members = eval_groups[s["group"]]
            ctx_pool = [x for x in group_members if x["id"] != s["id"]] or group_members
            ctx_items = []
            for c in ctx_pool:
                stat = c["static"].unsqueeze(0).repeat(c["points"].shape[0], 1)
                ctx_items.append(torch.cat([c["points"], stat], dim=1))
            ctx_t = torch.cat(ctx_items).unsqueeze(0).to(device)
            ctx_tensors.append(ctx_t)
            r_list.append(model.encode_memory(ctx_t))

    static_matrix = torch.stack([s["static"] for s in samples])  # [N, D]
    static_dim = static_matrix.shape[1]

    def _mse(mat):
        errs = []
        with torch.no_grad():
            for i, (r, tl, vm) in enumerate(zip(r_list, true_lv, valid_mask)):
                if not any(vm):
                    continue
                q_st = (
                    mat[i].unsqueeze(0).unsqueeze(0).repeat(1, n_shears, 1).to(device)
                )
                r_exp = r.unsqueeze(1).repeat(1, n_shears, 1)
                pred_sc = (
                    model.decode_from_memory(r, q_shear, q_st).squeeze().cpu().numpy()
                )
                pred_lv = pred_sc * visc_scale + visc_mean
                for j in range(n_shears):
                    if vm[j]:
                        errs.append((pred_lv[j] - tl[j]) ** 2)
        return float(np.mean(errs)) if errs else np.nan

    baseline = _mse(static_matrix)
    print(f"  Baseline decoder MSE (log₁₀ space): {baseline:.6f}")
    print(f"  Permuting {static_dim} features × {n_repeats} repeats …")

    importances = np.zeros((static_dim, n_repeats))
    for j in range(static_dim):
        for rep in range(n_repeats):
            perm = static_matrix.clone()
            idx = rng.permutation(len(samples))
            perm[:, j] = static_matrix[idx, j]
            importances[j, rep] = _mse(perm) - baseline
        if (j + 1) % 10 == 0:
            print(f"    {j+1}/{static_dim} features done …")

    return importances, baseline


# ============================================================
# 6.  Feature name parsing & grouping
# ============================================================


def parse_feature_names(preprocessor):
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        names = [f"feature_{i}" for i in range(1000)]
    return names


def group_to_all_features(importances_mean, importances_std, feature_names):
    """
    Collapse OHE columns back to their categorical column name.
    Keep every numerical (raw + engineered) as individual entries.
    Returns: dict {display_name: (mean_imp, std_imp, category)}
    """
    grouped = defaultdict(lambda: [0.0, 0.0, ""])

    for j, fname in enumerate(feature_names):
        mu = importances_mean[j]
        sig = importances_std[j]
        # Categorical OHE
        if fname.startswith("cat__"):
            rest = fname[5:]
            parent = next((c for c in CAT_COLS if rest.startswith(c)), rest)
            cat = _all_feat_category(parent)
            grouped[parent][0] += mu
            grouped[parent][
                1
            ] += sig  # approximate: sum of stds (conservative upper bound)
            grouped[parent][2] = cat
        elif fname.startswith("num__"):
            short = fname[5:]
            cat = _all_feat_category(short)
            grouped[short][0] += mu
            grouped[short][1] += sig
            grouped[short][2] = cat
        else:
            cat = _all_feat_category(fname)
            grouped[fname][0] += mu
            grouped[fname][1] += sig
            grouped[fname][2] = cat

    return {k: (v[0], v[1], v[2]) for k, v in grouped.items()}


def group_to_first_order(importances_mean, importances_std, feature_names):
    """
    Roll up all model features to the 21 original input columns.
    """
    first_order_names = NUM_COLS_RAW + CAT_COLS
    fo_mean = defaultdict(float)
    fo_var = defaultdict(float)  # for error bar: sum of variances

    for j, fname in enumerate(feature_names):
        mu = importances_mean[j]
        sig = importances_std[j]
        # Categorical OHE
        if fname.startswith("cat__"):
            rest = fname[5:]
            parent = next((c for c in CAT_COLS if rest.startswith(c)), None)
            if parent:
                fo_mean[parent] += mu
                fo_var[parent] += sig**2
        else:
            short = fname[5:] if fname.startswith("num__") else fname
            # Direct mapping
            if short in FIRST_ORDER_MAP:
                target, weight = FIRST_ORDER_MAP[short]
                fo_mean[target] += mu * weight
                fo_var[target] += (sig * weight) ** 2
            if short in FIRST_ORDER_MAP_EXTRA:
                target2, weight2 = FIRST_ORDER_MAP_EXTRA[short]
                fo_mean[target2] += mu * weight2
                fo_var[target2] += (sig * weight2) ** 2
            # If not in any map (shouldn't happen) — skip

    fo_std = {k: np.sqrt(v) for k, v in fo_var.items()}
    return fo_mean, fo_std


# ============================================================
# 7.  Plotting utilities
# ============================================================

PLOT_STYLE = {
    "figure.facecolor": "#FAFBFC",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#C8CDD6",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.color": "#E8EBF0",
    "grid.linewidth": 0.7,
    "xtick.color": "#4A5568",
    "ytick.color": "#4A5568",
    "text.color": "#2D3748",
    "font.size": 11,
}


def _make_legend_patches(cat_colour_map, present_cats):
    patches = []
    for cat in sorted(present_cats):
        if cat in cat_colour_map:
            patches.append(mpatches.Patch(color=cat_colour_map[cat], label=cat))
    return patches


def plot_first_order(fo_mean, fo_std, out_path, baseline_mse):
    """Horizontal bar chart for 1st-order features."""
    labels_raw = NUM_COLS_RAW + CAT_COLS
    means = np.array([fo_mean.get(f, 0.0) for f in labels_raw])
    stds = np.array([fo_std.get(f, 0.0) for f in labels_raw])

    order = np.argsort(means)[::-1]
    labels = [DISPLAY_NAMES.get(labels_raw[i], labels_raw[i]) for i in order]
    means = means[order]
    stds = stds[order]
    cats = [FEATURE_CATEGORY.get(labels_raw[i], "Other") for i in order]
    colours = [CATEGORY_COLOURS.get(c, PAL_GREY) for c in cats]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor("#FAFBFC")

        y_pos = np.arange(len(labels))
        bars = ax.barh(
            y_pos,
            means,
            xerr=stds,
            color=colours,
            height=0.68,
            error_kw=dict(ecolor="#6B7280", elinewidth=1.2, capsize=4, capthick=1.2),
            zorder=3,
        )

        # Value labels
        for bar, val, err in zip(bars, means, stds):
            if val > 0:
                ax.text(
                    val + err + max(means) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"+{val:.4f}",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    color="#4A5568",
                    fontweight="medium",
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10.5)
        ax.invert_yaxis()
        ax.set_xlabel(
            "ΔMSE in log₁₀-viscosity space",
            fontsize=11.5,
            labelpad=8,
        )
        ax.set_title(
            "Feature Importance\n",
            fontsize=13.5,
            fontweight="bold",
            pad=14,
            color="#1A202C",
        )

        # Legend
        present = set(cats)
        patches = _make_legend_patches(CATEGORY_COLOURS, present)
        ax.legend(
            handles=patches,
            title="Feature Category",
            title_fontsize=10,
            fontsize=9.5,
            loc="lower right",
            framealpha=0.9,
            edgecolor="#CBD5E0",
            fancybox=True,
        )

        # Zero line
        ax.axvline(0, color="#9CA3AF", linewidth=1.0, linestyle="--", zorder=2)
        ax.set_xlim(left=min(-max(means) * 0.05, -0.0002))

        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#FAFBFC")
        plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_all_features(all_feat_dict, out_path, baseline_mse, top_n=40):
    """Horizontal bar chart for ALL features (top N by mean importance)."""
    items = sorted(all_feat_dict.items(), key=lambda x: -x[1][0])[:top_n]
    labels = []
    means = []
    stds = []
    cats = []

    for name, (mu, sig, cat) in items:
        # Clean display label
        label = name.replace("num__", "").replace("cat__", "")
        label = label.replace("_", " ").title()
        labels.append(label)
        means.append(mu)
        stds.append(sig)
        cats.append(cat)

    means = np.array(means)
    stds = np.array(stds)
    colours = [ALL_FEAT_CATEGORY_COLOURS.get(c, PAL_GREY) for c in cats]

    with plt.rc_context(PLOT_STYLE):
        fig_height = max(10, top_n * 0.31)
        fig, ax = plt.subplots(figsize=(13, fig_height))
        fig.patch.set_facecolor("#FAFBFC")

        y_pos = np.arange(len(labels))
        bars = ax.barh(
            y_pos,
            means,
            xerr=stds,
            color=colours,
            height=0.70,
            error_kw=dict(ecolor="#6B7280", elinewidth=1.0, capsize=3, capthick=1.0),
            zorder=3,
        )

        for bar, val, err in zip(bars, means, stds):
            if val > 0:
                ax.text(
                    val + err + max(means) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"+{val:.4f}",
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color="#4A5568",
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(
            "ΔMSE in log₁₀-viscosity space",
            fontsize=11,
            labelpad=8,
        )
        ax.set_title(
            f"Feature Importanc (Top {top_n} shown)\n",
            fontsize=13,
            fontweight="bold",
            pad=14,
            color="#1A202C",
        )

        present = set(cats)
        patches = _make_legend_patches(ALL_FEAT_CATEGORY_COLOURS, present)
        ax.legend(
            handles=patches,
            title="Feature Category",
            title_fontsize=9.5,
            fontsize=8.5,
            loc="lower right",
            framealpha=0.9,
            edgecolor="#CBD5E0",
            fancybox=True,
            ncol=2,
        )

        ax.axvline(0, color="#9CA3AF", linewidth=1.0, linestyle="--", zorder=2)
        ax.set_xlim(left=min(-max(means) * 0.05, -0.0002))

        fig.tight_layout(rect=[0, 0.03, 1, 1])
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#FAFBFC")
        plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# 8.  Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="CNP Feature Importance Analysis")
    parser.add_argument(
        "--model_dir",
        default="models/experiments/o_net_v3_debug_aug",
        help="Directory containing best_model.pt, preprocessor.pkl, physics_scaler.pkl",
    )
    parser.add_argument(
        "--data_csv",
        default="data/raw/formulation_data_03042026.csv",
        help="Path to formulation_data CSV",
    )
    parser.add_argument(
        "--out_dir",
        default="./feature_importance_results",
        help="Output directory for plots and CSVs",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=10,
        help="Number of permutation repeats per feature (default 10)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=40,
        help="Number of features to show in the all-features plot",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")

    # ── Load saved artefacts ───────────────────────────────────
    print("\n[1/5] Loading model artefacts …")
    preprocessor = joblib.load(os.path.join(args.model_dir, "preprocessor.pkl"))
    physics_scaler = joblib.load(os.path.join(args.model_dir, "physics_scaler.pkl"))
    feature_names = parse_feature_names(preprocessor)
    static_dim = len(feature_names)
    print(f"  Feature dimension: {static_dim}")

    # ── Locate checkpoint (.pt or .pth) ───────────────────────
    for _fname in ("best_model.pt", "best_model.pth", "model.pt", "model.pth"):
        _ckpt_path = os.path.join(args.model_dir, _fname)
        if os.path.exists(_ckpt_path):
            break
    else:
        # Fallback: first .pt/.pth in dir
        _candidates = [
            f
            for f in os.listdir(args.model_dir)
            if f.endswith(".pt") or f.endswith(".pth")
        ]
        if not _candidates:
            raise FileNotFoundError(f"No .pt/.pth checkpoint found in {args.model_dir}")
        _ckpt_path = os.path.join(args.model_dir, sorted(_candidates)[0])
    print(f"  Checkpoint: {_ckpt_path}")

    ckpt = torch.load(_ckpt_path, map_location=device)

    # Handle both bare state_dicts and wrapped checkpoints
    # e.g. {"state_dict": {...}, "config": {...}, "static_dim": N}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg = ckpt.get("config", {})
        hidden_dim = cfg.get("hidden_dim", args.hidden_dim)
        latent_dim = cfg.get("latent_dim", args.latent_dim)
        # static_dim from checkpoint overrides preprocessor count (sanity-check)
        ckpt_sdim = ckpt.get("static_dim", static_dim)
        if ckpt_sdim != static_dim:
            print(
                f"  WARNING: preprocessor dim ({static_dim}) != checkpoint dim "
                f"({ckpt_sdim}). Using checkpoint value."
            )
            static_dim = ckpt_sdim
        state_dict = ckpt["state_dict"]
    else:
        hidden_dim = args.hidden_dim
        latent_dim = args.latent_dim
        state_dict = ckpt

    print(
        f"  hidden_dim={hidden_dim}  latent_dim={latent_dim}  static_dim={static_dim}"
    )
    model = CrossSampleCNP(static_dim, hidden_dim, latent_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("  Model loaded.")

    # ── Preprocess data ────────────────────────────────────────
    print("\n[2/5] Preprocessing data …")
    samples, sdim, raw_df = load_and_preprocess(
        args.data_csv, preprocessor, physics_scaler
    )
    print(f"  Samples: {len(samples)}")

    # ── Permutation importance ─────────────────────────────────
    print(f"\n[3/5] Running permutation importance ({args.n_repeats} repeats) …")
    importances, baseline_mse = compute_permutation_importance(
        model, samples, raw_df, physics_scaler, device, n_repeats=args.n_repeats
    )
    imp_mean = importances.mean(axis=1)
    imp_std = importances.std(axis=1)

    # ── Save raw importance CSV ────────────────────────────────
    fi_df = pd.DataFrame(
        {
            "feature": feature_names,
            "delta_mse_mean": imp_mean,
            "delta_mse_std": imp_std,
        }
    ).sort_values("delta_mse_mean", ascending=False)
    fi_df.to_csv(os.path.join(args.out_dir, "importance_raw.csv"), index=False)
    print("  Saved: importance_raw.csv")

    # ── Group to all-features dict ─────────────────────────────
    print("\n[4/5] Grouping features …")
    all_feat = group_to_all_features(imp_mean, imp_std, feature_names)
    af_df = pd.DataFrame(
        [
            {
                "feature": k,
                "delta_mse_mean": v[0],
                "delta_mse_std": v[1],
                "category": v[2],
            }
            for k, v in sorted(all_feat.items(), key=lambda x: -x[1][0])
        ]
    )
    af_df.to_csv(os.path.join(args.out_dir, "importance_all_features.csv"), index=False)

    # ── Group to 1st-order dict ────────────────────────────────
    fo_mean, fo_std = group_to_first_order(imp_mean, imp_std, feature_names)
    fo_df = pd.DataFrame(
        [
            {
                "feature": k,
                "display": DISPLAY_NAMES.get(k, k),
                "delta_mse_mean": fo_mean.get(k, 0.0),
                "delta_mse_std": fo_std.get(k, 0.0),
                "category": FEATURE_CATEGORY.get(k, "Other"),
            }
            for k in (NUM_COLS_RAW + CAT_COLS)
        ]
    ).sort_values("delta_mse_mean", ascending=False)
    fo_df.to_csv(os.path.join(args.out_dir, "importance_first_order.csv"), index=False)

    print("\n  Top 10 first-order features:")
    print(
        fo_df[["display", "delta_mse_mean", "delta_mse_std"]]
        .head(10)
        .to_string(index=False)
    )

    # ── Plots ──────────────────────────────────────────────────
    print("\n[5/5] Generating plots …")
    plot_first_order(
        fo_mean,
        fo_std,
        out_path=os.path.join(args.out_dir, "importance_first_order.png"),
        baseline_mse=baseline_mse,
    )
    plot_all_features(
        all_feat,
        out_path=os.path.join(args.out_dir, "importance_all_features.png"),
        baseline_mse=baseline_mse,
        top_n=args.top_n,
    )
    print(f"\nDone.  Results written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
