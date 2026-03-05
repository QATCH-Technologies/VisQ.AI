"""
train_o_net_v3.py
=================
Improved training script for CrossSampleCNP.

Changes from train_o_net_no_phys.py (each labeled with WHY):

  [FIX-1] AttentionPool: Add LayerNorm after attention output.
          WHY: Without normalization the pooler output scale drifts during
          training, which destabilizes the decoder and makes the latent
          harder to learn from. LayerNorm stabilizes without suppressing info.

  [FIX-2] REMOVE latent_reg = torch.mean(r**2).
          WHY: This L2 penalty is the root cause of context collapse. It trains
          the model to keep r ≈ 0, so the decoder learns to ignore r and route
          everything through static features. The flat few-shot curve is a direct
          symptom of this. Removing it is the single most impactful change.

  [FIX-3] Add contrastive latent loss (inter-group repulsion).
          WHY: After removing the suppressive L2 penalty, we need a positive
          signal that forces r to carry group-discriminative information.
          Each training step now processes TWO groups. We minimize cosine
          similarity between their latent vectors r_A and r_B. This directly
          teaches the pooler that different proteins should have different r.

  [FIX-4] Add within-group latent consistency loss.
          WHY: The other half of the contrastive objective. Two random subsets
          of the same protein group's context should produce similar r. Without
          this, the contrastive loss alone could encourage r to encode
          arbitrary noise rather than meaningful protein-level structure.

  [FIX-5] Add context utility loss (null context baseline).
          WHY: Even with [FIX-3/4], the decoder could ignore r if it finds a
          shortcut through static features. This loss computes a null prediction
          using a zero-filled context vector and penalizes if the real context
          does NOT produce a lower MSE. It creates a direct gradient signal
          saying "your context must be worth something."

  [FIX-6] Hard group oversampling via EMA difficulty weights.
          WHY: Pembrolizumab (RMSE=0.75) and trastuzumab (RMSE=0.41) have 3-4x
          higher error than other groups but are sampled with equal probability.
          We maintain an exponential moving average of per-group loss and
          oversample groups proportional to their difficulty. This ensures hard
          groups receive more gradient updates without discarding any data.

  [FIX-7] Replace GroupShuffleSplit with group-held-out CV in Optuna.
          WHY: With only ~14 protein groups, GroupShuffleSplit(test_size=0.25)
          may put similar groups in both folds across repeated splits. True
          group-held-out CV holds out entire protein types, which is what you
          actually care about at deployment. This gives honest hyperparameter
          selection rather than an optimistic in-distribution signal.

  [FIX-8] Randomize validation context selection.
          WHY: The original validate() always used the first half of sorted
          samples as context. This is a deterministic and ordering-dependent
          split. Randomizing gives a less biased validation signal.

  [FIX-9] Latent variance diagnostic logging.
          WHY: After [FIX-2], the key question is whether r is now actually
          varying across groups. Each epoch logs mean inter-group latent L2
          distance. If this stays near zero, context collapse is still
          occurring via a different pathway and needs further investigation.

All preprocessing (load_and_preprocess) is UNCHANGED so existing saved
preprocessors and physics_scalers remain compatible.
"""

import copy
import os
from collections import defaultdict

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 0. Protein class map for hard negative mining (FIX-A)
# ==========================================
# Groups in the same class are "hard negatives" — they share similar static
# features and are the most important pairs for the contrastive loss to
# distinguish. We also exclude non-protein (buffer-only) groups from
# contrastive sampling so their easy protein-vs-buffer signal doesn't dominate.

# Lowercase keys matching the "group" field produced by load_and_preprocess
PROTEIN_CLASS_MAP = {
    "adalimumab": "igg1",
    "bevacizumab": "igg1",
    "trastuzumab": "igg1",
    "pembrolizumab": "igg4",
    "ibalizumab": "igg4",
    "nivolumab": "igg4",
    "belatacept": "fc_fusion",
    "etanercept": "fc_fusion",
    "vudalimab": "bispecific",
    "poly-higg": "polyclonal",
    "bgg": "polyclonal",
    "bsa": "other",
}
# Groups excluded from contrastive/consistency sampling — they don't represent
# a specific protein type and would produce trivially easy negative examples.
NON_PROTEIN_GROUPS = {"none"}


# ==========================================
# 1. Model Architecture
# ==========================================


class AttentionPool(nn.Module):
    def __init__(self, latent_dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        # [FIX-1 REINSTATED] We must constrain the magnitude of r so the
        # contrastive objective uses direction rather than explosive norm scaling.
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        # Apply LayerNorm before returning
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

    def forward(self, context_tensor, query_shear, query_static):
        encoded = self.encoder(context_tensor)
        r = self.pooler(encoded)
        n_queries = query_shear.size(1)
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)
        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)

    def encode_memory(self, context_tensor):
        return self.pooler(self.encoder(context_tensor))

    def decode_from_memory(self, memory_vector, query_shear, query_static):
        n_queries = query_shear.size(1)
        r_expanded = memory_vector.unsqueeze(1).repeat(1, n_queries, 1)
        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)


# ==========================================
# 2. Data Pipeline (UNCHANGED)
# ==========================================
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


def load_and_preprocess(csv_path, save_dir=None):
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
    # Assuming CONC_THRESHOLDS is globally defined
    for k in CONC_THRESHOLDS.keys():
        new_conc_cols.append(f"{k}_low")
        new_conc_cols.append(f"{k}_high")

    print("Normalizing units to mg/mL and calculating Physics Features...")

    # Define approximate molecular weights (g/mol) for conversions
    # You can expand these dictionaries based on the exact chemicals in your lab
    MW_MAP = {
        "sucrose": 342.3,
        "trehalose": 342.3,
        "arginine": 174.2,
        "proline": 115.13,
        "lysine": 149.19,
        "nacl": 58.44,
        "default_sugar": 342.3,
    }

    def get_mw(chemical_series, default_mw=342.3):
        return (
            chemical_series.astype(str)
            .str.lower()
            .map(
                lambda x: next(
                    (mw for name, mw in MW_MAP.items() if name in x), default_mw
                )
            )
        )

    # 1. Convert everything to mg/mL
    # Stabilizer (M -> mg/mL): M * MW
    stabilizer_mw = get_mw(df["Stabilizer_type"], default_mw=342.3)
    df["Stabilizer_mg_mL"] = df["Stabilizer_conc"] * stabilizer_mw

    # Salt (mM -> mg/mL): mM * MW / 1000
    salt_mw = get_mw(df["Salt_type"], default_mw=58.44)  # Default to NaCl
    df["Salt_mg_mL"] = (df["Salt_conc"] * salt_mw) / 1000.0

    # Excipient (mM -> mg/mL): mM * MW / 1000
    excipient_mw = get_mw(
        df["Excipient_type"], default_mw=150.0
    )  # generic amino acid guess
    df["Excipient_mg_mL"] = (df["Excipient_conc"] * excipient_mw) / 1000.0
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
    V_BAR_PROTEIN = 0.73 / 1000.0
    V_BAR_SUCROSE = 0.62 / 1000.0
    V_BAR_SALT = 0.30 / 1000.0
    V_BAR_AMINO = 0.70 / 1000.0

    df["Phi_Protein"] = df["Protein_conc"] * V_BAR_PROTEIN
    df["Phi_Stabilizer"] = df["Stabilizer_mg_mL"] * V_BAR_SUCROSE
    df["Phi_Salt"] = df["Salt_mg_mL"] * V_BAR_SALT
    df["Phi_Excipient"] = df["Excipient_mg_mL"] * V_BAR_AMINO

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

    # ---------------------------------------------------------
    # Priors and Regimes Mapping
    # ---------------------------------------------------------
    def process_row_features(row):
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

        regime = "Far"
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
        # Assuming PRIOR_TABLE is globally defined
        for key in PRIOR_TABLE.keys():
            if key != "default" and key in p_type:
                lookup_key = key
                break
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
                match = (target_ing in ing_name) or (
                    target_ing == "arginine" and "arg" in ing_name
                )
                if match:
                    concs[f"{target_ing}_low"] = min(ing_conc, threshold)
                    concs[f"{target_ing}_high"] = max(ing_conc - threshold, 0)

        return {**priors, **concs}

    print("Calculating Regimes and Concentration Splits...")
    features_df = df.apply(process_row_features, axis=1, result_type="expand")
    df = pd.concat([df, features_df], axis=1)

    # [FIX] Append all dynamically generated columns so ColumnTransformer processes them
    num_cols.extend(new_prior_cols)
    num_cols.extend(new_conc_cols)
    num_cols.extend(engineered_cols)

    # ---------------------------------------------------------
    # Preprocessor and Scaling
    # ---------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    X_matrix = preprocessor.fit_transform(df)
    if np.isnan(X_matrix).any():
        print("WARNING: NaNs found in X_matrix after preprocessing! Replacing with 0.")
        X_matrix = np.nan_to_num(X_matrix)

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

    # ---------------------------------------------------------
    # Target Resampling Logic
    # ---------------------------------------------------------
    samples = []
    key_shears = [100.0, 1000.0, 10000.0, 100000.0, 15000000.0]
    key_logs = np.log10(key_shears)

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
            interval_pts = np.linspace(
                interval_endpoints[j], interval_endpoints[j + 1], 10
            )
            if j < len(interval_endpoints) - 2:
                dense_x_list.append(interval_pts[:-1])
            else:
                dense_x_list.append(interval_pts)

        if len(dense_x_list) > 0:
            dense_x = np.concatenate(dense_x_list)
        else:
            dense_x = x_arr

        dense_y = interpolator(dense_x)
        pts = []
        for dx, dy in zip(dense_x, dense_y):
            scaled_point = physics_scaler.transform(np.array([[dx, dy]]))[0]
            pts.append(scaled_point)

        if pts:
            pts_np = np.stack(pts)
            samples.append(
                {
                    "static": torch.tensor(X_matrix[i], dtype=torch.float32),
                    "points": torch.tensor(pts_np, dtype=torch.float32),
                    "group": df.iloc[i]["Protein_type"],
                    "id": df.iloc[i]["ID"],
                }
            )

    return samples, X_matrix.shape[1]


# ==========================================
# 3. Training Helpers
# ==========================================


def _build_ctx_tensor(task_samples, indices, device):
    """Build a context tensor [1, N_points, 2+static_dim] from sample indices."""
    ctx_items = []
    for i in indices:
        s = task_samples[i]
        stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
        ctx_items.append(torch.cat([s["points"], stat], dim=1))
    return torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)


def _build_tgt_tensors(task_samples, indices, device):
    """Build query tensors for target samples."""
    shear_list, y_list, stat_list = [], [], []
    for i in indices:
        s = task_samples[i]
        n = s["points"].shape[0]
        shear_list.append(s["points"][:, [0]])
        y_list.append(s["points"][:, [1]])
        stat_list.append(s["static"].unsqueeze(0).repeat(n, 1))
    if not shear_list:
        return None, None, None
    q_x = torch.cat(shear_list, dim=0).unsqueeze(0).to(device)
    q_stat = torch.cat(stat_list, dim=0).unsqueeze(0).to(device)
    q_y = torch.cat(y_list, dim=0).unsqueeze(0).to(device)
    return q_x, q_stat, q_y


def train_epoch(
    model,
    samples,
    optimizer,
    device,
    iterations=100,
    group_weights=None,
    lambda_triplet=0.30,
    # [FIX-CONSISTENCY] Reduced slightly from 0.20 to avoid over-constraining the encoder
    lambda_consistency=0.10,
    lambda_utility=2.5,
    triplet_margin=3.0,
    # [FIX-NORM] Soft norm penalty parameters
    lambda_norm=0.05,
    norm_target=5.0,
):
    model.train()
    total_loss = 0
    count = 0

    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    all_protein_list = [
        g for g, sl in groups.items() if len(sl) >= 4 and g not in NON_PROTEIN_GROUPS
    ]
    protein_list = [g for g, sl in groups.items() if len(sl) >= 4]

    if group_weights is not None:
        raw_w = np.array([group_weights.get(g, 1.0) for g in protein_list], dtype=float)
    else:
        raw_w = np.ones(len(protein_list), dtype=float)
    sampling_probs = raw_w / raw_w.sum()

    group_loss_accum = defaultdict(float)
    group_loss_count = defaultdict(int)

    for _ in range(iterations):
        if len(protein_list) < 2:
            continue
        idx_anchor = np.random.choice(len(protein_list), p=sampling_probs)
        prot_A = protein_list[idx_anchor]
        task_A = groups[prot_A]

        idx_A = np.random.permutation(len(task_A))
        n_ctx_A = np.random.randint(1, min(12, len(idx_A) - 1))
        ctx_A = _build_ctx_tensor(task_A, idx_A[:n_ctx_A], device)
        qx_A, qstat_A, qy_A = _build_tgt_tensors(task_A, idx_A[n_ctx_A:], device)
        if qx_A is None:
            continue

        # Main forward pass (with masking to force r-dependence)
        if np.random.random() < 0.60:
            mask = torch.bernoulli(torch.full_like(qstat_A, 0.5))
            qstat_A_in = qstat_A * mask
        else:
            qstat_A_in = qstat_A

        pred_A = model(ctx_A, qx_A, qstat_A_in)
        mse_loss = F.mse_loss(pred_A, qy_A)

        # ---- Context utility loss ----
        with torch.no_grad():
            pred_null = model(torch.zeros_like(ctx_A), qx_A, qstat_A)
        mse_null = F.mse_loss(pred_null, qy_A).detach()

        pred_ctx_unmasked = model(ctx_A, qx_A, qstat_A)
        mse_ctx_unmasked = F.mse_loss(pred_ctx_unmasked, qy_A)

        utility_loss = torch.clamp(mse_ctx_unmasked - mse_null + 1e-3, min=0.0)

        # ---- [FIX-PRIORITY 1] Unconditional Soft Norm Penalty ----
        # Calculate norm penalty on the actual context used for the forward pass,
        # unconditionally applying it to all groups (including "none").
        r_current = model.encode_memory(ctx_A)
        r_norm_current = torch.norm(r_current, p=2, dim=-1)
        norm_penalty = torch.mean(
            torch.clamp(r_norm_current - norm_target, min=0.0) ** 2
        )

        # ---- Triplet loss & Consistency Loss ----
        triplet_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)

        if prot_A in all_protein_list and len(all_protein_list) >= 2:
            perm_full = np.random.permutation(len(task_A))
            half = max(1, len(perm_full) // 2)

            ctx_anchor = _build_ctx_tensor(task_A, perm_full[:half], device)
            ctx_pos = _build_ctx_tensor(task_A, perm_full[half:], device)

            # Pre-pooled mean consistency loss
            enc_anchor_mean = model.encoder(ctx_anchor).mean(dim=1)
            enc_pos_mean = model.encoder(ctx_pos).mean(dim=1)
            cos_within = F.cosine_similarity(enc_anchor_mean, enc_pos_mean, dim=-1)
            consistency_loss = (1.0 - cos_within).mean()

            r_anchor = model.encode_memory(ctx_anchor)
            r_pos = model.encode_memory(ctx_pos)

            # (Note: norm_penalty calculation removed from here)

            class_A = PROTEIN_CLASS_MAP.get(prot_A, "unknown")
            same_class_negs = [
                g
                for g in all_protein_list
                if g != prot_A and PROTEIN_CLASS_MAP.get(g, "") == class_A
            ]
            diff_class_negs = [g for g in all_protein_list if g != prot_A]

            # ... [Keep the rest of the triplet loss calculations the same] ...

            if same_class_negs and np.random.random() < 0.70:
                prot_B = np.random.choice(same_class_negs)
            elif diff_class_negs:
                prot_B = np.random.choice(diff_class_negs)
            else:
                prot_B = prot_A

            task_B = groups[prot_B]
            idx_B = np.random.permutation(len(task_B))
            n_ctx_B = np.random.randint(1, min(8, len(idx_B)))
            r_neg = model.encode_memory(
                _build_ctx_tensor(task_B, idx_B[:n_ctx_B], device)
            )

            d_pos = torch.sum((r_anchor - r_pos) ** 2, dim=-1).sqrt()
            d_neg = torch.sum((r_anchor - r_neg) ** 2, dim=-1).sqrt()

            triplet_loss = torch.clamp(d_pos - d_neg + triplet_margin, min=0.0).mean()

        # ---- Combined loss ----
        loss = (
            mse_loss
            + lambda_utility * utility_loss
            + lambda_triplet * triplet_loss
            + lambda_consistency * consistency_loss
            + lambda_norm * norm_penalty  # [ADDED]
        )

        if torch.isnan(loss):
            print("Warning: NaN loss encountered. Skipping batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

        group_loss_accum[prot_A] += mse_loss.item()
        group_loss_count[prot_A] += 1

    per_group_mse = {
        g: group_loss_accum[g] / group_loss_count[g]
        for g in group_loss_accum
        if group_loss_count[g] > 0
    }
    return total_loss / max(1, count), per_group_mse


def validate(model, samples, device, n_repeats=3):
    """
    [FIX-8] Randomized context selection during validation.
    Runs n_repeats random splits per group and averages, removing the
    ordering bias from the original fixed first-half-as-context approach.
    """
    model.eval()
    total_error = 0
    count = 0
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    with torch.no_grad():
        for prot, task_samples in groups.items():
            if len(task_samples) < 2:
                continue
            group_errors = []
            for _ in range(n_repeats):
                # [FIX-8] Randomize which samples serve as context vs target
                idx = np.random.permutation(len(task_samples))
                mid = max(1, len(idx) // 2)
                ctx_idx = idx[:mid]
                tgt_idx = idx[mid:]
                if len(tgt_idx) == 0:
                    continue

                ctx_list = []
                for i in ctx_idx:
                    s = task_samples[i]
                    stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                    ctx_list.append(torch.cat([s["points"], stat], dim=1))
                ctx_tensor = torch.cat(ctx_list, dim=0).unsqueeze(0).to(device)

                tgt_shear, tgt_y, tgt_stat = [], [], []
                for i in tgt_idx:
                    s = task_samples[i]
                    tgt_shear.append(s["points"][:, [0]])
                    tgt_y.append(s["points"][:, [1]])
                    tgt_stat.append(
                        s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                    )

                q_x = torch.cat(tgt_shear, dim=0).unsqueeze(0).to(device)
                q_stat = torch.cat(tgt_stat, dim=0).unsqueeze(0).to(device)
                true_y = torch.cat(tgt_y, dim=0).unsqueeze(0).to(device)
                pred = model(ctx_tensor, q_x, q_stat)
                loss = F.mse_loss(pred, true_y)
                if not torch.isnan(loss):
                    group_errors.append(loss.item())

            if group_errors:
                total_error += np.mean(group_errors)
                count += 1

    return total_error / max(1, count)


def log_latent_variance(model, samples, device):
    """
    [FIX-9 / FIX-B] Compute protein-only inter-group latent L2 distance.

    IMPORTANT: buffer-only groups (e.g. "none") are excluded from this metric.
    Including them was producing a misleadingly high separation ratio in v2
    because the protein-vs-buffer distinction is trivially easy and was masking
    the fact that protein-protein discrimination was near-zero.

    Returns:
        protein_separation: mean pairwise L2 between protein group centroids
                            (the number that should be growing during training)
    """
    model.eval()
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    group_r = {}
    with torch.no_grad():
        for prot, task_samples in groups.items():
            if len(task_samples) < 2:
                continue
            # [FIX-B] Skip non-protein groups — their easy separability
            # was inflating the v2 separation ratio to a false 18×
            if prot in NON_PROTEIN_GROUPS:
                continue
            idx = np.random.permutation(len(task_samples))[: min(5, len(task_samples))]
            ctx_items = []
            for i in idx:
                s = task_samples[i]
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_items.append(torch.cat([s["points"], stat], dim=1))
            ctx_t = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)
            r = model.encode_memory(ctx_t).squeeze(0).cpu().numpy()
            group_r[prot] = r

    if len(group_r) < 2:
        return 0.0

    vecs = np.stack(list(group_r.values()))  # [n_protein_groups, latent_dim]
    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(np.linalg.norm(vecs[i] - vecs[j]))
    return float(np.mean(dists))


def objective_cv(trial, samples, static_dim, device):
    # [KEPT] Force wider bottleneck to prevent latent routing failure.
    hidden_dim = trial.suggest_int("hidden_dim", 128, 256, step=64)
    latent_dim = trial.suggest_int("latent_dim", 128, 256, step=64)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    hard_groups = ["etanercept", "vudalimab", "pembrolizumab", "ibalizumab"]
    medium_groups = ["adalimumab", "poly-higg", "nivolumab"]
    priority_held_out = [
        g for g in hard_groups + medium_groups if any(s["group"] == g for s in samples)
    ]
    held_out_groups = priority_held_out[:6]

    fold_scores = []
    for fold_idx, held_out in enumerate(held_out_groups):
        train_fold = [s for s in samples if s["group"] != held_out]
        val_fold = [s for s in samples if s["group"] == held_out]

        if len(val_fold) < 2:
            continue

        model = CrossSampleCNP(static_dim, hidden_dim, latent_dim, dropout).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        for epoch in range(40):
            train_loss, _ = train_epoch(
                model, train_fold, optimizer, device, iterations=50
            )
            val_loss = validate(model, val_fold, device, n_repeats=2)
            trial.report(val_loss, fold_idx * 40 + epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        fold_scores.append(validate(model, val_fold, device, n_repeats=3))

    return float(np.mean(fold_scores)) if fold_scores else float("inf")


# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    data = "data/raw/formulation_data_03042026.csv"
    # data = "data/processed/augmented_formulation_data.csv"
    out = "./models/experiments/o_net_v3_debug_no_aug"
    trials = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_params = {
        # --- Model Architecture ---
        "hidden_dim": 128,  # Wide enough to capture non-linear interactions between pH/Conc/Salt
        "latent_dim": 64,  # MUST be divisible by n_heads=4. 64 is plenty to encode rheological physics without overfitting
        "dropout": 0.15,  # Modest dropout to prevent memorizing the ~1500 samples
        # --- Training/Optimizer (for your training loop) ---
        "lr": 5e-4,  # Slightly lower than standard 1e-3 to allow stable attention pooling
        "weight_decay": 1e-4,  # L2 regularization to keep the latent representation smooth
        "batch_size": 32,  # Small enough to provide noisy gradients, large enough for stable contrastive learning
        "epochs": 150,  # Given the small dataset, 100-200 epochs is usually the sweet spot
    }
    samples, static_dim = load_and_preprocess(data, save_dir=out)
    print(
        f"Loaded {len(samples)} samples from "
        f"{len(set(s['group'] for s in samples))} protein groups."
    )
    if trials > 0:
        print("Starting Group-Held-Out Optuna Optimization...")

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        study.optimize(
            lambda t: objective_cv(t, samples, static_dim, device),
            n_trials=trials,
        )

        print("\n--- Tuning Complete ---")
        print("Best params:", study.best_params)
        best_params = study.best_params
    else:
        print("Skipping hyperparameter tuning since trials=0. Using default params.")
        print("Default params:", best_params)
    # ==========================================
    # FINAL RETRAINING
    # ==========================================
    print("\nRetraining final model on ALL data...")

    final_model = CrossSampleCNP(
        static_dim,
        hidden_dim=best_params["hidden_dim"],
        latent_dim=best_params["latent_dim"],
        dropout=best_params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=25,
    )

    best_loss = float("inf")
    patience_counter = 0
    patience_limit = 80
    best_state = None

    # [FIX-WATCHLIST] Stratified 10% split. Hold out a few samples from every group
    # instead of holding out entire groups.
    final_train_set = []
    final_stop_set = []

    # Group samples to stratify
    groups_dict = defaultdict(list)
    for s in samples:
        groups_dict[s["group"]].append(s)

    for g, g_samples in groups_dict.items():
        # Shuffle samples within the group
        np.random.shuffle(g_samples)
        # Hold out 10% for validation (minimum of 1 if there are enough samples)
        n_val = max(1, int(len(g_samples) * 0.1))

        if len(g_samples) < 2:
            # If a group only has 1 sample, it must go to training
            final_train_set.extend(g_samples)
        else:
            final_stop_set.extend(g_samples[:n_val])
            final_train_set.extend(g_samples[n_val:])

    print(
        f"Final Train: {len(final_train_set)} samples | "
        f"Early Stop Watchlist (stratified 10%): {len(final_stop_set)} samples"
    )

    group_weights = {g: 1.0 for g in set(s["group"] for s in final_train_set)}
    ema_alpha = 0.3

    for ep in range(500):
        train_loss, per_group_mse = train_epoch(
            final_model,
            final_train_set,
            optimizer,
            device,
            iterations=100,
            group_weights=group_weights,
        )

        for g, mse in per_group_mse.items():
            group_weights[g] = ema_alpha * mse + (1 - ema_alpha) * group_weights[g]
        total_w = sum(group_weights.values())
        n_g = len(group_weights)
        for g in group_weights:
            group_weights[g] = group_weights[g] / total_w * n_g

        # [FIX-MONITORING] Increase n_repeats to 10 for the watchlist evaluation to heavily reduce noise
        val_loss = validate(final_model, final_stop_set, device, n_repeats=10)
        scheduler.step(val_loss)

        if ep % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            latent_var = log_latent_variance(final_model, final_train_set, device)

            # [FIX-PRIORITY 2] Track both Pembro Norm AND Intra-Group Spread
            pembro_samples = [
                s for s in final_train_set if s["group"] == "pembrolizumab"
            ]
            pembro_norm_str = "N/A"
            pembro_spread_str = "N/A"

            if len(pembro_samples) > 1:
                final_model.eval()
                with torch.no_grad():
                    idx = np.random.permutation(len(pembro_samples))[
                        : min(10, len(pembro_samples))
                    ]
                    r_list = []
                    for i in idx:
                        s = pembro_samples[i]
                        stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                        ctx_item = (
                            torch.cat([s["points"], stat], dim=1)
                            .unsqueeze(0)
                            .to(device)
                        )
                        r_list.append(final_model.encode_memory(ctx_item))

                    if r_list:
                        r_pembro = torch.cat(r_list, dim=0)  # [N, latent_dim]
                        pembro_norm = torch.norm(r_pembro, p=2, dim=-1).mean().item()
                        pembro_norm_str = f"{pembro_norm:.3f}"

                        # Calculate pairwise distance for intra-spread
                        dists = []
                        for i in range(len(r_pembro)):
                            for j in range(i + 1, len(r_pembro)):
                                dists.append(
                                    torch.norm(r_pembro[i] - r_pembro[j], p=2).item()
                                )
                        if dists:
                            pembro_spread = np.mean(dists)
                            pembro_spread_str = f"{pembro_spread:.3f}"
                final_model.train()  # Set back to train mode

            top_hard = sorted(group_weights.items(), key=lambda x: -x[1])[:3]
            hard_str = ", ".join(f"{g}:{w:.2f}" for g, w in top_hard)

            print(
                f"Epoch {ep:3d}: Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"LR {current_lr:.2e} | LatentVar {latent_var:.3f} | "
                f"Pembro [Norm: {pembro_norm_str} | Spread: {pembro_spread_str}] | "
                f"Top hard: [{hard_str}]"
            )

            # ... [Keep your latent variance warning logic the same] ...

            if ep >= 30 and latent_var < 0.2:
                print(
                    f"  *** WARNING: LatentVar={latent_var:.3f} is very low. "
                    "Context collapse may still be occurring. Consider increasing "
                    "lambda_contrastive or checking that multiple protein groups "
                    "are present in the training set. ***"
                )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(final_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Stopping early at epoch {ep}. Best Val Loss: {best_loss:.4f}")
            break

    if best_state is not None:
        final_model.load_state_dict(best_state)

    save_path = os.path.join(out, "best_model.pth")
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "config": best_params,
            "static_dim": static_dim,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")
    print(
        f"Final group difficulty weights: {dict(sorted(group_weights.items(), key=lambda x: -x[1]))}"
    )
