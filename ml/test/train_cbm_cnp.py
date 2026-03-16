"""
train_cbm_cnp.py
================
Concept Bottleneck Machine (CBM) extension of CrossSampleCNP.

All [FIX-N] changes from train_cnp_3.py are preserved UNCHANGED.
New additions are labeled [CBM-N].

  [CBM-1] ConceptBottleneckCNP: hard concept bottleneck architecture.
          The latent vector r (latent_dim) is projected through a linear
          layer to n_concepts named scalars c = tanh(W·r + b) before the
          decoder. The decoder receives [query_shear | query_static | c]
          instead of [query_shear | query_static | r]. No skip connection —
          all context knowledge MUST route through the named concepts.
          The first N_CONCEPTS_SUPERVISED concepts have soft proxy targets;
          any additional concepts are free/latent (unsupervised capacity).

  [CBM-2] Physical concept definitions (CONCEPT_DEFS).
          Eight domain-grounded concepts with proxy labels derived from
          engineered features already computed in load_and_preprocess:
          crowding, ionic_screening, self_interaction, hydrophobicity,
          excluded_volume, viscosity_divergence, cosolute_protection,
          charge_environment. Proxies are soft — the model may deviate
          from them when the viscosity signal provides a stronger gradient.

  [CBM-3] Concept supervision loss (lambda_concept_sup, default 0.10).
          Per-iteration MSE between concept activations and the mean
          normalized proxy values of the context samples. Applied only to
          the first N_CONCEPTS_SUPERVISED dimensions. Proxy targets are
          z-scored then tanh-compressed to [-1, 1] to match tanh output.

  [CBM-4] Concept consistency loss (lambda_concept_consist, default 0.05).
          Two random halves of the same protein's context should produce
          similar concept activations. Cosine-distance loss in concept space,
          analogous to the latent consistency loss [FIX-4] in train_cnp_3.py.

  [CBM-5] Concept analysis and intervention (post-training).
          After training: per-group concept heatmap, concept-proxy
          correlation table, and an intervention demo that clamps each
          concept one at a time and reports the viscosity delta.
          Saved to: concept_activations.csv, concept_heatmap.png,
          concept_intervention.csv.

  [CBM-6] encode_memory / decode_from_memory API preserved.
          For CBM, encode_memory returns the concept vector c (not r),
          so inference_cnp.py works without changes — the "memory vector"
          is now the concept vector. decode_from_memory takes concept
          vectors. An additional encode_latent() method exposes r when
          needed (e.g., log_latent_variance diagnostic).

Data pipeline (load_and_preprocess) is UNCHANGED except:
  - concept proxy values extracted pre-scaling and stored as
    sample["concept_targets"] ([N_CONCEPTS_SUPERVISED] float tensor).
  - concept_scaler.pkl saved alongside preprocessor.pkl.
"""

import copy
import os
from collections import defaultdict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 0. Protein class map and non-protein groups (UNCHANGED)
# ==========================================

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
NON_PROTEIN_GROUPS = {"none"}

# ==========================================
# [CBM-2] Concept definitions
# ==========================================
# Each entry: (concept_name, proxy_column, sign)
#   proxy_column: column in df AFTER feature engineering but BEFORE scaling
#   sign: +1 if high proxy value → concept activation should be positive
#         -1 if high proxy value → concept activation should be negative
#
# Physical interpretation of each concept:
#   crowding            — macromolecular crowding / volume exclusion effects
#   ionic_screening     — Debye screening of protein-protein charge repulsion
#   self_interaction    — attractive self-interaction (kP > 0 = net attractive)
#   hydrophobicity      — hydrophobic patch exposure driving association
#   excluded_volume     — total occupied volume fraction (approach to jamming)
#   viscosity_divergence— KD crowding enhancement, diverges near phi_max
#   cosolute_protection — cosolute-mediated steric/entropic viscosity reduction
#   charge_environment  — net charge-concentration class driving repulsion
#
# These are SOFT priors. The concept loss weight (lambda_concept_sup=0.10)
# is intentionally small so the viscosity MSE signal can override the proxy
# when the data warrants it.

CONCEPT_DEFS = [
    ("crowding", "Exp_Crowding", +1),
    ("ionic_screening", "Ionic_Strength_Proxy", +1),
    ("self_interaction", "kP", -1),  # high kP → more association → high visc
    ("hydrophobicity", "HCI", +1),
    ("excluded_volume", "Phi_Total", +1),
    ("viscosity_divergence", "KD_Asymptote", +1),
    (
        "cosolute_protection",
        "prior_stabilizer",
        -1,
    ),  # high stabilizer score → reduces visc
    ("charge_environment", "C_Class", +1),
]

N_CONCEPTS_SUPERVISED = len(CONCEPT_DEFS)  # 8 — these get proxy supervision
CONCEPT_NAMES = [cd[0] for cd in CONCEPT_DEFS]


# ==========================================
# 1. Model Architecture
# ==========================================


class AttentionPool(nn.Module):
    """[FIX-1] LayerNorm after attention output — stabilizes pooler scale."""

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
    """
    Original CNP — retained for Optuna baseline comparison.
    forward() now returns (pred, None) for API uniformity with CBM.
    """

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
        r = self.pooler(self.encoder(context_tensor))
        n_q = query_shear.size(1)
        r_exp = r.unsqueeze(1).repeat(1, n_q, 1)
        return self.decoder(torch.cat([query_shear, query_static, r_exp], dim=-1)), None

    def encode_memory(self, context_tensor):
        return self.pooler(self.encoder(context_tensor))

    def decode_from_memory(self, memory_vector, query_shear, query_static):
        n_q = query_shear.size(1)
        r_exp = memory_vector.unsqueeze(1).repeat(1, n_q, 1)
        return self.decoder(torch.cat([query_shear, query_static, r_exp], dim=-1))


class ConceptBottleneckCNP(nn.Module):
    """
    [CBM-1] Hard Concept Bottleneck CNP.

    Data flow:
        context_tensor
            → encoder (2 + static_dim → hidden_dim → latent_dim)
            → AttentionPool → r  [B, latent_dim]
            → concept_proj + tanh → c  [B, n_concepts]    ← inspectable
            → decoder (1 + static_dim + n_concepts → hidden_dim → 1)
            → log-viscosity prediction

    The decoder has NO access to r. Every bit of context knowledge must
    be expressed as one of the n_concepts concept scalars. The tanh keeps
    all concepts in [-1, 1], making activation magnitudes comparable.

    First N_CONCEPTS_SUPERVISED concepts correspond to named physical
    phenomena and receive soft proxy supervision (lambda_concept_sup).
    Any additional concepts (n_concepts > N_CONCEPTS_SUPERVISED) are
    free/latent — the model can use them for unexplained variance.

    encode_memory() returns c (concept vector) to preserve the
    encode_memory → decode_from_memory API used by inference_cnp.py.
    Use encode_latent() when you need r for diagnostic purposes.
    """

    def __init__(
        self,
        static_dim,
        hidden_dim=128,
        latent_dim=128,
        n_concepts=N_CONCEPTS_SUPERVISED,
        concept_names=None,
        dropout=0.0,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.n_concepts = n_concepts
        self.concept_names = (
            concept_names
            if concept_names is not None
            else CONCEPT_NAMES[:n_concepts]
            + [f"latent_{i}" for i in range(max(0, n_concepts - N_CONCEPTS_SUPERVISED))]
        )

        # Encoder + pooler: identical to CrossSampleCNP
        self.encoder = nn.Sequential(
            nn.Linear(2 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.pooler = AttentionPool(latent_dim)

        # [CBM-1] Concept projection: linear + tanh → c in [-1, 1]
        # Linear keeps concepts interpretable as weighted combinations of r;
        # tanh bounds prevent scale explosion and makes values readable as
        # directional effect size.
        self.concept_proj = nn.Linear(latent_dim, n_concepts)

        # Decoder: takes n_concepts instead of latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(1 + static_dim + n_concepts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _project_concepts(self, r):
        """r [B, latent_dim] → c [B, n_concepts] via linear + tanh."""
        return torch.tanh(self.concept_proj(r))

    def forward(self, context_tensor, query_shear, query_static):
        r = self.pooler(self.encoder(context_tensor))
        c = self._project_concepts(r)
        n_q = query_shear.size(1)
        c_exp = c.unsqueeze(1).repeat(1, n_q, 1)
        pred = self.decoder(torch.cat([query_shear, query_static, c_exp], dim=-1))
        return pred, c  # [CBM-5] concept vector is a first-class output

    def encode_latent(self, context_tensor):
        """Returns raw latent r [B, latent_dim] — for log_latent_variance diagnostic."""
        return self.pooler(self.encoder(context_tensor))

    def encode_memory(self, context_tensor):
        """
        [CBM-6] Returns concept vector c [B, n_concepts].
        API-compatible with CrossSampleCNP.encode_memory() so inference_cnp.py
        works unchanged — the 'memory vector' is now the concept vector.
        """
        r = self.pooler(self.encoder(context_tensor))
        return self._project_concepts(r)

    def decode_from_memory(self, concept_vector, query_shear, query_static):
        """Decode from a concept vector. Supports concept intervention."""
        n_q = query_shear.size(1)
        c_exp = concept_vector.unsqueeze(1).repeat(1, n_q, 1)
        return self.decoder(torch.cat([query_shear, query_static, c_exp], dim=-1))

    def intervene(
        self, context_tensor, query_shear, query_static, concept_idx, concept_value
    ):
        """
        [CBM-5] Causal intervention: clamp concept_idx to concept_value, re-decode.
        This is do(c_i = v) in the Pearl causal sense — not merely correlation.

        Args:
            concept_idx:   Index into concept vector (int or list of ints).
            concept_value: Scalar value to clamp to (float, in [-1, 1]).

        Returns:
            Predictions under the intervention [B, n_queries, 1].
        """
        c = self.encode_memory(context_tensor)
        c_mod = c.clone()
        if isinstance(concept_idx, int):
            concept_idx = [concept_idx]
        for idx in concept_idx:
            c_mod[:, idx] = concept_value
        return self.decode_from_memory(c_mod, query_shear, query_static)


# ==========================================
# Utility: uniform forward call
# ==========================================


def _forward(model, ctx, qx, qstat):
    """
    Uniform forward call for both CrossSampleCNP and ConceptBottleneckCNP.
    Returns (predictions [B, Q, 1], concepts [B, n_concepts] or None).
    """
    out = model(ctx, qx, qstat)
    pred, concepts = out
    return pred, concepts


def _encode_latent(model, context_tensor):
    """
    Return raw latent r for diagnostic purposes.
    For CBM uses encode_latent(); for plain CNP uses encode_memory().
    """
    if isinstance(model, ConceptBottleneckCNP):
        return model.encode_latent(context_tensor)
    return model.encode_memory(context_tensor)


# ==========================================
# 2. Data pipeline constants (UNCHANGED)
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


# ==========================================
# 2. Data pipeline (UNCHANGED except concept_targets)
# ==========================================


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
    for k in CONC_THRESHOLDS.keys():
        new_conc_cols.append(f"{k}_low")
        new_conc_cols.append(f"{k}_high")

    print("Normalizing units to mg/mL and calculating Physics Features...")

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

    stabilizer_mw = get_mw(df["Stabilizer_type"], default_mw=342.3)
    df["Stabilizer_mg_mL"] = df["Stabilizer_conc"] * stabilizer_mw

    salt_mw = get_mw(df["Salt_type"], default_mw=58.44)
    df["Salt_mg_mL"] = (df["Salt_conc"] * salt_mw) / 1000.0

    excipient_mw = get_mw(df["Excipient_type"], default_mw=150.0)
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
    V_BAR_PROTEIN, V_BAR_STAB, V_BAR_SALT, V_BAR_EXCIP = (
        0.73 / 1000,
        0.62 / 1000,
        0.30 / 1000,
        0.70 / 1000,
    )
    df["Phi_Protein"] = df["Protein_conc"] * V_BAR_PROTEIN
    df["Phi_Stabilizer"] = df["Stabilizer_mg_mL"] * V_BAR_STAB
    df["Phi_Salt"] = df["Salt_mg_mL"] * V_BAR_SALT
    df["Phi_Excipient"] = df["Excipient_mg_mL"] * V_BAR_EXCIP
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

    num_cols.extend(new_prior_cols)
    num_cols.extend(new_conc_cols)
    num_cols.extend(engineered_cols)

    # ---------------------------------------------------------------
    # [CBM-2] Extract concept proxy values BEFORE scaling.
    # These are raw domain values; we z-score then tanh-compress to
    # [-1, 1] so they match the tanh-bounded concept outputs.
    # ---------------------------------------------------------------
    proxy_cols = [cd[1] for cd in CONCEPT_DEFS]
    proxy_signs = np.array([cd[2] for cd in CONCEPT_DEFS], dtype=float)

    concept_raw = np.zeros((len(df), N_CONCEPTS_SUPERVISED), dtype=np.float64)
    for j, col in enumerate(proxy_cols):
        if col in df.columns:
            concept_raw[:, j] = df[col].fillna(0.0).values.astype(float)
    # Apply sign convention: negative sign means high value → negative concept
    concept_raw_signed = concept_raw * proxy_signs

    # Z-score, then apply tanh(z/2) → compresses ±2σ outliers to ±0.76,
    # saturates extreme outliers to ±1, matches the concept layer tanh output.
    c_mean = concept_raw_signed.mean(axis=0)
    c_std = concept_raw_signed.std(axis=0) + 1e-8
    concept_normalized = np.tanh((concept_raw_signed - c_mean) / c_std / 2.0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "concept_proxy_mean.npy"), c_mean)
        np.save(os.path.join(save_dir, "concept_proxy_std.npy"), c_std)
        np.save(os.path.join(save_dir, "concept_proxy_signs.npy"), proxy_signs)
        print(f"Concept proxy scaler saved to {save_dir}/concept_proxy_*.npy")

    # ---------------------------------------------------------------
    # Preprocessor and scaling (UNCHANGED)
    # ---------------------------------------------------------------
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
        joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.pkl"))
        joblib.dump(physics_scaler, os.path.join(save_dir, "physics_scaler.pkl"))

    # Build sample dicts
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
            interval_pts = np.linspace(
                interval_endpoints[j], interval_endpoints[j + 1], 10
            )
            dense_x_list.append(
                interval_pts[:-1] if j < len(interval_endpoints) - 2 else interval_pts
            )

        dense_x = np.concatenate(dense_x_list) if dense_x_list else x_arr
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
                    # [CBM-2] Normalized concept proxy targets for this sample
                    "concept_targets": torch.tensor(
                        concept_normalized[i], dtype=torch.float32
                    ),
                }
            )

    return samples, X_matrix.shape[1]


# ==========================================
# 3. Training helpers
# ==========================================


def _build_ctx_tensor(task_samples, indices, device):
    """Build context tensor [1, N_points, 2+static_dim] from sample indices."""
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


def _build_concept_targets(task_samples, indices, device):
    """
    [CBM-3] Mean concept proxy targets for a set of sample indices.
    Returns tensor [1, N_CONCEPTS_SUPERVISED] in [-1, 1].
    Returns None if samples lack concept_targets (backward-compat guard).
    """
    targets = []
    for i in indices:
        s = task_samples[i]
        if "concept_targets" in s:
            targets.append(s["concept_targets"])
    if not targets:
        return None
    return torch.stack(targets).mean(dim=0).unsqueeze(0).to(device)  # [1, N_SUP]


# ==========================================
# 4. Training epoch
# ==========================================


def train_epoch(
    model,
    samples,
    optimizer,
    device,
    iterations=100,
    group_weights=None,
    lambda_triplet=0.30,
    lambda_consistency=0.10,  # [FIX-4]
    lambda_utility=2.5,  # [FIX-5]
    triplet_margin=3.0,  # [FIX-3]
    lambda_norm=0.05,  # [FIX-NORM]
    norm_target=5.0,
    lambda_concept_sup=0.10,  # [CBM-3] concept supervision (proxy labels)
    lambda_concept_consist=0.05,  # [CBM-4] concept consistency (intra-group)
):
    """
    Train one epoch. Extends train_cnp_3.py with [CBM-3] and [CBM-4] losses.

    All [FIX-N] losses from train_cnp_3.py are preserved UNCHANGED.
    CBM losses activate only when model is ConceptBottleneckCNP and
    concept_targets are present in the sample dicts. Plain CrossSampleCNP
    training is unaffected.
    """
    model.train()
    total_loss = 0
    count = 0

    is_cbm = isinstance(model, ConceptBottleneckCNP)

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

        # ---- Main forward pass with static masking [FIX-5] ----
        if np.random.random() < 0.60:
            mask = torch.bernoulli(torch.full_like(qstat_A, 0.5))
            qstat_A_in = qstat_A * mask
        else:
            qstat_A_in = qstat_A

        pred_A, concepts_A = _forward(model, ctx_A, qx_A, qstat_A_in)
        mse_loss = F.mse_loss(pred_A, qy_A)

        # ---- Context utility loss [FIX-5] ----
        with torch.no_grad():
            pred_null, _ = _forward(model, torch.zeros_like(ctx_A), qx_A, qstat_A)
        mse_null = F.mse_loss(pred_null, qy_A).detach()

        pred_ctx_unmasked, _ = _forward(model, ctx_A, qx_A, qstat_A)
        mse_ctx_unmasked = F.mse_loss(pred_ctx_unmasked, qy_A)
        utility_loss = torch.clamp(mse_ctx_unmasked - mse_null + 1e-3, min=0.0)

        # ---- Soft norm penalty [FIX-NORM] ----
        r_current = _encode_latent(model, ctx_A)
        r_norm = torch.norm(r_current, p=2, dim=-1)
        norm_penalty = torch.mean(torch.clamp(r_norm - norm_target, min=0.0) ** 2)

        # ---- [CBM-3] Concept supervision loss ----
        # Mean context proxy targets vs. model concept activations.
        concept_sup_loss = torch.tensor(0.0, device=device)
        if is_cbm and concepts_A is not None:
            ctx_concept_targets = _build_concept_targets(
                task_A, idx_A[:n_ctx_A], device
            )
            if ctx_concept_targets is not None:
                n_sup = min(N_CONCEPTS_SUPERVISED, model.n_concepts)
                concept_sup_loss = F.mse_loss(
                    concepts_A[:, :n_sup],
                    ctx_concept_targets[:, :n_sup],
                )

        # ---- Triplet [FIX-3] + latent consistency [FIX-4]
        #      + [CBM-4] concept consistency ----
        triplet_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)
        concept_consist_loss = torch.tensor(0.0, device=device)

        if prot_A in all_protein_list and len(all_protein_list) >= 2:
            perm_full = np.random.permutation(len(task_A))
            half = max(1, len(perm_full) // 2)

            ctx_anchor = _build_ctx_tensor(task_A, perm_full[:half], device)
            ctx_pos = _build_ctx_tensor(task_A, perm_full[half:], device)

            # [FIX-4] Pre-pooled latent consistency
            enc_anchor_mean = model.encoder(ctx_anchor).mean(dim=1)
            enc_pos_mean = model.encoder(ctx_pos).mean(dim=1)
            cos_within = F.cosine_similarity(enc_anchor_mean, enc_pos_mean, dim=-1)
            consistency_loss = (1.0 - cos_within).mean()

            r_anchor = _encode_latent(model, ctx_anchor)
            r_pos = _encode_latent(model, ctx_pos)

            # [CBM-4] Concept consistency: same protein → similar concept activations
            if is_cbm:
                c_anchor = model.encode_memory(ctx_anchor)  # returns c
                c_pos = model.encode_memory(ctx_pos)
                concept_consist_loss = (
                    1.0 - F.cosine_similarity(c_anchor, c_pos, dim=-1).mean()
                )

            # [FIX-3] Triplet loss with hard negative mining
            class_A = PROTEIN_CLASS_MAP.get(prot_A, "unknown")
            same_class_negs = [
                g
                for g in all_protein_list
                if g != prot_A and PROTEIN_CLASS_MAP.get(g, "") == class_A
            ]
            diff_class_negs = [g for g in all_protein_list if g != prot_A]

            if same_class_negs and np.random.random() < 0.70:
                prot_B = np.random.choice(same_class_negs)
            elif diff_class_negs:
                prot_B = np.random.choice(diff_class_negs)
            else:
                prot_B = prot_A

            task_B = groups[prot_B]
            idx_B = np.random.permutation(len(task_B))
            n_ctx_B = np.random.randint(1, min(8, len(idx_B)))
            r_neg = _encode_latent(
                model, _build_ctx_tensor(task_B, idx_B[:n_ctx_B], device)
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
            + lambda_norm * norm_penalty
            + lambda_concept_sup * concept_sup_loss  # [CBM-3]
            + lambda_concept_consist * concept_consist_loss  # [CBM-4]
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


# ==========================================
# 5. Validation (UNCHANGED logic, updated for tuple forward)
# ==========================================


def validate(model, samples, device, n_repeats=3):
    """
    [FIX-8] Randomized context selection. Updated to unpack (pred, concepts)
    tuple returned by both CrossSampleCNP and ConceptBottleneckCNP.
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

                pred, _ = _forward(model, ctx_tensor, q_x, q_stat)
                loss = F.mse_loss(pred, true_y)
                if not torch.isnan(loss):
                    group_errors.append(loss.item())

            if group_errors:
                total_error += np.mean(group_errors)
                count += 1

    return total_error / max(1, count)


# ==========================================
# 6. Diagnostics
# ==========================================


def log_latent_variance(model, samples, device):
    """
    [FIX-9] Inter-group latent L2 distance in r-space.
    For CBM uses encode_latent() to get r (not the concept vector).
    """
    model.eval()
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    group_r = {}
    with torch.no_grad():
        for prot, task_samples in groups.items():
            if len(task_samples) < 2 or prot in NON_PROTEIN_GROUPS:
                continue
            idx = np.random.permutation(len(task_samples))[: min(5, len(task_samples))]
            ctx_items = []
            for i in idx:
                s = task_samples[i]
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_items.append(torch.cat([s["points"], stat], dim=1))
            ctx_t = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)
            r = _encode_latent(model, ctx_t).squeeze(0).cpu().numpy()
            group_r[prot] = r

    if len(group_r) < 2:
        return 0.0

    vecs = np.stack(list(group_r.values()))
    dists = [
        np.linalg.norm(vecs[i] - vecs[j])
        for i in range(len(vecs))
        for j in range(i + 1, len(vecs))
    ]
    return float(np.mean(dists))


def log_concept_activations(model, samples, device, n_draws=10, k=8):
    """
    [CBM-5] Compute mean concept activations per protein group.

    Returns:
        group_concepts: dict {group_name: np.ndarray [n_concepts]}
        concept_matrix: np.ndarray [n_groups, n_concepts] (for heatmap)
        group_names:    list of group names (row order)
    """
    if not isinstance(model, ConceptBottleneckCNP):
        return {}, None, []

    model.eval()
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    group_concepts = {}
    with torch.no_grad():
        for prot, task_samples in groups.items():
            if len(task_samples) < 2:
                continue
            draw_concepts = []
            for _ in range(n_draws):
                k_eff = min(k, len(task_samples))
                idx = np.random.choice(len(task_samples), size=k_eff, replace=False)
                ctx_items = []
                for i in idx:
                    s = task_samples[i]
                    stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                    ctx_items.append(torch.cat([s["points"], stat], dim=1))
                ctx_t = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)
                c = model.encode_memory(ctx_t).squeeze(0).cpu().numpy()
                draw_concepts.append(c)
            group_concepts[prot] = np.stack(draw_concepts).mean(axis=0)

    if not group_concepts:
        return {}, None, []

    group_names = sorted(group_concepts.keys())
    concept_matrix = np.stack([group_concepts[g] for g in group_names])
    return group_concepts, concept_matrix, group_names


def save_concept_heatmap(concept_matrix, group_names, concept_names, save_path):
    """Save concept activation heatmap PNG."""
    fig, ax = plt.subplots(
        figsize=(max(8, len(concept_names) * 0.9), max(4, len(group_names) * 0.45))
    )
    im = ax.imshow(concept_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(concept_names)))
    ax.set_xticklabels(concept_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names, fontsize=9)
    plt.colorbar(im, ax=ax, label="Concept activation [-1, 1]")
    ax.set_title("Concept activations by protein group", fontsize=11)
    # Annotate cells
    for i in range(len(group_names)):
        for j in range(len(concept_names)):
            val = concept_matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if abs(val) < 0.6 else "white",
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Concept heatmap saved to {save_path}")


def run_concept_intervention_demo(
    model,
    samples,
    device,
    physics_scaler_path,
    save_dir,
):
    """
    [CBM-5] Sweep each concept independently from -1 → +1 and record
    the mean change in predicted log-viscosity at 100 s⁻¹.

    For each protein group:
      - Encode group context → baseline concept vector c
      - For each concept i: clamp c_i to {-1, -0.5, 0, 0.5, 1}
      - Report Δlog-viscosity relative to baseline

    Saves concept_intervention.csv.
    """
    if not isinstance(model, ConceptBottleneckCNP):
        print("Concept intervention requires ConceptBottleneckCNP — skipping.")
        return

    physics_scaler = joblib.load(physics_scaler_path)
    shear_mean = physics_scaler.mean_[0]
    shear_scale = physics_scaler.scale_[0]
    visc_mean = physics_scaler.mean_[1]
    visc_scale = physics_scaler.scale_[1]

    # Predict at 100 s⁻¹
    log_shear_100 = np.log10(100.0)
    shear_scaled = (log_shear_100 - shear_mean) / shear_scale
    query_shear = torch.tensor([[[shear_scaled]]], dtype=torch.float32).to(device)

    model.eval()
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    intervention_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    records = []

    with torch.no_grad():
        for prot, task_samples in sorted(groups.items()):
            if len(task_samples) < 2 or prot in NON_PROTEIN_GROUPS:
                continue

            # Build full-group context
            ctx_items = []
            for s in task_samples:
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_items.append(torch.cat([s["points"], stat], dim=1))
            ctx_t = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)

            # Representative static features (mean of group)
            q_static = torch.stack([s["static"] for s in task_samples]).mean(0)
            q_static = (
                q_static.unsqueeze(0).unsqueeze(0).to(device)
            )  # [1, 1, static_dim]

            # Baseline prediction
            c_base = model.encode_memory(ctx_t)  # [1, n_concepts]
            pred_base_sc = model.decode_from_memory(c_base, query_shear, q_static)
            pred_base_lv = float(pred_base_sc.squeeze()) * visc_scale + visc_mean

            for ci, cname in enumerate(model.concept_names):
                for cval in intervention_values:
                    pred_int_sc = model.decode_from_memory(
                        model.intervene(
                            ctx_t,
                            query_shear,
                            q_static,
                            concept_idx=ci,
                            concept_value=cval,
                        ),
                        query_shear,
                        q_static,
                    )
                    # intervene() returns predictions directly — fix signature:
                    # use the lower-level API
                    c_mod = c_base.clone()
                    c_mod[:, ci] = cval
                    pred_int_sc = model.decode_from_memory(c_mod, query_shear, q_static)
                    pred_int_lv = float(pred_int_sc.squeeze()) * visc_scale + visc_mean
                    records.append(
                        {
                            "Group": prot,
                            "Concept": cname,
                            "Concept_idx": ci,
                            "Intervention_value": cval,
                            "Baseline_log_visc": pred_base_lv,
                            "Predicted_log_visc": pred_int_lv,
                            "Delta_log_visc": pred_int_lv - pred_base_lv,
                        }
                    )

    df_int = pd.DataFrame(records)
    save_path = os.path.join(save_dir, "concept_intervention.csv")
    df_int.to_csv(save_path, index=False)
    print(f"Concept intervention results saved to {save_path}")

    # Print summary: which concepts have largest effect magnitude
    effect = (
        df_int.groupby("Concept")["Delta_log_visc"]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
    )
    print("\nMean |Δlog-visc| per concept (intervention sensitivity):")
    for cname, mag in effect.items():
        bar = "█" * int(mag * 40)
        print(f"  {cname:<28s} {mag:.4f}  {bar}")


# ==========================================
# 7. Optuna objective (updated for CBM)
# ==========================================


def objective_cv(trial, samples, static_dim, device):
    hidden_dim = trial.suggest_int("hidden_dim", 128, 256, step=64)
    latent_dim = trial.suggest_int("latent_dim", 128, 256, step=64)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    # [CBM-3] Tune concept bottleneck width and supervision strength
    n_free = trial.suggest_int("n_free_concepts", 0, 8, step=2)
    n_concepts = N_CONCEPTS_SUPERVISED + n_free
    lambda_concept_sup = trial.suggest_float("lambda_concept_sup", 0.02, 0.3, log=True)

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

        # [CBM-1] Use ConceptBottleneckCNP in Optuna search
        model = ConceptBottleneckCNP(
            static_dim,
            hidden_dim,
            latent_dim,
            n_concepts=n_concepts,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        for epoch in range(40):
            train_loss, _ = train_epoch(
                model,
                train_fold,
                optimizer,
                device,
                iterations=50,
                lambda_concept_sup=lambda_concept_sup,
            )
            val_loss = validate(model, val_fold, device, n_repeats=2)
            trial.report(val_loss, fold_idx * 40 + epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        fold_scores.append(validate(model, val_fold, device, n_repeats=3))

    return float(np.mean(fold_scores)) if fold_scores else float("inf")


# ==========================================
# 8. Main execution
# ==========================================

if __name__ == "__main__":
    data = "data/processed/augmented_formulation_data.csv"
    out = "./models/experiments/cbm_cnp_v1"
    trials = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default params (used when trials=0 or as Optuna fallback)
    best_params = {
        "hidden_dim": 128,
        "latent_dim": 128,
        "dropout": 0.15,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "n_free_concepts": 4,  # [CBM-1] 8 supervised + 4 free = 12 total
        "lambda_concept_sup": 0.10,  # [CBM-3]
        "epochs": 150,
    }

    samples, static_dim = load_and_preprocess(data, save_dir=out)
    print(
        f"Loaded {len(samples)} samples from "
        f"{len(set(s['group'] for s in samples))} protein groups."
    )
    print(
        f"Concept bottleneck: {N_CONCEPTS_SUPERVISED} supervised + "
        f"{best_params['n_free_concepts']} free = "
        f"{N_CONCEPTS_SUPERVISED + best_params['n_free_concepts']} total concepts"
    )

    # ==========================================
    # Optuna hyperparameter search
    # ==========================================
    if trials > 0:
        print("Starting Group-Held-Out Optuna Optimization (CBM)...")
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
        best_params.update(study.best_params)
    else:
        print("Skipping hyperparameter tuning (trials=0). Using default params.")
        print("Default params:", best_params)

    n_concepts = N_CONCEPTS_SUPERVISED + best_params.get("n_free_concepts", 4)
    lambda_concept_sup = best_params.get("lambda_concept_sup", 0.10)

    # ==========================================
    # Final retraining
    # ==========================================
    print(
        f"\nRetraining final ConceptBottleneckCNP "
        f"(n_concepts={n_concepts}) on ALL data..."
    )

    final_model = ConceptBottleneckCNP(
        static_dim,
        hidden_dim=best_params["hidden_dim"],
        latent_dim=best_params["latent_dim"],
        n_concepts=n_concepts,
        concept_names=CONCEPT_NAMES
        + [f"latent_{i}" for i in range(max(0, n_concepts - N_CONCEPTS_SUPERVISED))],
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

    # Stratified 10% early-stop split [FIX-WATCHLIST]
    final_train_set, final_stop_set = [], []
    groups_dict = defaultdict(list)
    for s in samples:
        groups_dict[s["group"]].append(s)

    for g, g_samples in groups_dict.items():
        np.random.shuffle(g_samples)
        n_val = max(1, int(len(g_samples) * 0.1))
        if len(g_samples) < 2:
            final_train_set.extend(g_samples)
        else:
            final_stop_set.extend(g_samples[:n_val])
            final_train_set.extend(g_samples[n_val:])

    print(
        f"Final Train: {len(final_train_set)} | "
        f"Early Stop Watchlist (stratified 10%): {len(final_stop_set)}"
    )

    best_loss = float("inf")
    patience_counter = 0
    patience_limit = 80
    best_state = None

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
            lambda_concept_sup=lambda_concept_sup,
        )

        # [FIX-6] EMA difficulty reweighting
        for g, mse in per_group_mse.items():
            group_weights[g] = ema_alpha * mse + (1 - ema_alpha) * group_weights[g]
        total_w = sum(group_weights.values())
        n_g = len(group_weights)
        for g in group_weights:
            group_weights[g] = group_weights[g] / total_w * n_g

        val_loss = validate(final_model, final_stop_set, device, n_repeats=10)
        scheduler.step(val_loss)

        if ep % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            latent_var = log_latent_variance(final_model, final_train_set, device)

            # Pembrolizumab latent diagnostics [FIX-PRIORITY 2]
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
                        r_list.append(_encode_latent(final_model, ctx_item))
                    if r_list:
                        r_pembro = torch.cat(r_list, dim=0)
                        pembro_norm = torch.norm(r_pembro, p=2, dim=-1).mean().item()
                        pembro_norm_str = f"{pembro_norm:.3f}"
                        dists = [
                            torch.norm(r_pembro[i] - r_pembro[j], p=2).item()
                            for i in range(len(r_pembro))
                            for j in range(i + 1, len(r_pembro))
                        ]
                        if dists:
                            pembro_spread_str = f"{np.mean(dists):.3f}"
                final_model.train()

            top_hard = sorted(group_weights.items(), key=lambda x: -x[1])[:3]
            hard_str = ", ".join(f"{g}:{w:.2f}" for g, w in top_hard)

            print(
                f"Epoch {ep:3d}: Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"LR {current_lr:.2e} | LatentVar {latent_var:.3f} | "
                f"Pembro [Norm: {pembro_norm_str} | Spread: {pembro_spread_str}] | "
                f"Top hard: [{hard_str}]"
            )

            if ep >= 30 and latent_var < 0.2:
                print(
                    f"  *** WARNING: LatentVar={latent_var:.3f} is very low. "
                    "Context collapse may still be occurring. ***"
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

    # Save checkpoint — extended config includes CBM fields
    save_path = os.path.join(out, "best_model.pth")
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "config": best_params,
            "static_dim": static_dim,
            "n_concepts": n_concepts,  # [CBM-6]
            "concept_names": final_model.concept_names,
            "model_class": "ConceptBottleneckCNP",
        },
        save_path,
    )
    print(f"Model saved to {save_path}")
    print(
        f"Final group difficulty weights: "
        f"{dict(sorted(group_weights.items(), key=lambda x: -x[1]))}"
    )

    # ==========================================
    # [CBM-5] Concept analysis
    # ==========================================
    print("\n" + "=" * 60)
    print("CONCEPT ANALYSIS")
    print("=" * 60)

    group_concepts, concept_matrix, group_names = log_concept_activations(
        final_model,
        samples,
        device,
        n_draws=20,
        k=8,
    )

    if concept_matrix is not None:
        # Save heatmap
        save_concept_heatmap(
            concept_matrix,
            group_names,
            final_model.concept_names,
            save_path=os.path.join(out, "concept_heatmap.png"),
        )

        # Save CSV: rows=groups, cols=concepts
        df_concepts = pd.DataFrame(
            concept_matrix,
            index=group_names,
            columns=final_model.concept_names,
        )
        df_concepts.index.name = "Group"
        concepts_csv = os.path.join(out, "concept_activations.csv")
        df_concepts.to_csv(concepts_csv)
        print(f"Concept activations saved to {concepts_csv}")

        # Print summary table
        print("\nConcept activations by protein group (mean over 20 context draws):")
        col_w = 13
        header = f"  {'Group':<22}" + "".join(
            f"{c[:col_w]:>{col_w}}" for c in final_model.concept_names
        )
        print(header)
        print("  " + "-" * (22 + col_w * len(final_model.concept_names)))
        for gname in group_names:
            vals = group_concepts[gname]
            row = f"  {gname:<22}" + "".join(f"{v:>{col_w}.3f}" for v in vals)
            print(row)

        # Concept-proxy correlation (only supervised concepts)
        print("\nConcept-proxy correlation summary:")
        proxy_vals_all = []
        concept_vals_all = []
        for s in samples:
            if "concept_targets" in s:
                proxy_vals_all.append(s["concept_targets"].numpy())
                g = s["group"]
                if g in group_concepts:
                    concept_vals_all.append(group_concepts[g][:N_CONCEPTS_SUPERVISED])
        if proxy_vals_all and concept_vals_all:
            proxy_arr = np.stack(proxy_vals_all)  # [N, N_sup]
            concept_arr = np.stack(concept_vals_all)  # [N, N_sup]
            print(f"  {'Concept':<28} {'Proxy column':<26} {'Pearson r':>10}")
            print("  " + "-" * 66)
            for ci, (cname, pcol, _) in enumerate(CONCEPT_DEFS):
                if concept_arr.shape[0] > 1:
                    r = np.corrcoef(proxy_arr[:, ci], concept_arr[:, ci])[0, 1]
                    print(f"  {cname:<28} {pcol:<26} {r:>10.3f}")

    # Concept intervention demo
    run_concept_intervention_demo(
        final_model,
        samples,
        device,
        physics_scaler_path=os.path.join(out, "physics_scaler.pkl"),
        save_dir=out,
    )

    # ==========================================
    # Parity evaluation (UNCHANGED from train_cnp_3.py)
    # ==========================================
    print("\n" + "=" * 60)
    print("PARITY EVALUATION")
    print("=" * 60)
    print(f"Data: {data}")

    physics_scaler_eval = joblib.load(os.path.join(out, "physics_scaler.pkl"))
    raw_df = pd.read_csv(data)

    parity_shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1000.0,
        "Viscosity_10000": 10000.0,
        "Viscosity_100000": 100000.0,
        "Viscosity_15000000": 1.5e7,
    }
    key_shears_eval = list(parity_shear_map.values())
    key_log_shears_eval = np.log10(key_shears_eval)

    shear_mean = physics_scaler_eval.mean_[0]
    shear_scale = physics_scaler_eval.scale_[0]
    visc_mean = physics_scaler_eval.mean_[1]
    visc_scale = physics_scaler_eval.scale_[1]

    scaled_log_shears = torch.tensor(
        [(ls - shear_mean) / shear_scale for ls in key_log_shears_eval],
        dtype=torch.float32,
    ).to(device)

    eval_groups = defaultdict(list)
    for s in samples:
        eval_groups[s["group"]].append(s)

    all_actual, all_predicted = [], []
    all_eval_groups, all_sample_ids, all_shear_rates = [], [], []
    n_shears = len(key_shears_eval)

    final_model.eval()
    with torch.no_grad():
        for sample in samples:
            sid = sample["id"]
            group = sample["group"]
            task_samples = eval_groups[group]

            ctx_samples = [s for s in task_samples if s["id"] != sid]
            if not ctx_samples:
                ctx_samples = task_samples

            ctx_items = []
            for s in ctx_samples:
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_items.append(torch.cat([s["points"], stat], dim=1))
            ctx_tensor = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)

            q_shear = scaled_log_shears.view(1, n_shears, 1)
            q_static = (
                sample["static"]
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, n_shears, 1)
                .to(device)
            )

            # encode_memory returns concept vector c; decode_from_memory takes c
            memory = final_model.encode_memory(ctx_tensor)
            pred_sc = final_model.decode_from_memory(memory, q_shear, q_static)
            pred_sc = pred_sc.squeeze().cpu().numpy()
            pred_log = pred_sc * visc_scale + visc_mean
            pred_visc = 10.0**pred_log

            row_mask = raw_df["ID"] == sid
            if not row_mask.any():
                continue
            row = raw_df[row_mask].iloc[0]
            for i, (col, shear) in enumerate(parity_shear_map.items()):
                if col in raw_df.columns and pd.notna(row[col]) and row[col] > 0:
                    all_actual.append(float(row[col]))
                    all_predicted.append(float(pred_visc[i]))
                    all_eval_groups.append(group)
                    all_sample_ids.append(sid)
                    all_shear_rates.append(shear)

    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    all_eval_groups = np.array(all_eval_groups)
    all_shear_rates = np.array(all_shear_rates)

    log_actual = np.log10(np.clip(all_actual, 1e-6, None))
    log_predicted = np.log10(np.clip(all_predicted, 1e-6, None))

    ss_res = np.sum((log_actual - log_predicted) ** 2)
    ss_tot = np.sum((log_actual - log_actual.mean()) ** 2)
    rmse_log = float(np.sqrt(np.mean((log_actual - log_predicted) ** 2)))
    r2_log = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    print(f"\nOverall ({len(all_actual)} sample-shear pairs):")
    print(f"  RMSE (log10 viscosity): {rmse_log:.4f}")
    print(f"  R²   (log10 viscosity): {r2_log:.4f}")

    print("\nPer-group parity (RMSE in log10 space):")
    for g in sorted(set(all_eval_groups)):
        mask = all_eval_groups == g
        g_rmse = float(np.sqrt(np.mean((log_actual[mask] - log_predicted[mask]) ** 2)))
        print(f"  {g:28s}: RMSE={g_rmse:.4f}  (n={mask.sum()})")

    print("\nPer-shear-rate parity (RMSE in log10 space):")
    for shear in key_shears_eval:
        mask = all_shear_rates == shear
        if not mask.any():
            continue
        s_rmse = float(np.sqrt(np.mean((log_actual[mask] - log_predicted[mask]) ** 2)))
        print(f"  {shear:12.0f} s⁻¹: RMSE={s_rmse:.4f}  (n={mask.sum()})")

    # Save parity CSV + parity plot
    parity_df = pd.DataFrame(
        {
            "ID": all_sample_ids,
            "Group": all_eval_groups,
            "Shear_Rate": all_shear_rates,
            "Actual_Viscosity": all_actual,
            "Predicted_Viscosity": all_predicted,
            "Log10_Actual": log_actual,
            "Log10_Predicted": log_predicted,
            "Log10_Error": log_predicted - log_actual,
        }
    )
    parity_csv_path = os.path.join(out, "parity_results.csv")
    parity_df.to_csv(parity_csv_path, index=False)
    print(f"\nParity results saved to {parity_csv_path}")

    try:
        import matplotlib.cm as cm

        unique_groups_plot = sorted(set(all_eval_groups))
        cmap = cm.get_cmap("tab20", len(unique_groups_plot))
        color_map = {g: cmap(i) for i, g in enumerate(unique_groups_plot)}

        fig, ax = plt.subplots(figsize=(7, 7))
        for g in unique_groups_plot:
            mask = all_eval_groups == g
            ax.scatter(
                log_actual[mask],
                log_predicted[mask],
                color=color_map[g],
                label=g,
                alpha=0.65,
                s=20,
            )

        lims = [
            min(log_actual.min(), log_predicted.min()) - 0.1,
            max(log_actual.max(), log_predicted.max()) + 0.1,
        ]
        ax.plot(lims, lims, "k--", linewidth=1, label="Parity (y=x)")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_xlabel("log₁₀(Actual Viscosity)")
        ax.set_ylabel("log₁₀(Predicted Viscosity)")
        ax.set_title(
            f"Parity Plot — All Samples & Shear Rates\n"
            f"RMSE={rmse_log:.4f}, R²={r2_log:.4f}"
        )
        ax.legend(fontsize=7, markerscale=1.5, loc="upper left", ncol=2)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "parity_plot.png"), dpi=150)
        plt.close(fig)
        print(f"Parity plot saved to {os.path.join(out, 'parity_plot.png')}")
    except Exception as e:
        print(f"(Parity plot skipped: {e})")

    # ==========================================
    # Feature importance (UNCHANGED from train_cnp_3.py)
    # encode_memory returns c; decoder takes c — permutation
    # correctly measures impact routed through the concept layer.
    # ==========================================
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Permutation, Concept-pathway)")
    print("=" * 60)

    preprocessor_fi = joblib.load(os.path.join(out, "preprocessor.pkl"))
    try:
        feature_names_fi = list(preprocessor_fi.get_feature_names_out())
    except Exception:
        feature_names_fi = [f"feature_{i}" for i in range(static_dim)]

    fi_ctx_tensors, fi_static_vecs = [], []
    fi_true_log_visc, fi_valid_masks = [], []

    final_model.eval()
    with torch.no_grad():
        for sample in samples:
            sid = sample["id"]
            group = sample["group"]
            task_samples = eval_groups[group]
            ctx_samples = [s for s in task_samples if s["id"] != sid] or task_samples

            ctx_items = []
            for s in ctx_samples:
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_items.append(torch.cat([s["points"], stat], dim=1))
            fi_ctx_tensors.append(torch.cat(ctx_items, dim=0).unsqueeze(0).to(device))
            fi_static_vecs.append(sample["static"])

            row_mask = raw_df["ID"] == sid
            true_lv, valid = [0.0] * 5, [False] * 5
            if row_mask.any():
                row_fi = raw_df[row_mask].iloc[0]
                for j, col in enumerate(parity_shear_map):
                    if (
                        col in raw_df.columns
                        and pd.notna(row_fi[col])
                        and row_fi[col] > 0
                    ):
                        true_lv[j] = np.log10(float(row_fi[col]))
                        valid[j] = True
            fi_true_log_visc.append(true_lv)
            fi_valid_masks.append(valid)

    fi_static_matrix = torch.stack(fi_static_vecs)
    fi_true_log_visc = np.array(fi_true_log_visc)
    fi_valid_masks = np.array(fi_valid_masks)

    # Pre-compute concept vectors (memory) for each sample — permutation re-runs decoder only
    fi_memory_list = []
    with torch.no_grad():
        for ctx_t in fi_ctx_tensors:
            fi_memory_list.append(final_model.encode_memory(ctx_t))  # [1, n_concepts]

    q_shear_fi = scaled_log_shears.view(1, n_shears, 1)

    def _decoder_mse(static_mat):
        errs = []
        with torch.no_grad():
            for i, (mem, true_lv, valid) in enumerate(
                zip(fi_memory_list, fi_true_log_visc, fi_valid_masks, strict=False)
            ):
                if not any(valid):
                    continue
                q_st = (
                    static_mat[i]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(1, n_shears, 1)
                    .to(device)
                )
                pred_sc = (
                    final_model.decode_from_memory(mem, q_shear_fi, q_st)
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                pred_lv = pred_sc * visc_scale + visc_mean
                for j in range(5):
                    if valid[j]:
                        errs.append((pred_lv[j] - true_lv[j]) ** 2)
        return float(np.mean(errs)) if errs else float("nan")

    baseline_fi_mse = _decoder_mse(fi_static_matrix)
    print(f"Baseline decoder MSE (log10 viscosity): {baseline_fi_mse:.6f}")
    print(f"Permuting {static_dim} features across {len(samples)} samples...")

    fi_importances = np.zeros(static_dim)
    for j in range(static_dim):
        perm = fi_static_matrix.clone()
        perm[:, j] = fi_static_matrix[torch.randperm(len(samples)), j]
        fi_importances[j] = _decoder_mse(perm) - baseline_fi_mse

    ranked_idx = np.argsort(-fi_importances)
    print("\nTop 20 most important features (individual):")
    print(f"  {'Feature':<55} {'ΔMSE':>10}")
    print(f"  {'-'*55} {'-'*10}")
    for k in ranked_idx[:20]:
        fname = feature_names_fi[k] if k < len(feature_names_fi) else f"feature_{k}"
        print(f"  {fname:<55} {fi_importances[k]:>10.6f}")

    # Grouped rollup
    cat_cols_fi = [
        "Protein_type",
        "Protein_class_type",
        "Buffer_type",
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
        "Excipient_type",
    ]
    grouped_imp = defaultdict(float)
    for k, imp in enumerate(fi_importances):
        fname = feature_names_fi[k] if k < len(feature_names_fi) else f"feature_{k}"
        if fname.startswith("cat__"):
            rest = fname[5:]
            matched = next(
                (
                    col
                    for col in cat_cols_fi
                    if rest.startswith(col + "_") or rest == col
                ),
                None,
            )
            grouped_imp[matched if matched else rest] += imp
        elif fname.startswith("num__"):
            grouped_imp[fname[5:]] += imp
        else:
            grouped_imp[fname] += imp

    grouped_ranked = sorted(grouped_imp.items(), key=lambda x: -x[1])
    print("\nGrouped feature importance (categoricals summed by column):")
    print(f"  {'Feature':<45} {'ΔMSE':>10}")
    print(f"  {'-'*45} {'-'*10}")
    for fname, imp in grouped_ranked:
        print(f"  {fname:<45} {imp:>10.6f}")

    fi_df = pd.DataFrame(
        {
            "Feature": [
                feature_names_fi[k] if k < len(feature_names_fi) else f"feature_{k}"
                for k in range(static_dim)
            ],
            "Importance_dMSE": fi_importances,
        }
    ).sort_values("Importance_dMSE", ascending=False)
    fi_df.to_csv(os.path.join(out, "feature_importance.csv"), index=False)

    fi_grp_df = pd.DataFrame(
        grouped_ranked, columns=["Feature_Group", "Importance_dMSE"]
    )
    fi_grp_df.to_csv(os.path.join(out, "feature_importance_grouped.csv"), index=False)
    print(f"\nFeature importance saved to {out}/feature_importance*.csv")
