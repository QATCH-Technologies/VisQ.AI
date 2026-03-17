"""
inference_cbm.py
================
Inference module for ConceptBottleneckCNP (CBM) viscosity predictor.

Fully backward-compatible with inference_o_net.py via a shared base class.
Auto-detects model type from checkpoint and returns the appropriate predictor.

Public API
----------
load_predictor(model_dir, verbose=False)
    Factory function. Returns ViscosityPredictorCBM if the checkpoint contains
    a ConceptBottleneckCNP, otherwise ViscosityPredictorCNP (plain CNP).
    Use this instead of instantiating either class directly.

ViscosityPredictorCNP  (base — identical to inference_o_net.py)
    .learn(df, n_draws=20, k=8)
    .predict(df) → pd.DataFrame
    .predict_with_uncertainty(df, n_samples=100) → (mean_pred, stats)

ViscosityPredictorCBM  (subclass — CBM-specific extensions)
    All methods above, plus:
    .get_concepts(df=None) → pd.DataFrame
        Returns named concept activations for current memory (or re-encodes
        from df if supplied).  Values in [-1, 1].

    .predict_with_concepts(df) → (results_df, concepts_df)
        predict() but also returns concept activations alongside predictions.

    .intervene(df, concept, value) → pd.DataFrame
        Predict with one concept clamped to `value` ∈ [-1, 1].
        `concept` can be a name (str) or index (int).

    .concept_sweep(df, concept, values=None) → pd.DataFrame
        Sweep `concept` across `values` (default: 21 steps in [-1, 1]).
        Returns table of mean log-viscosity and Δ relative to baseline,
        one row per (value × shear_rate).

    .explain(df=None) → str
        Human-readable summary of current concept activations, including
        proxy interpretations and sensitivity estimates where available.

Checkpoint compatibility
------------------------
Checkpoints saved by train_cbm_cnp_v2.py include:
  state_dict, config, static_dim, n_concepts, concept_names, model_class

Plain CNP checkpoints (train_cnp_3.py) only include:
  state_dict, config, static_dim

Both formats are handled automatically.
"""

import copy
import datetime
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. Logging
# ==========================================
log_filename = (
    f"debug_inference_cbm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("VisQ_CBM_Inference")


# ==========================================
# 1. Model Architecture (embedded for standalone use)
# ==========================================


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

    def forward(self, context_tensor, query_shear, query_static):
        r = self.pooler(self.encoder(context_tensor))
        n_q = query_shear.size(1)
        r_exp = r.unsqueeze(1).repeat(1, n_q, 1)
        return self.decoder(torch.cat([query_shear, query_static, r_exp], dim=-1))

    def encode_memory(self, context_tensor):
        return self.pooler(self.encoder(context_tensor))

    def decode_from_memory(self, memory_vector, query_shear, query_static):
        n_q = query_shear.size(1)
        r_exp = memory_vector.unsqueeze(1).repeat(1, n_q, 1)
        return self.decoder(torch.cat([query_shear, query_static, r_exp], dim=-1))


class ConceptBottleneckCNP(nn.Module):
    """
    Hard concept bottleneck: r → c = tanh(W·r + b) → decoder.
    encode_memory() returns c (concept vector), not r.
    decode_from_memory() takes concept vectors.
    Fully API-compatible with CrossSampleCNP for learn/predict.
    """

    def __init__(
        self,
        static_dim,
        hidden_dim=128,
        latent_dim=128,
        n_concepts=8,
        concept_names=None,
        dropout=0.0,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.concept_names = concept_names or [
            f"concept_{i}" for i in range(n_concepts)
        ]
        self.encoder = nn.Sequential(
            nn.Linear(2 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.pooler = AttentionPool(latent_dim)
        self.concept_proj = nn.Linear(latent_dim, n_concepts)
        self.decoder = nn.Sequential(
            nn.Linear(1 + static_dim + n_concepts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _project_concepts(self, r):
        return torch.tanh(self.concept_proj(r))

    def forward(self, context_tensor, query_shear, query_static):
        r = self.pooler(self.encoder(context_tensor))
        c = self._project_concepts(r)
        n_q = query_shear.size(1)
        c_exp = c.unsqueeze(1).repeat(1, n_q, 1)
        return self.decoder(torch.cat([query_shear, query_static, c_exp], dim=-1))

    def encode_latent(self, context_tensor):
        """Raw latent r — for diagnostics only."""
        return self.pooler(self.encoder(context_tensor))

    def encode_memory(self, context_tensor):
        """Returns concept vector c [B, n_concepts] in [-1, 1]."""
        return self._project_concepts(self.pooler(self.encoder(context_tensor)))

    def decode_from_memory(self, concept_vector, query_shear, query_static):
        n_q = query_shear.size(1)
        c_exp = concept_vector.unsqueeze(1).repeat(1, n_q, 1)
        return self.decoder(torch.cat([query_shear, query_static, c_exp], dim=-1))


# ==========================================
# 2. Configuration constants (UNCHANGED from inference_o_net.py)
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

# Human-readable descriptions for the v2 supervised concepts
CONCEPT_DESCRIPTIONS = {
    "self_interaction": "Protein self-interaction propensity (kP proxy). "
    "High → attractive interactions → elevated viscosity.",
    "hydrophobicity": "Hydrophobic patch exposure (HCI proxy). "
    "High → surface hydrophobicity → aggregation risk.",
    "charge_environment": "Net charge-concentration class (C_Class proxy). "
    "High → near-pI → stronger protein-protein attraction.",
    "ionic_screening": "Ionic screening of charge repulsion (√[NaCl] proxy). "
    "High → screened repulsion → potential viscosity increase.",
    "crowding": "Macromolecular crowding from protein concentration. "
    "High → strong excluded-volume effects.",
    "nonlinear_conc": "Non-linear concentration dependence (conc² proxy). "
    "High → super-linear viscosity growth regime.",
    "cosolute_interaction": "Protein–stabilizer cross-interaction (conc × stabilizer). "
    "Negative → combined cosolute reduces viscosity.",
    "cosolute_protection": "Stabilizer concentration effect. "
    "Negative → higher stabilizer loading reduces viscosity.",
}


# ==========================================
# 3. Base predictor (identical logic to inference_o_net.py)
# ==========================================


class ViscosityPredictorCNP:
    """
    Base predictor for CrossSampleCNP.
    Identical public API to inference_o_net.py: learn / predict / predict_with_uncertainty.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self, model_dir: str, verbose: bool = False, _defer_model_load: bool = False
    ):
        """
        _defer_model_load: internal flag used by ViscosityPredictorCBM.
        When True, skips CrossSampleCNP construction so the subclass can
        build ConceptBottleneckCNP instead. Do not set from user code.
        """
        self._logger = logging.getLogger(f"VisQ_CBM.{id(self)}")
        if not verbose:
            self._logger.setLevel(logging.CRITICAL)
        self._logger.info(f"Initializing ViscosityPredictorCNP: {model_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir

        # Preprocessors
        for attr, fname in [
            ("preprocessor_path", "preprocessor.pkl"),
            ("scaler_path", "physics_scaler.pkl"),
        ]:
            path = os.path.join(model_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{fname} not found at {path}")
            setattr(self, attr, path)

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.physics_scaler = joblib.load(self.scaler_path)

        # Checkpoint
        self.model_path = os.path.join(model_dir, "best_model.pth")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.config = checkpoint["config"]
        self.static_dim = checkpoint["static_dim"]

        # Build model — skipped when called from ViscosityPredictorCBM
        # (_defer_model_load=True), which builds ConceptBottleneckCNP itself.
        if not _defer_model_load:
            self.model = CrossSampleCNP(
                static_dim=self.static_dim,
                hidden_dim=self.config["hidden_dim"],
                latent_dim=self.config["latent_dim"],
                dropout=self.config.get("dropout", 0.0),
            ).to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self._original_state = copy.deepcopy(self.model.state_dict())
        else:
            # Subclass will build and load the model; set safe placeholders
            # so any accidental early access raises a clear AttributeError.
            self.model = None
            self._original_state = None

        # Memory state
        self.memory_vector: Optional[torch.Tensor] = None
        self.context_t: Optional[torch.Tensor] = None

        # Memory dim for zero-shot fallback = latent_dim for plain CNP
        self._memory_dim = self.config["latent_dim"]

        # Column definitions
        self.shear_map = {
            "Viscosity_100": 100.0,
            "Viscosity_1000": 1000.0,
            "Viscosity_10000": 10000.0,
            "Viscosity_100000": 100000.0,
            "Viscosity_15000000": 1.5e7,
        }
        self.cat_cols = [
            "Protein_type",
            "Protein_class_type",
            "Buffer_type",
            "Salt_type",
            "Stabilizer_type",
            "Surfactant_type",
            "Excipient_type",
        ]
        self.new_prior_cols = [
            "prior_arginine",
            "prior_lysine",
            "prior_proline",
            "prior_nacl",
            "prior_stabilizer",
            "prior_tween-20",
            "prior_tween-80",
        ]
        self.new_conc_cols = []
        for k in CONC_THRESHOLDS:
            self.new_conc_cols += [f"{k}_low", f"{k}_high"]

        # ------------------------------------------------------------------
        # Validate that the preprocessor output width matches static_dim.
        # These must agree: the preprocessor.pkl and best_model.pth must come
        # from the SAME training run. A mismatch means they are from different
        # runs (e.g. preprocessor from raw-data training, model from augmented-
        # data training, or vice versa).
        #
        # We measure the true preprocessor output width by building a minimal
        # dummy DataFrame with all expected column names set to zero / "unknown",
        # then calling transform(). This is fast (1-row probe) and requires no
        # real data.
        # ------------------------------------------------------------------
        self._preprocessor_dim = self._probe_preprocessor_dim()
        if self._preprocessor_dim != self.static_dim:
            msg = (
                f"\n\n{'='*64}\n"
                f"  PREPROCESSING DIMENSION MISMATCH\n"
                f"{'='*64}\n"
                f"  checkpoint static_dim : {self.static_dim}\n"
                f"  preprocessor output   : {self._preprocessor_dim}\n"
                f"\n"
                f"  The preprocessor.pkl and best_model.pth in:\n"
                f"    {model_dir}\n"
                f"  were saved by DIFFERENT training runs.\n"
                f"\n"
                f"  Likely cause: the model was retrained on augmented data\n"
                f"  (which has more OHE categories → larger static_dim), but\n"
                f"  the preprocessor in this directory is from an older run on\n"
                f"  raw data (fewer OHE categories → smaller static_dim).\n"
                f"\n"
                f"  FIX: re-run training with the same output directory so\n"
                f"  both files are regenerated from the same data.\n"
                f"  As a temporary workaround, _preprocess() will pad or trim\n"
                f"  static features to {self.static_dim} columns, but predictions\n"
                f"  may be degraded for the padded/trimmed dimensions.\n"
                f"{'='*64}\n"
            )
            print(msg)
            self._logger.warning(msg)

        self._logger.info(
            f"Model loaded. static_dim={self.static_dim}, "
            f"preprocessor_dim={self._preprocessor_dim}, "
            f"memory_dim={self._memory_dim}, device={self.device}"
        )

    def _probe_preprocessor_dim(self) -> int:
        """
        Measure the actual output width of the saved preprocessor by running
        a 1-row dummy transform. All numerics = 0.0, all categoricals = 'unknown'.
        Returns the number of features the preprocessor outputs.
        """
        try:
            # Build a minimal dummy row covering every column the ColumnTransformer
            # was trained on. Numeric cols → 0.0, categorical cols → 'unknown'.
            dummy = {}
            if hasattr(self.preprocessor, "feature_names_in_"):
                for col in self.preprocessor.feature_names_in_:
                    dummy[col] = [0.0]
            else:
                # Fallback: populate every known column name
                for col in (
                    [
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
                    + self.new_prior_cols
                    + self.new_conc_cols
                    + [
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
                ):
                    dummy[col] = [0.0]
                for col in self.cat_cols:
                    dummy[col] = ["unknown"]

            df_probe = pd.DataFrame(dummy)
            result = self.preprocessor.transform(df_probe)
            return result.shape[1]
        except Exception as e:
            self._logger.warning(
                f"_probe_preprocessor_dim failed ({e}), assuming static_dim."
            )
            return self.static_dim

    # ------------------------------------------------------------------
    # Physics helpers (UNCHANGED from inference_o_net.py)
    # ------------------------------------------------------------------
    def _calculate_cci(self, row) -> float:
        try:
            c_class = float(row.get("C_Class", 1.0))
            ph = float(row.get("Buffer_pH", 7.0))
            pi = float(row.get("PI_mean", 7.0))
        except ValueError:
            c_class, ph, pi = 1.0, 7.0, 7.0
        ph = 7.0 if pd.isna(ph) else ph
        pi = 7.0 if pd.isna(pi) else pi
        return c_class * np.exp(-abs(ph - pi) / 1.5)

    def _calculate_physics_features(self, row) -> dict:
        cci = self._calculate_cci(row)
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
        for key in PRIOR_TABLE:
            if key != "default" and key in p_type:
                lookup_key = key
                break
        regime_dict = PRIOR_TABLE[lookup_key].get(
            regime, PRIOR_TABLE[lookup_key]["Far"]
        )

        priors = {k: 0.0 for k in self.new_prior_cols}
        concs = {k: 0.0 for k in self.new_conc_cols}

        for type_col, conc_col in [
            ("Salt_type", "Salt_conc"),
            ("Stabilizer_type", "Stabilizer_conc"),
            ("Excipient_type", "Excipient_conc"),
            ("Surfactant_type", "Surfactant_conc"),
        ]:
            ing_name = str(row.get(type_col, "none")).lower()
            try:
                ing_conc = float(row.get(conc_col, 0.0))
            except Exception:
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

    def _preprocess(
        self, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._logger.debug(f"Preprocessing {len(df)} rows...")
        df_proc = df.copy()

        for col in df_proc.select_dtypes(include=["object"]):
            df_proc[col] = df_proc[col].apply(
                lambda x: x.value if hasattr(x, "value") else x
            )
        if "ID" in df_proc.columns:
            df_proc.drop(columns=["ID"], inplace=True)

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
            df_proc[c] = df_proc[c].fillna(0.0) if c in df_proc.columns else 0.0

        MW_MAP = {
            "sucrose": 342.3,
            "trehalose": 342.3,
            "arginine": 174.2,
            "proline": 115.13,
            "lysine": 149.19,
            "nacl": 58.44,
            "default_sugar": 342.3,
        }

        def get_mw(series, default=342.3):
            return (
                series.astype(str)
                .str.lower()
                .map(
                    lambda x: next((mw for n, mw in MW_MAP.items() if n in x), default)
                )
            )

        stab_mw = get_mw(
            df_proc.get("Stabilizer_type", pd.Series(["unknown"] * len(df_proc))), 342.3
        )
        df_proc["Stabilizer_mg_mL"] = df_proc["Stabilizer_conc"] * stab_mw
        salt_mw = get_mw(
            df_proc.get("Salt_type", pd.Series(["unknown"] * len(df_proc))), 58.44
        )
        df_proc["Salt_mg_mL"] = df_proc["Salt_conc"] * salt_mw / 1000.0
        excip_mw = get_mw(
            df_proc.get("Excipient_type", pd.Series(["unknown"] * len(df_proc))), 150.0
        )
        df_proc["Excipient_mg_mL"] = df_proc["Excipient_conc"] * excip_mw / 1000.0
        df_proc["Surfactant_mg_mL"] = df_proc["Surfactant_conc"] * 10.0

        df_proc["log_conc"] = np.log1p(df_proc["Protein_conc"])
        df_proc["conc_sq"] = df_proc["Protein_conc"] ** 2
        df_proc["conc_x_kP"] = df_proc["Protein_conc"] * df_proc["kP"]
        df_proc["conc_x_HCI"] = df_proc["Protein_conc"] * df_proc["HCI"]
        df_proc["Crowding_Index"] = (
            df_proc["Protein_conc"] * df_proc["Stabilizer_mg_mL"]
        )
        df_proc["Stabilizer_Squared"] = df_proc["Stabilizer_mg_mL"] ** 2
        df_proc["Total_Solute_Mass"] = (
            df_proc["Protein_conc"]
            + df_proc["Stabilizer_mg_mL"]
            + df_proc["Excipient_mg_mL"]
            + df_proc["Salt_mg_mL"]
            + df_proc["Surfactant_mg_mL"]
        )
        df_proc["Effective_Protein_Fraction"] = df_proc["Protein_conc"] / df_proc[
            "Total_Solute_Mass"
        ].replace(0, 1e-6)

        VBP, VBS, VBSa, VBE = 0.73 / 1000, 0.62 / 1000, 0.30 / 1000, 0.70 / 1000
        df_proc["Phi_Protein"] = df_proc["Protein_conc"] * VBP
        df_proc["Phi_Stabilizer"] = df_proc["Stabilizer_mg_mL"] * VBS
        df_proc["Phi_Salt"] = df_proc["Salt_mg_mL"] * VBSa
        df_proc["Phi_Excipient"] = df_proc["Excipient_mg_mL"] * VBE
        df_proc["Phi_Total"] = (
            df_proc["Phi_Protein"]
            + df_proc["Phi_Stabilizer"]
            + df_proc["Phi_Salt"]
            + df_proc["Phi_Excipient"]
        )

        safe_phi = df_proc["Phi_Total"].clip(upper=0.64)
        df_proc["KD_Asymptote"] = (1.0 - safe_phi / 0.65) ** -2.0
        df_proc["Exp_Crowding"] = np.exp(safe_phi * 2.5)
        df_proc["Ionic_Strength_Proxy"] = np.sqrt(df_proc["Salt_conc"] / 1000.0)

        for c in self.cat_cols:
            df_proc[c] = (
                df_proc[c].astype(str).str.lower().replace("nan", "unknown")
                if c in df_proc.columns
                else "unknown"
            )

        new_features = df_proc.apply(
            self._calculate_physics_features, axis=1, result_type="expand"
        )
        df_proc = pd.concat([df_proc, new_features], axis=1)

        feature_names = (
            self.preprocessor.feature_names_in_
            if hasattr(self.preprocessor, "feature_names_in_")
            else []
        )
        expected_missing = ["ID"] + list(self.shear_map.keys())
        for col in feature_names:
            if col not in df_proc.columns and col not in expected_missing:
                df_proc[col] = 0.0

        X_static = self.preprocessor.transform(df_proc)
        if np.isnan(X_static).any():
            X_static = np.nan_to_num(X_static)

        # Guard: align preprocessor output to the model's expected static_dim.
        # This silently handles the case where preprocessor.pkl and best_model.pth
        # come from runs with different static_dims (diagnosed at __init__ time).
        actual_dim = X_static.shape[1]
        if actual_dim != self.static_dim:
            if actual_dim < self.static_dim:
                # Pad with zeros on the right (missing OHE categories → zero columns)
                pad = np.zeros(
                    (X_static.shape[0], self.static_dim - actual_dim),
                    dtype=X_static.dtype,
                )
                X_static = np.hstack([X_static, pad])
            else:
                # Truncate (extra features the model was not trained on)
                X_static = X_static[:, : self.static_dim]

        n_rows, n_shears = len(df_proc), len(self.shear_map)
        raw_points = np.empty((n_rows * n_shears, 2), dtype=np.float64)
        static_list = []
        row_idx = 0
        for i in range(n_rows):
            for col, shear_val in self.shear_map.items():
                val = 1.0
                if col in df_proc.columns and pd.notna(df_proc.iloc[i][col]):
                    val = float(df_proc.iloc[i][col])
                raw_points[row_idx, 0] = np.log10(shear_val)
                raw_points[row_idx, 1] = np.log10(max(val, 1e-6))
                static_list.append(X_static[i])
                row_idx += 1

        scaled_points = self.physics_scaler.transform(raw_points)

        static_t = (
            torch.tensor(np.array(static_list), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        points_t = (
            torch.tensor(scaled_points.astype(np.float32)).unsqueeze(0).to(self.device)
        )
        shear_t = points_t[:, :, [0]]
        visc_t = points_t[:, :, [1]]
        return static_t, shear_t, visc_t

    def _inverse_to_log(
        self, q_shear: torch.Tensor, out_scaled: torch.Tensor
    ) -> np.ndarray:
        q_np = q_shear.cpu().numpy().reshape(-1, 1)
        out_np = out_scaled.cpu().numpy().reshape(-1, 1)
        return self.physics_scaler.inverse_transform(np.hstack([q_np, out_np]))[:, 1]

    def _zero_shot_memory(self) -> torch.Tensor:
        """Zero vector of appropriate memory dimension."""
        return torch.zeros((1, self._memory_dim), device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def learn(
        self,
        df: pd.DataFrame,
        steps: int = 50,
        lr: float = 1e-3,
        n_draws: int = 20,
        k: int = 8,
    ):
        """
        Encode context samples into a memory vector. No weight updates.
        Memory is the latent vector r for CrossSampleCNP,
        or the concept vector c for ConceptBottleneckCNP.
        """
        if df.empty:
            print("Warning: Context DataFrame is empty. Skipping learn.")
            return

        print(f" > Encoding context: {len(df)} samples, {n_draws} draws of k={k}...")
        self.model.load_state_dict(self._original_state)

        static_t, shear_t, visc_t = self._preprocess(df)
        context_t = torch.cat([shear_t, visc_t, static_t], dim=-1)
        self.context_t = context_t

        n_ctx = context_t.size(1)
        k_eff = min(k, n_ctx)
        self.model.eval()

        with torch.no_grad():
            if n_ctx <= k_eff:
                self.memory_vector = self.model.encode_memory(context_t)
                print(
                    f" > Encoding complete (single pass). Memory shape: {self.memory_vector.shape}"
                )
                return

            draws = []
            for _ in range(n_draws):
                idx = torch.randperm(n_ctx, device=self.device)[:k_eff]
                subset = context_t[:, idx, :]
                draws.append(self.model.encode_memory(subset))

        self.memory_vector = torch.stack(draws, dim=0).mean(dim=0)
        print(
            f" > Encoding complete. Memory norm: {self.memory_vector.norm().item():.3f}"
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict viscosity at 5 shear rates using the cached memory vector."""
        memory = (
            self.memory_vector
            if self.memory_vector is not None
            else self._zero_shot_memory()
        )
        q_static, q_shear, _ = self._preprocess(df)

        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model.decode_from_memory(memory, q_shear, q_static)

        pred_log = self._inverse_to_log(q_shear, y_scaled)
        pred_visc = np.power(10, pred_log)

        results = df.copy()
        n_shears = len(self.shear_map)
        shear_keys = list(self.shear_map.keys())
        new_cols = {k: [] for k in shear_keys}
        for i in range(len(df)):
            s = pred_visc[i * n_shears : (i + 1) * n_shears]
            for j, key in enumerate(shear_keys):
                new_cols[key].append(s[j])
        for k, v in new_cols.items():
            results[f"Pred_{k}"] = v
        return results

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        n_samples: int = 100,
        ci_range: Tuple[float, float] = (2.5, 97.5),
        k: Optional[int] = None,
    ):
        """
        MC Dropout uncertainty. Returns (mean_pred_cP, stats_dict).
        stats keys: mean_log10, std_log10, lower_ci, upper_ci.
        """
        if self.config.get("dropout", 0.0) == 0.0:
            print(
                "WARNING: dropout=0.0 — CI will be zero-width. Retrain with dropout>0."
            )

        memory = (
            self.memory_vector
            if self.memory_vector is not None
            else self._zero_shot_memory()
        )
        q_static, q_shear, _ = self._preprocess(df)

        self.model.train()  # activates dropout
        preds_log = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.model.decode_from_memory(memory, q_shear, q_static)
                preds_log.append(self._inverse_to_log(q_shear, out))
        self.model.eval()

        stack = np.stack(preds_log)
        mean_l = np.mean(stack, axis=0)
        std_l = np.std(stack, axis=0)
        lo_l = np.percentile(stack, ci_range[0], axis=0)
        hi_l = np.percentile(stack, ci_range[1], axis=0)

        return np.power(10, mean_l), {
            "mean_log10": mean_l,
            "std_log10": std_l,
            "lower_ci": np.power(10, lo_l),
            "upper_ci": np.power(10, hi_l),
        }


# ==========================================
# 4. CBM Predictor subclass
# ==========================================


class ViscosityPredictorCBM(ViscosityPredictorCNP):
    """
    Extends ViscosityPredictorCNP with concept-level capabilities.

    All base-class methods (learn, predict, predict_with_uncertainty) work
    identically — the concept bottleneck is transparent at that level.

    Additional methods:
        get_concepts(df=None)
        predict_with_concepts(df)
        intervene(df, concept, value)
        concept_sweep(df, concept, values=None)
        explain(df=None)
    """

    def __init__(self, model_dir: str, verbose: bool = False):
        # Pass _defer_model_load=True so the base class sets up all preprocessors,
        # config, and column definitions WITHOUT building CrossSampleCNP — we build
        # ConceptBottleneckCNP below with the correct architecture.
        super().__init__(model_dir, verbose, _defer_model_load=True)

        # Re-load checkpoint to extract CBM-specific fields
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.n_concepts = checkpoint["n_concepts"]
        self.concept_names = checkpoint["concept_names"]

        # Rebuild model as ConceptBottleneckCNP
        self.model = ConceptBottleneckCNP(
            static_dim=self.static_dim,
            hidden_dim=self.config["hidden_dim"],
            latent_dim=self.config["latent_dim"],
            n_concepts=self.n_concepts,
            concept_names=self.concept_names,
            dropout=self.config.get("dropout", 0.0),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self._original_state = copy.deepcopy(self.model.state_dict())

        # Memory dim for zero-shot fallback = n_concepts (not latent_dim)
        self._memory_dim = self.n_concepts

        # Optional: concept proxy normalization stats saved by train_cbm_cnp_v2.py
        self._concept_proxy_mean = None
        self._concept_proxy_std = None
        self._concept_proxy_signs = None
        for attr, fname in [
            ("_concept_proxy_mean", "concept_proxy_mean.npy"),
            ("_concept_proxy_std", "concept_proxy_std.npy"),
            ("_concept_proxy_signs", "concept_proxy_signs.npy"),
        ]:
            path = os.path.join(model_dir, fname)
            if os.path.exists(path):
                setattr(self, attr, np.load(path))

        self._logger.info(
            f"CBM loaded. n_concepts={self.n_concepts}, "
            f"concept_names={self.concept_names}"
        )
        print(
            f" > ConceptBottleneckCNP loaded: {self.n_concepts} concepts "
            f"({len(self.concept_names)} named)."
        )

    # ------------------------------------------------------------------
    # Internal: concept vector from memory or zero-shot
    # ------------------------------------------------------------------
    def _current_concept_vector(self) -> torch.Tensor:
        """Return the stored concept vector, or zeros if learn() not called."""
        if self.memory_vector is not None:
            return self.memory_vector  # already a concept vector [1, n_concepts]
        return torch.zeros((1, self.n_concepts), device=self.device)

    def _resolve_concept(self, concept: Union[str, int]) -> int:
        """Resolve a concept name or index to an integer index."""
        if isinstance(concept, int):
            if not 0 <= concept < self.n_concepts:
                raise IndexError(
                    f"Concept index {concept} out of range [0, {self.n_concepts - 1}]."
                )
            return concept
        if isinstance(concept, str):
            try:
                return self.concept_names.index(concept)
            except ValueError:
                close = [n for n in self.concept_names if concept.lower() in n.lower()]
                hint = f" Did you mean: {close}?" if close else ""
                raise ValueError(
                    f"Concept '{concept}' not found in {self.concept_names}.{hint}"
                )
        raise TypeError(f"concept must be str or int, got {type(concept)}")

    # ------------------------------------------------------------------
    # get_concepts
    # ------------------------------------------------------------------
    def get_concepts(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Return named concept activations as a single-row DataFrame.

        If `df` is provided, re-encodes from that context (but does NOT
        update self.memory_vector). Pass your own context to compare
        concept signatures across different formulation families without
        calling learn() repeatedly.

        If `df` is None, returns activations from the current memory_vector
        (i.e., whatever was last set by learn()).

        Returns
        -------
        pd.DataFrame with columns = concept_names, one row, values in [-1, 1].
        """
        if df is not None:
            static_t, shear_t, visc_t = self._preprocess(df)
            ctx = torch.cat([shear_t, visc_t, static_t], dim=-1)
            self.model.eval()
            with torch.no_grad():
                c = self.model.encode_memory(ctx)
        else:
            c = self._current_concept_vector()

        c_np = c.squeeze(0).cpu().numpy()
        return pd.DataFrame([c_np], columns=self.concept_names)

    # ------------------------------------------------------------------
    # predict_with_concepts
    # ------------------------------------------------------------------
    def predict_with_concepts(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        predict() but also return the concept activations.

        Returns
        -------
        results_df   : same as predict() — DataFrame with Pred_Viscosity_* columns
        concepts_df  : DataFrame with concept_names columns, one row per sample
                       (concept vector is the same for all rows in a batch since
                       it's derived from the shared memory_vector)
        """
        results = self.predict(df)
        concepts = self.get_concepts()  # uses current memory
        concepts_df = pd.concat([concepts] * len(df), ignore_index=True)
        # Add sample ID if available
        if "ID" in df.columns:
            concepts_df.insert(0, "ID", df["ID"].values)
        return results, concepts_df

    # ------------------------------------------------------------------
    # intervene
    # ------------------------------------------------------------------
    def intervene(
        self,
        df: pd.DataFrame,
        concept: Union[str, int],
        value: float,
    ) -> pd.DataFrame:
        """
        Predict with one concept clamped to `value` ∈ [-1, 1].

        This is a causal do(c_i = value) intervention — not a correlation.
        All other concepts retain their current memory values.

        Parameters
        ----------
        df      : query samples (same format as predict())
        concept : concept name (str) or index (int)
        value   : activation value to clamp to, in [-1, 1]

        Returns
        -------
        pd.DataFrame identical to predict() but with an extra column
        'Intervened_concept' and 'Intervened_value' for auditability.
        """
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"Concept value must be in [-1, 1], got {value}.")
        ci = self._resolve_concept(concept)
        cname = self.concept_names[ci]

        c_mod = self._current_concept_vector().clone()
        c_mod[:, ci] = value

        q_static, q_shear, _ = self._preprocess(df)
        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model.decode_from_memory(c_mod, q_shear, q_static)

        pred_log = self._inverse_to_log(q_shear, y_scaled)
        pred_visc = np.power(10, pred_log)

        results = df.copy()
        n_shears = len(self.shear_map)
        shear_keys = list(self.shear_map.keys())
        for i in range(len(df)):
            s = pred_visc[i * n_shears : (i + 1) * n_shears]
            for j, key in enumerate(shear_keys):
                results.loc[results.index[i], f"Pred_{key}"] = s[j]

        results["Intervened_concept"] = cname
        results["Intervened_value"] = value
        return results

    # ------------------------------------------------------------------
    # concept_sweep
    # ------------------------------------------------------------------
    def concept_sweep(
        self,
        df: pd.DataFrame,
        concept: Union[str, int],
        values: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Sweep a single concept across `values` and record log-viscosity
        and Δlog-viscosity relative to baseline (current memory).

        Parameters
        ----------
        df      : query samples — typically a single representative formulation
        concept : concept name or index
        values  : array of sweep values in [-1, 1].
                  Default: 21 evenly-spaced steps from -1 to +1.

        Returns
        -------
        pd.DataFrame with columns:
            Concept, Value, Shear_Rate,
            Log_Viscosity, Linear_Viscosity_cP,
            Delta_Log_Visc (relative to baseline at current memory value),
            Baseline_Log_Visc
        """
        if values is None:
            values = np.linspace(-1.0, 1.0, 21)

        ci = self._resolve_concept(concept)
        cname = self.concept_names[ci]

        # Baseline (no intervention)
        q_static, q_shear, _ = self._preprocess(df)
        c_base = self._current_concept_vector()

        self.model.eval()
        with torch.no_grad():
            base_scaled = self.model.decode_from_memory(c_base, q_shear, q_static)
        base_log = self._inverse_to_log(q_shear, base_scaled)

        # Mean over all query rows for a clean single-line output
        n_shears = len(self.shear_map)
        shear_vals = list(self.shear_map.values())
        shear_keys = list(self.shear_map.keys())

        # mean baseline per shear rate across df rows
        baseline_per_shear = np.array(
            [
                np.mean([base_log[i * n_shears + j] for i in range(len(df))])
                for j in range(n_shears)
            ]
        )

        records = []
        with torch.no_grad():
            for v in values:
                c_mod = c_base.clone()
                c_mod[:, ci] = float(v)
                y_sc = self.model.decode_from_memory(c_mod, q_shear, q_static)
                log_v = self._inverse_to_log(q_shear, y_sc)

                for j in range(n_shears):
                    mean_log = np.mean(
                        [log_v[i * n_shears + j] for i in range(len(df))]
                    )
                    records.append(
                        {
                            "Concept": cname,
                            "Value": round(float(v), 4),
                            "Shear_Rate": shear_vals[j],
                            "Log_Viscosity": round(float(mean_log), 5),
                            "Linear_Viscosity_cP": round(float(10**mean_log), 4),
                            "Delta_Log_Visc": round(
                                float(mean_log - baseline_per_shear[j]), 5
                            ),
                            "Baseline_Log_Visc": round(float(baseline_per_shear[j]), 5),
                        }
                    )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # explain
    # ------------------------------------------------------------------
    def explain(self, df: Optional[pd.DataFrame] = None) -> str:
        """
        Human-readable interpretation of current concept activations.

        If `df` is provided, re-encodes from that context for the explanation
        without modifying self.memory_vector.

        Returns a formatted string suitable for printing or logging.
        Lines are also written to the instance logger at INFO level.
        """
        concepts_df = self.get_concepts(df)
        c_vals = concepts_df.iloc[0].to_dict()

        lines = ["=" * 62, "  Concept Activation Report", "=" * 62]

        # --- Supervised concepts ---
        lines.append("  Named / supervised concepts:")
        lines.append(f"  {'Concept':<26} {'Value':>7}  {'Interpretation'}")
        lines.append(f"  {'-'*26} {'-'*7}  {'-'*26}")

        for cname, val in c_vals.items():
            if cname.startswith("latent_"):
                continue
            bar_len = int(abs(val) * 12)
            bar_sign = "+" if val >= 0 else "-"
            bar = bar_sign * bar_len + " " * (12 - bar_len)
            direction = "high" if val > 0.3 else ("low" if val < -0.3 else "neutral")
            desc_short = CONCEPT_DESCRIPTIONS.get(cname, "")
            # Pick first clause
            desc_first = desc_short.split(".")[0] if desc_short else ""
            lines.append(f"  {cname:<26} {val:>+7.3f}  [{bar}]  {direction}")
            if desc_first:
                lines.append(f"  {'':26}           → {desc_first}")

        # --- Free latents ---
        free = {k: v for k, v in c_vals.items() if k.startswith("latent_")}
        if free:
            lines.append("")
            lines.append("  Free (unsupervised) latents:")
            sat_count = sum(1 for v in free.values() if abs(v) > 0.85)
            if sat_count > 0:
                lines.append(
                    f"  Note: {sat_count}/{len(free)} free concepts near saturation "
                    "(|c| > 0.85) — likely encoding protein-type identity."
                )
            for cname, val in free.items():
                bar_len = int(abs(val) * 12)
                bar = ("+" if val >= 0 else "-") * bar_len
                lines.append(f"  {cname:<26} {val:>+7.3f}  [{bar:<12}]")

        # --- State summary ---
        lines.append("")
        if self.memory_vector is not None:
            lines.append("  Memory state: context-encoded (learn() was called).")
        else:
            lines.append("  Memory state: zero-shot (learn() not yet called).")
        lines.append("=" * 62)

        report = "\n".join(lines)
        self._logger.info(report)
        return report

    # ------------------------------------------------------------------
    # Utility: compare two context DataFrames' concept signatures
    # ------------------------------------------------------------------
    def compare_concepts(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        label_a: str = "A",
        label_b: str = "B",
    ) -> pd.DataFrame:
        """
        Compare concept activations between two context sets side by side.

        Neither call modifies self.memory_vector.

        Returns
        -------
        pd.DataFrame with columns: concept, label_a_value, label_b_value, delta (B-A)
        """
        ca = self.get_concepts(df_a).iloc[0]
        cb = self.get_concepts(df_b).iloc[0]

        rows = []
        for cname in self.concept_names:
            rows.append(
                {
                    "concept": cname,
                    f"{label_a}_value": round(float(ca[cname]), 4),
                    f"{label_b}_value": round(float(cb[cname]), 4),
                    "delta (B-A)": round(float(cb[cname] - ca[cname]), 4),
                }
            )
        return pd.DataFrame(rows)


# ==========================================
# 5. Factory function
# ==========================================


def load_predictor(
    model_dir: str, verbose: bool = False
) -> Union["ViscosityPredictorCBM", "ViscosityPredictorCNP"]:
    """
    Auto-detects model type from checkpoint and returns the right predictor.

    Returns ViscosityPredictorCBM if checkpoint contains model_class='ConceptBottleneckCNP'
    and the n_concepts / concept_names fields. Returns ViscosityPredictorCNP otherwise.

    Usage
    -----
    predictor = load_predictor("models/experiments/cbm_cnp_v2")
    predictor.learn(context_df)
    results = predictor.predict(query_df)

    # CBM-only extras (safe — no-op if plain CNP returned):
    if isinstance(predictor, ViscosityPredictorCBM):
        print(predictor.explain())
        sweep = predictor.concept_sweep(query_df, "self_interaction")
    """
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    is_cbm = (
        ckpt.get("model_class") == "ConceptBottleneckCNP"
        and "n_concepts" in ckpt
        and "concept_names" in ckpt
    )

    if is_cbm:
        print(
            f"Detected ConceptBottleneckCNP checkpoint — loading ViscosityPredictorCBM."
        )
        return ViscosityPredictorCBM(model_dir, verbose=verbose)
    else:
        print(f"Detected CrossSampleCNP checkpoint — loading ViscosityPredictorCNP.")
        return ViscosityPredictorCNP(model_dir, verbose=verbose)


# ==========================================
# 6. __main__ test harness
# ==========================================

if __name__ == "__main__":
    import io

    model_dir = "models/experiments/cbm_cnp_v2"
    training_file = "data/raw/formulation_data_03042026.csv"

    # -----------------------------------------------------------------------
    # 1. Load predictor (auto-detects CBM vs plain CNP)
    # -----------------------------------------------------------------------
    try:
        predictor = load_predictor(model_dir)
    except FileNotFoundError as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    is_cbm = isinstance(predictor, ViscosityPredictorCBM)

    if not os.path.exists(training_file):
        print(f"Error: Training file not found at {training_file}")
        sys.exit(1)

    print(f"\nLoading context pool from {training_file}...")
    full_train_df = pd.read_csv(training_file)
    for col in full_train_df.select_dtypes(include=["int64", "int32"]).columns:
        if col != "ID":
            full_train_df[col] = full_train_df[col].astype(float)
    full_train_df["ID"] = full_train_df["ID"].astype(str)

    # -----------------------------------------------------------------------
    # 2. Target samples (same as inference_o_net.py test harness)
    # -----------------------------------------------------------------------
    target_data = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
511,poly-hIgG,Polyclonal,3.0,150.0,7.6,1.0,83.0,25.0,Histidine,6.0,15.0,None,0.0,Sucrose,0.25,None,0.0,None,0.0,0.9,0.9,4.15,4.24,4.22,3.89,1.72
510,poly-hIgG,Polyclonal,3.0,150.0,7.6,1.0,242.0,25.6,Histidine,6.0,15.0,None,0.0,Sucrose,0.25,None,0.0,None,0.0,0.9,0.9,29.38,30.34,31.30,28.07,6.11
630,Adalimumab,mAb_IgG1,3.0,148.0,8.7,0.3,206.0,25.35,Histidine,6.0,15.0,NaCl,70.0,Sucrose,0.2,tween-80,0.05,None,0.0,1.0,1.0,36.41,38.01,39.60,40.87,6.36
631,Adalimumab,mAb_IgG1,3.0,148.0,8.7,0.3,206.0,25.32,Histidine,6.0,15.0,NaCl,70.0,Sucrose,0.2,tween-80,0.05,None,0.0,1.0,1.0,52.20,52.12,53.14,47.45,8.14"""

    target_df = pd.read_csv(io.StringIO(target_data))
    target_df["ID"] = target_df["ID"].astype(str)

    # -----------------------------------------------------------------------
    # 3. Predict + concept report per protein group
    # -----------------------------------------------------------------------
    all_results = []
    shear_cols = ["100", "1000", "10000", "100000", "15000000"]

    for protein in target_df["Protein_type"].unique():
        print(f"\n{'='*60}")
        print(f"  Group: {protein}")
        print(f"{'='*60}")

        prot_target_df = target_df[target_df["Protein_type"] == protein].copy()
        target_ids = prot_target_df["ID"].tolist()

        history_df = full_train_df[
            (full_train_df["Protein_type"] == protein)
            & (~full_train_df["ID"].isin(target_ids))
        ].copy()

        predictor.memory_vector = None
        predictor.context_t = None

        if not history_df.empty:
            print(f"Adapting to {protein} ({len(history_df)} context samples)...")
            predictor.learn(history_df)
        else:
            print(f"No history for {protein} — zero-shot prediction.")

        # -- Concept report (CBM only) --
        if is_cbm:
            print(predictor.explain())

        # -- Predictions --
        if is_cbm:
            results_df, concepts_df = predictor.predict_with_concepts(prot_target_df)
            print("\n  Concept activations for this prediction:")
            print(concepts_df.to_string(index=False))
        else:
            results_df = predictor.predict(prot_target_df)

        all_results.append(results_df)

        # -- Parity table --
        print(
            f"\n  {'ID':>6} | {'Shear':>12} | {'Actual cP':>10} | {'Pred cP':>10} | {'% Err':>8}"
        )
        print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
        for _, row in results_df.iterrows():
            for shear in shear_cols:
                actual_col = f"Viscosity_{shear}"
                pred_col = f"Pred_Viscosity_{shear}"
                actual = row.get(actual_col, np.nan)
                pred = row.get(pred_col, np.nan)
                if pd.notna(actual) and pd.notna(pred) and actual > 0:
                    err = abs(pred - actual) / actual * 100
                    print(
                        f"  {str(row['ID']):>6} | {shear:>12} | {actual:>10.2f} | {pred:>10.2f} | {err:>7.1f}%"
                    )
                else:
                    print(
                        f"  {str(row['ID']):>6} | {shear:>12} | {'N/A':>10} | {pred:>10.2f} | {'N/A':>8}"
                    )

    # -----------------------------------------------------------------------
    # 4. CBM-specific demonstrations
    # -----------------------------------------------------------------------
    if is_cbm:
        print(f"\n{'='*60}")
        print("  CBM DEMONSTRATIONS")
        print(f"{'='*60}")

        # Pick adalimumab as demo group
        demo_protein = "Adalimumab"
        demo_target = target_df[target_df["Protein_type"] == demo_protein].copy()
        demo_history = full_train_df[
            (full_train_df["Protein_type"] == demo_protein)
            & (~full_train_df["ID"].isin(demo_target["ID"].tolist()))
        ].copy()

        predictor.memory_vector = None
        if not demo_history.empty:
            predictor.learn(demo_history)
        demo_row = demo_target.head(1)

        # --- 4a. Concept sweep: self_interaction ---
        print(f"\n  [Sweep] self_interaction vs log-viscosity @ 100 s⁻¹ (Adalimumab)")
        print(
            f"  {'c_self_interaction':>20} | {'Log_Visc (100/s)':>17} | {'Δ log_visc':>12}"
        )
        print(f"  {'-'*20}-+-{'-'*17}-+-{'-'*12}")

        sweep_df = predictor.concept_sweep(
            demo_row, "self_interaction", values=np.linspace(-1, 1, 11)
        )
        shear_100 = sweep_df[sweep_df["Shear_Rate"] == 100.0]
        for _, r in shear_100.iterrows():
            marker = (
                " <-- current"
                if abs(
                    r["Value"]
                    - float(predictor.get_concepts().iloc[0]["self_interaction"])
                )
                < 0.12
                else ""
            )
            print(
                f"  {r['Value']:>20.2f} | {r['Log_Viscosity']:>17.4f} | "
                f"{r['Delta_Log_Visc']:>+12.4f}{marker}"
            )
        sweep_df.to_csv("concept_sweep_self_interaction.csv", index=False)
        print("  Sweep saved to concept_sweep_self_interaction.csv")

        # --- 4b. Intervention: force high vs low hydrophobicity ---
        print(f"\n  [Intervention] hydrophobicity clamped to ±0.8 vs baseline")
        baseline_row = predictor.predict(demo_row)
        high_hci_row = predictor.intervene(demo_row, "hydrophobicity", +0.8)
        low_hci_row = predictor.intervene(demo_row, "hydrophobicity", -0.8)

        print(
            f"  {'Condition':<22} | {'100 s⁻¹':>10} | {'1000 s⁻¹':>10} | {'10000 s⁻¹':>10}"
        )
        print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        for label, res in [
            ("Baseline", baseline_row),
            ("hydrophobicity=+0.8", high_hci_row),
            ("hydrophobicity=-0.8", low_hci_row),
        ]:
            v100 = res.iloc[0].get("Pred_Viscosity_100", np.nan)
            v1000 = res.iloc[0].get("Pred_Viscosity_1000", np.nan)
            v10000 = res.iloc[0].get("Pred_Viscosity_10000", np.nan)
            print(f"  {label:<22} | {v100:>10.2f} | {v1000:>10.2f} | {v10000:>10.2f}")

        # --- 4c. Uncertainty ---
        print(f"\n  [Uncertainty] MC Dropout 95% CI (Adalimumab, 100 passes)")
        mean_pred, stats = predictor.predict_with_uncertainty(demo_row, n_samples=100)
        n_shears = len(predictor.shear_map)
        for j, shear in enumerate(shear_cols):
            lo = stats["lower_ci"][j]
            hi = stats["upper_ci"][j]
            std = stats["std_log10"][j]
            print(
                f"    {shear:>12} s⁻¹ : mean={mean_pred[j]:7.2f} cP | "
                f"95% CI [{lo:7.2f}, {hi:7.2f}] | σ={std:.3f} log₁₀"
            )

        # --- 4d. Compare two proteins' concept signatures ---
        print(f"\n  [Compare concepts] poly-hIgG vs Adalimumab")
        poly_ctx = full_train_df[full_train_df["Protein_type"] == "poly-hIgG"].copy()
        adal_ctx = full_train_df[full_train_df["Protein_type"] == "Adalimumab"].copy()
        if not poly_ctx.empty and not adal_ctx.empty:
            compare_df = predictor.compare_concepts(
                poly_ctx.head(10),
                adal_ctx.head(10),
                label_a="poly-hIgG",
                label_b="Adalimumab",
            )
            print(compare_df.to_string(index=False))
            compare_df.to_csv("concept_comparison.csv", index=False)
            print("  Comparison saved to concept_comparison.csv")

    # -----------------------------------------------------------------------
    # 5. Save all predictions
    # -----------------------------------------------------------------------
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv("cbm_predict_results.csv", index=False)
    print(f"\nAll predictions saved to cbm_predict_results.csv")
