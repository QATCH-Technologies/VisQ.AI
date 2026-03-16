import copy
import datetime
import io
import logging
import os
import sys
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. Logging Configuration
# ==========================================
# Create a timestamped log file
log_filename = (
    f"debug_inference_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),  # Also print to console
    ],
)
logger = logging.getLogger("VisQ_Inference")


# ==========================================
# 1. Model Architecture (Embedded for Standalone Use)
# ==========================================


class AttentionPool(nn.Module):
    def __init__(self, latent_dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        # [ADDED] LayerNorm to match the v3 architecture update
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        # [ADDED] Apply LayerNorm before returning
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
# 2. Configuration & Physics Priors
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
# 3. The Predictor Class
# ==========================================
class ViscosityPredictorCNP:
    def __init__(self, model_dir: str, verbose: bool = False):
        # Create a per-instance logger that can be silenced independently
        # of the module-level logger used elsewhere in this file.
        self._logger = logging.getLogger(f"VisQ_Inference.{id(self)}")
        if not verbose:
            self._logger.setLevel(
                logging.CRITICAL
            )  # suppress everything below CRITICAL
        self._logger.info(
            f"Initializing ViscosityPredictorCNP with model_dir: {model_dir}"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._logger.info(f"Using device: {self.device}")
        self.model_dir = model_dir
        self.memory_vector = None  # Stores the calibrated context

        # 1. Load Preprocessors
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        self.scaler_path = os.path.join(model_dir, "physics_scaler.pkl")

        if not os.path.exists(self.preprocessor_path):
            self._logger.error(f"Preprocessor not found at {self.preprocessor_path}")
            raise FileNotFoundError(
                f"Preprocessor not found at {self.preprocessor_path}"
            )
        if not os.path.exists(self.scaler_path):
            self._logger.error(f"Physics Scaler not found at {self.scaler_path}")
            raise FileNotFoundError(f"Physics Scaler not found at {self.scaler_path}")

        self._logger.debug("Loading preprocessor and scaler...")
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.physics_scaler = joblib.load(self.scaler_path)

        # 2. Load Model
        self.model_path = os.path.join(model_dir, "best_model.pth")
        if not os.path.exists(self.model_path):
            self._logger.error(f"Model checkpoint not found at {self.model_path}")
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        self._logger.debug(f"Loading model checkpoint from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.config = checkpoint["config"]
        self.static_dim = checkpoint["static_dim"]
        self._logger.debug(f"Model config: {self.config}")
        self._logger.debug(f"Static dimension: {self.static_dim}")

        # Initialize the standalone model class
        self.model = CrossSampleCNP(
            static_dim=self.static_dim,
            hidden_dim=self.config["hidden_dim"],
            latent_dim=self.config["latent_dim"],
            dropout=self.config["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        # [FIX-2] Save a pristine copy of the model weights so we can restore
        # them before each learn() call, preventing cross-protein contamination.
        self._original_state = copy.deepcopy(self.model.state_dict())

        # Stores the raw context tensor from the last learn() call,
        # used by predict_with_uncertainty for context-subsampling CI.
        self.context_t: Optional[torch.Tensor] = None

        # Shear Mapping (Log-Log scaling logic)
        self.shear_map = {
            "Viscosity_100": 100.0,
            "Viscosity_1000": 1000.0,
            "Viscosity_10000": 10000.0,
            "Viscosity_100000": 100000.0,
            "Viscosity_15000000": 1.5e7,
        }

        # Categorical columns
        self.cat_cols = [
            "Protein_type",
            "Protein_class_type",
            "Buffer_type",
            "Salt_type",
            "Stabilizer_type",
            "Surfactant_type",
            "Excipient_type",
        ]

        # Physics Features
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
        for k in CONC_THRESHOLDS.keys():
            self.new_conc_cols.append(f"{k}_low")
            self.new_conc_cols.append(f"{k}_high")

    # ------------------------------------------------------------------
    # Physics Helpers (Internal)
    # ------------------------------------------------------------------
    def _calculate_cci(self, row):
        try:
            c_class = float(row.get("C_Class", 1.0))
            ph = float(row.get("Buffer_pH", 7.0))
            pi = float(row.get("PI_mean", 7.0))
        except ValueError as e:
            self._logger.warning(
                f"Error converting CCI inputs to float: {e}. Row: {row.to_dict()}"
            )
            c_class, ph, pi = 1.0, 7.0, 7.0

        if pd.isna(ph):
            ph = 7.0
        if pd.isna(pi):
            pi = 7.0

        delta_ph = abs(ph - pi)
        tau = 1.5
        return c_class * np.exp(-delta_ph / tau)

    def _calculate_physics_features(self, row):
        """
        Computes separate Prior scores and split-concentration features.
        Matches train preprocessing logic exactly.
        """
        cci = self._calculate_cci(row)
        p_type = str(row.get("Protein_class_type", "default")).lower()

        # Determine Regime
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

        # Get Prior Table
        lookup_key = "default"
        for key in PRIOR_TABLE.keys():
            if key != "default" and key in p_type:
                lookup_key = key
                break
        table = PRIOR_TABLE[lookup_key]
        regime_dict = table.get(regime, table["Far"])

        priors = {k: 0.0 for k in self.new_prior_cols}
        concs = {k: 0.0 for k in self.new_conc_cols}

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
            except Exception:
                ing_conc = 0.0

            if ing_name in ["none", "unknown", "nan"] or ing_conc <= 0:
                continue

            # Map to Priors
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

            # Map to Concentration Splits
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
        self._logger.debug(f"--- PREPROCESSING START ---")
        self._logger.debug(f"Input DataFrame Shape: {df.shape}")

        df_proc = df.copy()

        # 0. Extract objects to values if needed
        for col in df_proc.select_dtypes(include=["object"]):
            df_proc[col] = df_proc[col].apply(
                lambda x: x.value if hasattr(x, "value") else x
            )

        if "ID" in df_proc.columns:
            df_proc.drop(columns=["ID"], inplace=True)

        # 1. Fill defaults for numeric columns (Matches Train)
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
            if c not in df_proc.columns:
                df_proc[c] = 0.0
            else:
                df_proc[c] = df_proc[c].fillna(0.0)

        # ---------------------------------------------------------
        # 2. Add Missing Feature Engineering & Unit Normalization
        # ---------------------------------------------------------
        self._logger.debug(
            "Normalizing units to mg/mL and calculating Physics Features..."
        )

        MW_MAP = {
            "sucrose": 342.3,
            "trehalose": 342.3,
            "arginine": 174.2,
            "proline": 115.1,
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

        # Unit conversions to mg/mL
        if "Stabilizer_type" in df_proc.columns:
            stabilizer_mw = get_mw(df_proc["Stabilizer_type"], default_mw=342.3)
        else:
            stabilizer_mw = 342.3
        df_proc["Stabilizer_mg_mL"] = df_proc["Stabilizer_conc"] * stabilizer_mw

        if "Salt_type" in df_proc.columns:
            salt_mw = get_mw(df_proc["Salt_type"], default_mw=58.44)
        else:
            salt_mw = 58.44
        df_proc["Salt_mg_mL"] = (df_proc["Salt_conc"] * salt_mw) / 1000.0

        if "Excipient_type" in df_proc.columns:
            excipient_mw = get_mw(df_proc["Excipient_type"], default_mw=150.0)
        else:
            excipient_mw = 150.0
        df_proc["Excipient_mg_mL"] = (df_proc["Excipient_conc"] * excipient_mw) / 1000.0

        df_proc["Surfactant_mg_mL"] = df_proc["Surfactant_conc"] * 10.0

        # Feature calculation
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

        # 3. Normalize Categories
        for c in self.cat_cols:
            if c in df_proc.columns:
                df_proc[c] = df_proc[c].astype(str).str.lower()
                df_proc[c] = df_proc[c].replace("nan", "unknown")
            else:
                df_proc[c] = "unknown"

        # 4. Compute New Prior Features
        self._logger.debug("Computing physics priors...")
        new_features = df_proc.apply(
            self._calculate_physics_features, axis=1, result_type="expand"
        )
        df_proc = pd.concat([df_proc, new_features], axis=1)

        # 5. Handle missing statically expected features
        feature_names = (
            self.preprocessor.feature_names_in_
            if hasattr(self.preprocessor, "feature_names_in_")
            else []
        )

        # Target columns and ID are expected to be missing during inference
        expected_missing = ["ID"] + list(self.shear_map.keys())

        missing_feats = [col for col in feature_names if col not in df_proc.columns]
        actual_missing = [col for col in missing_feats if col not in expected_missing]

        if actual_missing:
            self._logger.warning(
                f"Missing static features filled with 0.0: {actual_missing}"
            )
        for col in missing_feats:
            df_proc[col] = 0.0

        X_static = self.preprocessor.transform(df_proc)

        # Ensure no NaNs leaked into the matrix
        if np.isnan(X_static).any():
            self._logger.warning(
                "NaNs found in X_static after preprocessing! Replacing with 0."
            )
            X_static = np.nan_to_num(X_static)

        self._logger.debug(f"Static features transformed shape: {X_static.shape}")

        # 6. Physics Flattening & Scaling (Log-Log) — batched for speed
        n_rows = len(df_proc)
        n_shears = len(self.shear_map)

        # Build all (log_shear, log_visc) pairs in a single numpy array,
        # then call physics_scaler.transform once instead of n_rows*n_shears times.
        raw_points = np.empty((n_rows * n_shears, 2), dtype=np.float64)
        static_list = []

        row_idx = 0
        for i in range(n_rows):
            for col, shear_val in self.shear_map.items():
                val = 1.0
                if col in df_proc.columns and pd.notna(df_proc.iloc[i][col]):
                    val = float(df_proc.iloc[i][col])
                if val <= 0:
                    val = 1e-6

                raw_points[row_idx, 0] = np.log10(shear_val)
                raw_points[row_idx, 1] = np.log10(val)
                static_list.append(X_static[i])
                row_idx += 1

        # [FIX-6] Single batched transform — avoids n*m repeated sklearn calls
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

        self._logger.debug(
            f"Final Tensor Shapes -> Static: {static_t.shape}, Shear: {shear_t.shape}, Visc: {visc_t.shape}"
        )
        return static_t, shear_t, visc_t

    def learn(
        self,
        df: pd.DataFrame,
        steps: int = 50,  # kept for API compatibility — no longer used
        lr: float = 1e-3,  # kept for API compatibility — no longer used
        n_draws: int = 20,
        k: int = 8,
    ):
        """
        Adapts the predictor to a new protein group by encoding its context
        samples into a stable latent memory vector.

        [FIX-1] No weight updates are performed. The CNP is designed to adapt
        entirely through the latent vector r — fine-tuning weights on 20-30
        samples corrupts the encoder's learned representations and causes
        cross-protein contamination.

        [FIX-2] Model weights are restored from the pristine checkpoint before
        encoding, so successive calls for different proteins are fully isolated.

        [FIX-3] Multi-draw averaging: n_draws random subsets of size k are
        encoded and their latent vectors are averaged. This smooths out the
        stochastic context variance (intra-group spread) and produces a more
        stable memory vector than a single full-context encoding.

        Args:
            df:      DataFrame of context samples for the target protein.
            steps:   Ignored. Retained for API compatibility.
            lr:      Ignored. Retained for API compatibility.
            n_draws: Number of random context subsets to encode and average.
            k:       Size of each random subset (matches the few-shot elbow k=8).
        """
        if df.empty:
            self._logger.warning("Context DataFrame is empty. Skipping learning.")
            print("Warning: Context DataFrame is empty. Skipping learning.")
            return

        self._logger.info(
            f" > Learn triggered on {len(df)} samples "
            f"(n_draws={n_draws}, k={k}, no weight updates)."
        )
        print(f" > Encoding context: {len(df)} samples, {n_draws} draws of k={k}...")

        # [FIX-2] Restore pristine weights before every encode so successive
        # calls for different proteins never contaminate each other.
        self.model.load_state_dict(self._original_state)

        # 1. Preprocess all context samples into tensors
        static_t, shear_t, visc_t = self._preprocess(df)
        context_t = torch.cat([shear_t, visc_t, static_t], dim=-1)

        # 2. Store full context tensor for uncertainty estimation
        self.context_t = context_t

        n_ctx = context_t.size(1)
        k_eff = min(k, n_ctx)  # can't draw more than we have

        self.model.eval()
        memory_draws = []
        with torch.no_grad():
            if n_ctx <= k_eff:
                # Not enough context for subsampling — encode everything once
                self._logger.debug(
                    f"n_ctx={n_ctx} <= k={k_eff}: encoding full context once."
                )
                self.memory_vector = self.model.encode_memory(context_t)
                self._logger.info(
                    f" > Encoding complete (single pass). "
                    f"Memory shape: {self.memory_vector.shape}"
                )
                return

            # [FIX-3] Multi-draw averaging
            for draw_i in range(n_draws):
                idx = torch.randperm(n_ctx, device=self.device)[:k_eff]
                subset = context_t[:, idx, :]
                r = self.model.encode_memory(subset)
                memory_draws.append(r)
                self._logger.debug(
                    f"  Draw {draw_i+1}/{n_draws}: "
                    f"idx={idx.tolist()}, norm={r.norm().item():.3f}"
                )

        # Average across draws → stable latent representation
        self.memory_vector = torch.stack(memory_draws, dim=0).mean(dim=0)
        self._logger.info(
            f" > Encoding complete ({n_draws} draws averaged). "
            f"Memory norm: {self.memory_vector.norm().item():.3f}, "
            f"shape: {self.memory_vector.shape}"
        )
        print(
            f" > Encoding complete. "
            f"Memory norm: {self.memory_vector.norm().item():.3f}"
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts using the cached memory (calibrated state).
        """
        self._logger.info(f"Predict triggered on {len(df)} samples.")

        # 1. Context Resolution
        memory_vector = self.memory_vector

        # Handle Zero-Shot (if learn was never called)
        if memory_vector is None:
            self._logger.warning(
                "Memory vector is None. Performing Zero-Shot prediction (using zero tensor)."
            )
            memory_vector = torch.zeros((1, self.config["latent_dim"])).to(self.device)

        # 2. Prepare Query
        q_static, q_shear, _ = self._preprocess(df)

        # 3. Decode from Memory (using cached vector)
        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model.decode_from_memory(memory_vector, q_shear, q_static)

        # 4. Inverse Scaling — batched via _inverse_to_log helper
        pred_log10 = self._inverse_to_log(q_shear, y_scaled)
        pred_visc_cp = np.power(10, pred_log10)

        # 5. Format Results
        results = df.copy()
        n_shears = len(self.shear_map)
        shear_keys = list(self.shear_map.keys())
        new_cols = {k: [] for k in shear_keys}

        for i in range(len(df)):
            start = i * n_shears
            sample_preds = pred_visc_cp[start : start + n_shears]
            for j, key in enumerate(shear_keys):
                new_cols[key].append(sample_preds[j])

        for k, v in new_cols.items():
            results[f"Pred_{k}"] = v

        self._logger.info("Prediction complete.")
        return results

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        n_samples: int = 100,
        ci_range: Tuple[float, float] = (2.5, 97.5),
        k: Optional[int] = None,  # retained for API compatibility — no longer used
    ):
        """
        Estimates the model's predictive uncertainty via MC Dropout.

        The memory vector is fixed (already encoded from context), and the
        decoder is run n_samples times in train() mode so that each forward
        pass samples a different dropout mask. The spread of those predictions
        reflects the model's own uncertainty about the query — not variance in
        the training history.

        CIs are computed in log10 space (where the model was trained) and
        exponentiated for the final output. This avoids distortion from the
        heavy right tail of viscosity values.

        Requires dropout > 0.0 in the model config. If dropout == 0.0,
        all passes are deterministic and the CI will be zero-width.

        Args:
            df:        DataFrame of query samples to predict.
            n_samples: Number of stochastic forward passes (100 recommended).
            ci_range:  Percentile bounds for the confidence interval.
            k:         Deprecated. Accepted for backwards compatibility but ignored.
                       Previously controlled context-subsampling subset size.

        Returns:
            mean_pred:  Mean prediction in linear cP, shape (n_queries,).
            stats:      Dict with keys 'mean_log10', 'std_log10', 'lower_ci',
                        'upper_ci'. std_log10 is in log10 units (0.1 ≈ ±26%
                        factor); lower/upper_ci are in linear cP.
        """
        dropout_val = self.config.get("dropout", 0.0)
        self._logger.info(
            f"MC Dropout uncertainty: n_samples={n_samples}, "
            f"ci_range={ci_range}, dropout={dropout_val}"
        )
        if dropout_val == 0.0:
            self._logger.warning(
                "Model config has dropout=0.0. MC Dropout will produce a "
                "zero-width CI because all forward passes are deterministic. "
                "Retrain with dropout > 0 (e.g. 0.1) for meaningful uncertainty."
            )
            print(
                "WARNING: dropout=0.0 in checkpoint — CI will be zero-width. "
                "Retrain with dropout > 0.0 for real uncertainty estimates."
            )

        # 1. Prepare query tensors
        q_static, q_shear, _ = self._preprocess(df)

        # 2. Fix the memory vector — uncertainty comes from the decoder, not context
        memory_fixed = (
            self.memory_vector
            if self.memory_vector is not None
            else torch.zeros((1, self.config["latent_dim"]), device=self.device)
        )
        if self.memory_vector is None:
            self._logger.warning(
                "No memory vector (zero-shot). CI reflects decoder noise only."
            )

        # 3. Run n_samples stochastic decoder passes with dropout active
        self.model.train()  # activates dropout masks
        preds_log = []
        with torch.no_grad():
            for i in range(n_samples):
                out_scaled = self.model.decode_from_memory(
                    memory_fixed, q_shear, q_static
                )
                log_vals = self._inverse_to_log(q_shear, out_scaled)
                preds_log.append(log_vals)
                if (i + 1) % 25 == 0:
                    self._logger.debug(f"  MC Dropout pass {i+1}/{n_samples}")
        self.model.eval()

        # [FIX-4b] All statistics computed in log10 space
        stack_log = np.stack(preds_log)  # (n_samples, n_queries)
        mean_log = np.mean(stack_log, axis=0)
        std_log = np.std(stack_log, axis=0)
        lower_log = np.percentile(stack_log, ci_range[0], axis=0)
        upper_log = np.percentile(stack_log, ci_range[1], axis=0)

        mean_pred = np.power(10, mean_log)
        lower_ci = np.power(10, lower_log)
        upper_ci = np.power(10, upper_log)

        stats = {
            "mean_log10": mean_log,  # log10 units — use for model diagnostics
            "std_log10": std_log,  # log10 units — 0.1 ≈ ±26% factor error
            "lower_ci": lower_ci,  # linear cP
            "upper_ci": upper_ci,  # linear cP
        }

        self._logger.info(
            f"Uncertainty complete. Mean log10 RMSE across queries: "
            f"{std_log.mean():.4f} log10 units."
        )
        return mean_pred, stats

    # ------------------------------------------------------------------
    # Internal helper: inverse-transform a scaled decoder output to log10 visc
    # ------------------------------------------------------------------
    def _inverse_to_log(
        self, q_shear: torch.Tensor, out_scaled: torch.Tensor
    ) -> np.ndarray:
        """Inverse-scales a decoder output tensor to log10 viscosity values."""
        q_shear_np = q_shear.cpu().numpy().reshape(-1, 1)
        out_np = out_scaled.cpu().numpy().reshape(-1, 1)
        combined = np.hstack([q_shear_np, out_np])
        log_vals = self.physics_scaler.inverse_transform(combined)[:, 1]
        return log_vals


if __name__ == "__main__":
    # Test Configuration
    model_dir = "models/experiments/o_net_v3"
    training_file = "data/raw/formulation_data_03022026.csv"

    # 1. Initialize Predictor
    try:
        predictor = ViscosityPredictorCNP(model_dir)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit()

    # 2. Load Full Training Data (for context)
    if not os.path.exists(training_file):
        print(f"Error: Training file not found at {training_file}")
        exit()

    print(f"Loading context pool from {training_file}...")
    full_train_df = pd.read_csv(training_file)
    int_cols = full_train_df.select_dtypes(include=["int", "int64", "int32"]).columns

    for col in int_cols:
        if col != "ID":
            full_train_df[col] = full_train_df[col].astype(float)
    full_train_df["ID"] = full_train_df["ID"].astype(str)

    # 3. Define the 4 Target Samples
    target_data = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
F425,Pembrolizumab,mAb_IgG4,3.5,149,7.57,0.3,150,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.3,1.1,25.63,21,18.38,14.61,8.3
F427,Pembrolizumab,mAb_IgG4,3.5,149,7.57,0.3,214,25,Histidine,6,15,none,0,none,0,none,0,none,0,1.3,1.1,43.75,43.75,43.75,40.25,14
F446,Adalimumab,mAb_IgG1,3,148,8.7,0.3,135,25,Histidine,6,15,none,0,none,0,none,0,none,0,1,1,10.31,8.75,7.88,7.25,1.35
F448,Adalimumab,mAb_IgG1,3,148,8.7,0.3,135,25,Histidine,6,15,none,0,Sucrose,0.4,none,0,none,0,1,1,8.38,8.38,8.38,7.71,1.4"""

    target_df = pd.read_csv(io.StringIO(target_data))
    target_df["ID"] = target_df["ID"].astype(str)

    # 4. Process predictions by Protein Type
    all_results = []

    for protein in target_df["Protein_type"].unique():
        print(f"\n--- Processing Target Group: {protein} ---")
        prot_target_df = target_df[target_df["Protein_type"] == protein].copy()
        target_ids = prot_target_df["ID"].tolist()

        # Build history strictly EXCLUDING the target samples to avoid data leakage
        history_df = full_train_df[
            (full_train_df["Protein_type"] == protein)
            & (~full_train_df["ID"].isin(target_ids))
        ].copy()

        if not history_df.empty:
            print(f"Adapting to {protein} ({len(history_df)} context samples)...")
            # [FIX-2] Reset both memory state and stored context before each group
            predictor.memory_vector = None
            predictor.context_t = None
            predictor.learn(history_df)
        else:
            print(
                f"Warning: No history found for {protein}. Falling back to Zero-Shot."
            )
            predictor.memory_vector = None
            predictor.context_t = None

        # Predict for this protein group
        print(f"Predicting {len(prot_target_df)} target sample(s)...")
        results_df = predictor.predict(prot_target_df)
        all_results.append(results_df)

    # 5. Compile and Measure Pred vs Actual
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv("debug_predict_results.csv", index=False)

    print("\n" + "=" * 60)
    print("PREDICTED vs ACTUAL VISCOSITY (cP)")
    print("=" * 60)

    shear_cols = ["100", "1000", "10000", "100000", "15000000"]

    for _, row in final_results.iterrows():
        print(f"\nSample ID: {row['ID']} | Protein: {row['Protein_type']}")
        print(
            f"{'Shear Rate':>12} | {'Actual cP':>10} | {'Pred cP':>10} | {'% Error':>10}"
        )
        print("-" * 52)

        for shear in shear_cols:
            actual_col = f"Viscosity_{shear}"
            pred_col = f"Pred_Viscosity_{shear}"

            # Safely grab values, handling cases where they might be missing/NaN
            actual_val = row.get(actual_col, np.nan)
            pred_val = row.get(pred_col, np.nan)

            if pd.notna(actual_val) and pd.notna(pred_val) and actual_val > 0:
                error = abs(pred_val - actual_val) / actual_val * 100
                print(
                    f"{shear:>12} | {actual_val:10.2f} | {pred_val:10.2f} | {error:9.1f}%"
                )
            else:
                print(f"{shear:>12} | {'N/A':>10} | {pred_val:10.2f} | {'N/A':>10}")

    # 6. Uncertainty Estimates (per protein group)
    print("\n" + "=" * 60)
    print("UNCERTAINTY ESTIMATES (MC Dropout, 95% CI)")
    print("=" * 60)

    for protein in target_df["Protein_type"].unique():
        prot_target_df = target_df[target_df["Protein_type"] == protein].copy()
        target_ids = prot_target_df["ID"].tolist()

        # Re-build context (same as above, excluding targets)
        history_df = full_train_df[
            (full_train_df["Protein_type"] == protein)
            & (~full_train_df["ID"].isin(target_ids))
        ].copy()

        predictor.memory_vector = None
        predictor.context_t = None

        if not history_df.empty:
            predictor.learn(history_df)
            mean_pred, stats = predictor.predict_with_uncertainty(
                prot_target_df, n_samples=100
            )

            pred_ids = prot_target_df["ID"].tolist()
            n_shears = len(predictor.shear_map)

            print(f"\n  {protein}")
            for i, sid in enumerate(pred_ids):
                print(f"    Sample {sid}:")
                for j, shear in enumerate(shear_cols):
                    q_idx = i * n_shears + j
                    if q_idx < len(mean_pred):
                        std_log = stats["std_log10"][q_idx]
                        lo = stats["lower_ci"][q_idx]
                        hi = stats["upper_ci"][q_idx]
                        print(
                            f"      {shear:>12} s⁻¹ | "
                            f"mean={mean_pred[q_idx]:7.2f} cP | "
                            f"95% CI [{lo:7.2f}, {hi:7.2f}] | "
                            f"sigma={std_log:.3f} log₁₀"
                        )
        else:
            print(f"\n  {protein}: No context — zero-shot, uncertainty not available.")
