import io
import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.pyplot import hist

# ==========================================
# 1. Physics Prior Configuration
# ==========================================
PRIOR_TABLE = {
    "igg1": {
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
    "igg4": {
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

# Split Thresholds (E_low vs E_high)
CONC_THRESHOLDS = {
    "arginine": 150.0,
    "lysine": 100.0,
    "proline": 200.0,
    "nacl": 150.0,
    "tween-20": 0.01,
    "tween-80": 0.01,
    "sucrose": 0.2,
    "trehalose": 0.2,
}


# ==========================================
# 2. Model Definition
# ==========================================
class CrossSampleCNP(nn.Module):
    def __init__(self, static_dim, hidden_dim=128, latent_dim=128, dropout=0.0):
        super().__init__()

        # --- ARCHITECTURE CHANGE ---
        # Encoder now takes ONLY 2 inputs: (Shear_Rate, Viscosity)
        # We removed 'static_dim' from here.
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # <--- Changed from (2 + static_dim)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder remains the same (Integrates Static + Context)
        self.decoder = nn.Sequential(
            nn.Linear(1 + static_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context_tensor, query_shear, query_static):
        # --- INPUT SLICING ---
        # context_tensor is [Batch, Points, (Shear, Visc, Static...)]
        # We slice it to keep ONLY the first 2 dimensions (Shear, Visc) for the encoder
        context_physics = context_tensor[:, :, :2]

        # 1. Encode Physics Only
        encoded = self.encoder(context_physics)
        r = torch.mean(encoded, dim=1)

        # 2. Decode using Physics Context + Static Identity
        n_queries = query_shear.size(1)
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)

        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)

    # Helper for inference (add this if missing)
    def encode_memory(self, context_tensor):
        context_physics = context_tensor[:, :, :2]
        encoded = self.encoder(context_physics)
        return torch.mean(encoded, dim=1)

    def decode_from_memory(self, memory_vector, query_shear, query_static):
        n_queries = query_shear.size(1)
        r_expanded = memory_vector.unsqueeze(1).repeat(1, n_queries, 1)
        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)


# ==========================================
# 3. The Predictor Class
# ==========================================
class ViscosityPredictorCNP:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir

        # Load Preprocessor
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {self.preprocessor_path}"
            )
        self.preprocessor = joblib.load(self.preprocessor_path)

        # Load Model
        self.model_path = os.path.join(model_dir, "best_model.pth")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.config = checkpoint["config"]
        self.static_dim = checkpoint["static_dim"]

        self.model = CrossSampleCNP(
            static_dim=self.static_dim,
            hidden_dim=self.config["hidden_dim"],
            latent_dim=self.config["latent_dim"],
            dropout=self.config["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.cached_memory = None

        self.shear_map = {
            "Viscosity_100": 100.0,
            "Viscosity_1000": 1000.0,
            "Viscosity_10000": 10000.0,
            "Viscosity_100000": 100000.0,
            "Viscosity_15000000": 1.5e7,
        }

        # Categorical columns that need normalization
        self.cat_cols = [
            "Protein_type",
            "Protein_class_type",
            "Buffer_type",
            "Salt_type",
            "Stabilizer_type",
            "Surfactant_type",
            "Excipient_type",
        ]

        # New Feature Columns defined by the update
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
        c_class = row.get("C_Class", 1.0)
        # Assuming missing defaults
        ph = row.get("Buffer_pH", 7.0)
        pi = row.get("PI_mean", 7.0)
        delta_ph = abs(ph - pi)
        tau = 1.5
        return c_class * np.exp(-delta_ph / tau)

    def _calculate_physics_features(self, row):
        """
        Computes separate Prior scores and split-concentration features.
        Returns a dictionary mapping {feature_name: value}.
        """
        # [cite_start]--- A. Determine Regime [cite: 9] ---
        cci = self._calculate_cci(row)
        p_type = str(row.get("Protein_class_type", "default")).lower()
        regime = "Far"

        if "igg1" in p_type:
            if cci >= 0.90:
                regime = "Near-pI"
            elif cci >= 0.50:
                regime = "Mixed"
        elif "igg4" in p_type:
            if cci >= 0.80:
                regime = "Near-pI"
            elif cci >= 0.40:
                regime = "Mixed"
        elif any(x in p_type for x in ["fc-fusion", "trispecific"]):
            if cci >= 0.70:
                regime = "Near-pI"
            elif cci >= 0.40:
                regime = "Mixed"
        elif any(x in p_type for x in ["bispecific", "adc"]):
            if cci >= 0.80:
                regime = "Near-pI"
            elif cci >= 0.45:
                regime = "Mixed"
        elif any(x in p_type for x in ["bsa", "polyclonal"]):
            if cci >= 0.70:
                regime = "Near-pI"
            elif cci >= 0.40:
                regime = "Mixed"
        else:
            if cci >= 0.70:
                regime = "Near-pI"
            elif cci >= 0.40:
                regime = "Mixed"

        # [cite_start]--- B. Get Prior Table [cite: 10, 27] ---
        lookup_key = "default"
        for key in PRIOR_TABLE.keys():
            if key != "default" and key in p_type:
                lookup_key = key
                break

        table = PRIOR_TABLE[lookup_key]
        regime_dict = table.get(regime, table["Far"])

        # [cite_start]--- C. Calculate Priors & Split Concentrations [cite: 12] ---
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
            ing_conc = float(row.get(conc_col, 0.0))

            if ing_name in ["none", "unknown", "nan"] or ing_conc <= 0:
                continue

            # 1. Map to Priors
            if "arginine" in ing_name or "arg" in ing_name:
                priors["prior_arginine"] = regime_dict.get("arginine", 0)
            elif "lysine" in ing_name or "lys" in ing_name:
                priors["prior_lysine"] = regime_dict.get("lysine", 0)
            elif "proline" in ing_name:
                priors["prior_proline"] = regime_dict.get("proline", 0)
            elif "nacl" in ing_name:
                priors["prior_nacl"] = regime_dict.get("nacl", 0)
            elif type_col == "Stabilizer_type":
                # General Stabilizer bucket
                priors["prior_stabilizer"] = regime_dict.get("stabilizer", 0)
            elif "tween" in ing_name or "polysorbate" in ing_name:
                t_key = "tween-20" if "20" in ing_name else "tween-80"
                priors[f"prior_{t_key}"] = regime_dict.get(t_key, 0)

            # 2. Map to Concentration Splits
            for target_ing, threshold in CONC_THRESHOLDS.items():
                match = False
                if target_ing in ing_name:
                    match = True
                elif target_ing == "arginine" and "arg" in ing_name:
                    match = True

                if match:
                    e_low = min(ing_conc, threshold)
                    e_high = max(ing_conc - threshold, 0)
                    concs[f"{target_ing}_low"] = e_low
                    concs[f"{target_ing}_high"] = e_high

        return {**priors, **concs}

    def _preprocess(
        self, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full preprocessing pipeline including Physics Prior calculation.
        """
        df_proc = df.copy()

        # 1. Normalize Categories to Lowercase
        for c in self.cat_cols:
            if c in df_proc.columns:
                df_proc[c] = df_proc[c].astype(str).str.lower()
            else:
                df_proc[c] = "unknown"

        # 2. Compute New Features (Row-wise)
        # Apply logic and expand dictionary to new columns
        new_features = df_proc.apply(
            self._calculate_physics_features, axis=1, result_type="expand"
        )
        df_proc = pd.concat([df_proc, new_features], axis=1)

        # 3. Run Scikit-Learn Pipeline
        # Ensure all columns expected by the preprocessor exist (fill 0.0 if not)
        feature_names = (
            self.preprocessor.feature_names_in_
            if hasattr(self.preprocessor, "feature_names_in_")
            else []
        )

        for col in feature_names:
            if col not in df_proc.columns:
                df_proc[col] = 0.0

        try:
            X_static = self.preprocessor.transform(df_proc)
        except ValueError as e:
            # Fallback if transformation fails due to schema mismatch
            print(f"Warning: Preprocessing schema mismatch: {e}")
            X_static = self.preprocessor.transform(df_proc)

        # 4. Flatten to Tensors
        points_list = []
        static_list = []

        for i in range(len(df_proc)):
            for col, shear_val in self.shear_map.items():
                val = 1.0
                if col in df_proc.columns and pd.notna(df_proc.iloc[i][col]):
                    val = df_proc.iloc[i][col]

                # Physical constraint check
                if val <= 0:
                    val = 1e-6

                points_list.append([np.log10(shear_val), np.log10(val)])
                static_list.append(X_static[i])

        static_t = (
            torch.tensor(np.array(static_list), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        points_t = (
            torch.tensor(np.array(points_list), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        shear_t = points_t[:, :, [0]]
        visc_t = points_t[:, :, [1]]
        return static_t, shear_t, visc_t

    def learn(
        self,
        df: pd.DataFrame,
        fine_tune: bool = True,
        steps: int = 50,
        lr: float = 1e-3,
    ):
        """
        Adapts the model to new data.
        fine_tune=True (default) updates weights to capture specific effects (e.g. Sucrose sensitivity).
        """
        static_t, shear_t, visc_t = self._preprocess(df)
        context_t = torch.cat([shear_t, visc_t, static_t], dim=-1)

        if fine_tune:
            self.model.train()
            # Use a slightly higher LR for fine-tuning to overcome "average" priors
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            for _ in range(steps):
                pred = self.model(context_t, shear_t, static_t)
                loss = F.mse_loss(pred, visc_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.model.eval()

        with torch.no_grad():
            self.cached_memory = self.model.encode_memory(context_t)

    def predict(
        self, df: pd.DataFrame, context_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Predicts viscosity using In-Context Learning.

        CRITICAL: This assumes all rows in 'df' belong to the SAME system/protein
        as the 'context_df' (or self.cached_memory).
        """
        # 1. Resolve Memory (The Context)
        memory_vector = self.cached_memory

        # Allow on-the-fly context override (Stateless prediction)
        if context_df is not None:
            c_static, c_shear, c_visc = self._preprocess(context_df)
            c_tensor = torch.cat([c_shear, c_visc, c_static], dim=-1)
            with torch.no_grad():
                memory_vector = self.model.encode_memory(c_tensor)

        # Handle Zero-Shot (No context available)
        if memory_vector is None:
            # Warn if this wasn't intended
            # print("Warning: Predicting in Zero-Shot mode (No Context).")
            memory_vector = torch.zeros((1, self.config["latent_dim"])).to(self.device)

        # 2. Prepare Query (The Target Conditions)
        # Ensure _preprocess uses the SAME scalers as training!
        q_static, q_shear, _ = self._preprocess(df)

        # 3. Decode (The Prediction)
        self.model.eval()
        with torch.no_grad():
            # NOTE: If the model was trained with "Static Dropout", the Decoder
            # is now robust enough to combine this 'q_static' with 'memory_vector'
            # effectively.
            pred_log = self.model.decode_from_memory(memory_vector, q_shear, q_static)

        # 4. Format Results
        pred_visc = torch.pow(10, pred_log).cpu().numpy().flatten()

        results = df.copy()
        n_shears = len(self.shear_map)
        shear_keys = list(self.shear_map.keys())
        new_cols = {k: [] for k in shear_keys}

        for i in range(len(df)):
            start = i * n_shears
            sample_preds = pred_visc[start : start + n_shears]
            for j, key in enumerate(shear_keys):
                new_cols[key].append(sample_preds[j])

        for k, v in new_cols.items():
            results[f"Pred_{k}"] = v
        return results

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        n_samples: int = 20,
        ci_range: Tuple[float, float] = (2.5, 97.5),
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Runs Monte Carlo Dropout and returns format compatible with VisQAI.
        """
        self.model.train()  # Enable Dropout

        memory_vector = self.cached_memory
        if memory_vector is None:
            memory_vector = torch.zeros((1, self.config["latent_dim"])).to(self.device)

        q_static, q_shear, _ = self._preprocess(df)

        preds_log = []
        with torch.no_grad():
            for _ in range(n_samples):
                out_log = self.model.decode_from_memory(
                    memory_vector, q_shear, q_static
                )
                preds_log.append(out_log.cpu().numpy())

        self.model.eval()

        stack_log = np.stack(preds_log)
        stack_linear = np.power(10, stack_log).squeeze()
        if stack_linear.ndim == 1:
            stack_linear = stack_linear[:, None]

        mean_pred = np.mean(stack_linear, axis=0)
        std_pred = np.std(stack_linear, axis=0)

        lower_ci = np.percentile(stack_linear, ci_range[0], axis=0)
        upper_ci = np.percentile(stack_linear, ci_range[1], axis=0)

        stats = {"std": std_pred, "lower_ci": lower_ci, "upper_ci": upper_ci}

        return mean_pred, stats

    def save(self, output_dir: str = None):
        if output_dir is None:
            output_dir = self.model_dir
        save_path = os.path.join(output_dir, "best_model.pth")
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "config": self.config,
            "static_dim": self.static_dim,
        }
        torch.save(checkpoint, save_path)


if __name__ == "__main__":
    import io
    import os

    import pandas as pd

    # Configuration
    model_dir = r"models/experiments/o_net"
    training_file = "data/processed/formulation_data_augmented.csv"

    print(f"Loading training data from {training_file}...")
    if not os.path.exists(training_file):
        raise FileNotFoundError(f"Could not find training file: {training_file}")
    hisorical_type = "Pembrolizumab"
    full_train_df = pd.read_csv(training_file)
    history_df = full_train_df[full_train_df["Protein_type"] == hisorical_type].copy()
    print(f"Found {len(history_df)} historical '{hisorical_type}' samples.")
    target_data = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
F426,Random,mAb_IgG4,3.5,149.0,7.57,0.3,135.0,25.0,Histidine,6.0,15,none,0,none,0.0,none,0.0,Arginine,100,1.3,1.1,21.6,16.1,12.7,9.4,2.2"""

    target_df = pd.read_csv(io.StringIO(target_data))

    predictor = ViscosityPredictorCNP(model_dir)
    if len(history_df) > 0:
        print("\nAdapting model to class behavior...")
        predictor.learn(history_df, fine_tune=True, steps=50, lr=1e-3)

    print("\nPredicting Sample based on adapted knowledge...")
    out = predictor.predict(target_df)

    cols_actual = ["Viscosity_100", "Viscosity_1000", "Viscosity_15000000"]
    cols_pred = ["Pred_Viscosity_100", "Pred_Viscosity_1000", "Pred_Viscosity_15000000"]

    print("\n--- RESULTS ---")
    print("Actual (Ground Truth):")
    print(out[cols_actual].to_string(index=False))

    print("\nPredicted (After Class Fine-Tuning):")
    print(out[cols_pred].to_string(index=False))
