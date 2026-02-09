import io
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. Model Architecture (Embedded for Standalone Use)
# ==========================================
class CrossSampleCNP(nn.Module):
    """
    Conditional Neural Process for Viscosity Prediction.
    Matches the architecture defined in train_o_net.py.
    """

    def __init__(self, static_dim, hidden_dim=128, latent_dim=128, dropout=0.0):
        super().__init__()

        # Encoder: Takes [Shear, Viscosity, Static_Features] -> Latent Representation
        self.encoder = nn.Sequential(
            nn.Linear(2 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder: Takes [Query_Shear, Query_Static, Latent_Rep] -> Predicted Viscosity
        self.decoder = nn.Sequential(
            nn.Linear(1 + static_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context_tensor, query_shear, query_static):
        # 1. Encode Context
        encoded = self.encoder(context_tensor)
        r = torch.mean(encoded, dim=1)  # Aggregate context into single vector

        # 2. Decode
        n_queries = query_shear.size(1)
        # Expand latent vector r to match the number of query points
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)

        # Concatenate: [Shear, Static, Latent]
        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)

    def encode_memory(self, context_tensor):
        """
        Encodes the context into a single latent vector (memory).
        Used during the 'learn' phase to cache knowledge.
        """
        encoded = self.encoder(context_tensor)
        return torch.mean(encoded, dim=1)

    def decode_from_memory(self, memory_vector, query_shear, query_static):
        """
        Decodes targets using a pre-computed latent vector.
        Used during the 'predict' phase for fast inference.
        """
        n_queries = query_shear.size(1)
        # Handle batch dimension if necessary, though usually memory_vector is [1, latent]
        if memory_vector.dim() == 2 and memory_vector.size(0) == 1:
            r_expanded = memory_vector.unsqueeze(1).repeat(1, n_queries, 1)
        else:
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
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.memory_vector = None  # Stores the calibrated context

        # 1. Load Preprocessors
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        self.scaler_path = os.path.join(model_dir, "physics_scaler.pkl")

        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {self.preprocessor_path}"
            )
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Physics Scaler not found at {self.scaler_path}")

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.physics_scaler = joblib.load(self.scaler_path)

        # 2. Load Model
        self.model_path = os.path.join(model_dir, "best_model.pth")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.config = checkpoint["config"]
        self.static_dim = checkpoint["static_dim"]

        # Initialize the standalone model class
        self.model = CrossSampleCNP(
            static_dim=self.static_dim,
            hidden_dim=self.config["hidden_dim"],
            latent_dim=self.config["latent_dim"],
            dropout=self.config["dropout"],
        ).to(self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

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
        c_class = row.get("C_Class", 1.0)
        ph = row.get("Buffer_pH", 7.0)
        pi = row.get("PI_mean", 7.0)
        # Handle potential NaNs
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
        Matches train_o_net.py logic exactly.
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
            ing_conc = float(row.get(conc_col, 0.0))

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
        df_proc = df.copy()

        # 1. Normalize Categories
        for c in self.cat_cols:
            if c in df_proc.columns:
                df_proc[c] = df_proc[c].astype(str).str.lower()
                df_proc[c] = df_proc[c].replace("nan", "unknown")
            else:
                df_proc[c] = "unknown"

        # 2. Compute New Features
        new_features = df_proc.apply(
            self._calculate_physics_features, axis=1, result_type="expand"
        )
        df_proc = pd.concat([df_proc, new_features], axis=1)

        # 3. Static Features Transformation
        feature_names = (
            self.preprocessor.feature_names_in_
            if hasattr(self.preprocessor, "feature_names_in_")
            else []
        )
        for col in feature_names:
            if col not in df_proc.columns:
                df_proc[col] = 0.0

        X_static = self.preprocessor.transform(df_proc)

        # 4. Physics Flattening & Scaling (Log-Log)
        points_list = []
        static_list = []

        for i in range(len(df_proc)):
            for col, shear_val in self.shear_map.items():
                val = 1.0
                # Use value if present (for context), else placeholder (for query)
                if col in df_proc.columns and pd.notna(df_proc.iloc[i][col]):
                    val = df_proc.iloc[i][col]

                if val <= 0:
                    val = 1e-6

                # Transform using the scaler: [log_shear, log_visc] -> [scaled_shear, scaled_visc]
                raw_point = np.array([[np.log10(shear_val), np.log10(val)]])
                scaled_point = self.physics_scaler.transform(raw_point)[0]

                points_list.append(scaled_point)
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
        steps: int = 50,
        lr: float = 1e-3,
    ):
        """
        Fine-tunes the model on the provided samples (Calibrator Logic).
        """
        if df.empty:
            print("Warning: Context DataFrame is empty. Skipping learning.")
            return

        print(f" > Calibrating on {len(df)} samples for {steps} steps...")

        # 1. Preprocess Data to get Tensors
        # _preprocess effectively does the "stacking" of all points from all samples
        # into a single batch [1, Total_Points, Dim]
        static_t, shear_t, visc_t = self._preprocess(df)

        # 2. Build Context Tensor: [Shear, Visc, Static]
        context_t = torch.cat([shear_t, visc_t, static_t], dim=-1)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for i in range(steps):
            # Forward: Model predicts based on context (Self-Consistency)
            # We predict the same points we are looking at to learn their physics
            pred = self.model(context_t, shear_t, static_t)

            loss = F.mse_loss(pred, visc_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f" > Calibration complete. Final MSE: {loss.item():.5f}")

        self.model.eval()
        with torch.no_grad():
            self.memory_vector = self.model.encode_memory(context_t)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts using the cached memory (calibrated state).
        """
        # 1. Context Resolution
        memory_vector = self.memory_vector

        # Handle Zero-Shot (if learn was never called)
        if memory_vector is None:
            memory_vector = torch.zeros((1, self.config["latent_dim"])).to(self.device)

        # 2. Prepare Query
        # _preprocess handles expanding the static vector for the query shear rates
        q_static, q_shear, _ = self._preprocess(df)

        # 3. Decode from Memory (using cached vector)
        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model.decode_from_memory(memory_vector, q_shear, q_static)

        # 4. Inverse Scaling (StandardScaler -> Log10 -> Linear cP)
        q_shear_np = q_shear.cpu().numpy().reshape(-1, 1)
        pred_visc_np = y_scaled.cpu().numpy().reshape(-1, 1)

        # The scaler expects [shear, viscosity], so we stack them
        combined_scaled = np.hstack([q_shear_np, pred_visc_np])
        combined_log = self.physics_scaler.inverse_transform(combined_scaled)

        # Extract viscosity (index 1) and delog (10^x)
        pred_visc_cp = np.power(10, combined_log[:, 1])

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

        return results

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        n_samples: int = 20,
        ci_range: Tuple[float, float] = (2.5, 97.5),
    ):
        self.model.train()  # Enable Dropout

        memory_vector = self.memory_vector
        if memory_vector is None:
            memory_vector = torch.zeros((1, self.config["latent_dim"])).to(self.device)

        q_static, q_shear, _ = self._preprocess(df)

        preds_log = []
        with torch.no_grad():
            for _ in range(n_samples):
                # Use decode_from_memory loop
                out_scaled = self.model.decode_from_memory(
                    memory_vector, q_shear, q_static
                )

                # Inverse Transform per sample
                q_shear_np = q_shear.cpu().numpy().reshape(-1, 1)
                out_np = out_scaled.cpu().numpy().reshape(-1, 1)
                combined = np.hstack([q_shear_np, out_np])

                # Inverse to Log10 space
                log_vals = self.physics_scaler.inverse_transform(combined)[:, 1]
                preds_log.append(log_vals)

        self.model.eval()

        # Convert Log10 -> Linear
        stack_log = np.stack(preds_log)
        stack_linear = np.power(10, stack_log)

        mean_pred = np.mean(stack_linear, axis=0)
        std_pred = np.std(stack_linear, axis=0)
        lower_ci = np.percentile(stack_linear, ci_range[0], axis=0)
        upper_ci = np.percentile(stack_linear, ci_range[1], axis=0)

        stats = {"std": std_pred, "lower_ci": lower_ci, "upper_ci": upper_ci}
        return mean_pred, stats


# ==========================================
# 4. Main Execution (Test)
# ==========================================
if __name__ == "__main__":
    # Test Configuration
    model_dir = "models/experiments/o_net"
    training_file = "data/raw/formulation_data_02052026.csv"

    # 1. Initialize
    try:
        predictor = ViscosityPredictorCNP(model_dir)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit()

    # 2. Simulate Context Loading (Optional)
    if os.path.exists(training_file):
        print(f"Loading context from {training_file}...")
        full_train_df = pd.read_csv(training_file)
        # Filter for a specific protein to simulate "learning" a molecule
        molecule_name = "Vudalimab"
        history_df = full_train_df[
            full_train_df["Protein_type"] == molecule_name
        ].copy()

        if not history_df.empty:
            print(f"Adapting to {molecule_name} ({len(history_df)} samples)...")
            # Using the new Calibrator-style learn loop
            predictor.learn(history_df, steps=50, lr=1e-3)

    # 3. Simulate Prediction
    target_data = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
    F471,Vudalimab,Bispecific,5,152,8.1,0.3,100,25,Histidine,6,15,none,0,none,0,tween-80,0.05,none,0,1.5,1.3,2.63,2.63,2.5,2.19,1.9"""

    target_df = pd.read_csv(io.StringIO(target_data))

    print("\nRunning Prediction...")
    # Uses the cached memory vector
    results = predictor.predict(target_df)

    print("\n--- Results ---")
    cols_pred = [c for c in results.columns if "Pred_" in c]
    print(results[cols_pred].to_string(index=False))
