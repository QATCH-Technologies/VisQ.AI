import io
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Physics Prior Configuration (Table C)
# ==========================================
PRIOR_TABLE = {
    "igg1": {
        "Near-pI": {"arginine": -2, "lysine": -1, "nacl": -1, "proline": 0, "stabilizer": 1, "tween-20": -1, "tween-80": -1},
        "Mixed":   {"arginine": -1, "lysine": -1, "nacl": -1, "proline": -1, "stabilizer": 1, "tween-20": -1, "tween-80": -1},
        "Far":     {"arginine": 0,  "lysine": -1, "nacl": -1, "proline": -1, "stabilizer": 1, "tween-20": -1, "tween-80": -1},
    },
    "igg4": {
        "Near-pI": {"arginine": -2, "lysine": -1, "nacl": -1, "proline": 0, "stabilizer": 1, "tween-20": -1, "tween-80": -1},
        "Mixed":   {"arginine": -2, "lysine": -1, "nacl": -1, "proline": -1, "stabilizer": 1, "tween-20": -1, "tween-80": -1},
        "Far":     {"arginine": -1, "lysine": -1, "nacl": -1, "proline": -1, "stabilizer": 1, "tween-20": -1, "tween-80": -1},
    },
    "fc-fusion": {
        "Near-pI": {"arginine": -1, "lysine": -1, "nacl": -1, "proline": -1, "stabilizer": 1, "tween-20": -2, "tween-80": -2},
        "Mixed":   {"arginine": -1, "lysine": 0,  "nacl": 0,  "proline": -2, "stabilizer": 1, "tween-20": -2, "tween-80": -2},
        "Far":     {"arginine": 0,  "lysine": 0,  "nacl": 0,  "proline": -2, "stabilizer": 1, "tween-20": -2, "tween-80": -2},
    },
    "bispecific": {
        "Near-pI": {"arginine": -2, "lysine": -1, "nacl": -1, "proline": 0, "stabilizer": 1, "tween-20": -1, "tween-80": -1},
        "Mixed":   {"arginine": -1, "lysine": 0,  "nacl": 0,  "proline": -1, "stabilizer": 1, "tween-20": -2, "tween-80": -2},
        "Far":     {"arginine": 0,  "lysine": 0,  "nacl": 0,  "proline": -1, "stabilizer": 1, "tween-20": -2, "tween-80": -2},
    },
    "adc": {
        "Near-pI": {"arginine": -2, "lysine": -1, "nacl": -1, "proline": 0, "stabilizer": 1, "tween-20": -1, "tween-80": -1},
        "Mixed":   {"arginine": -1, "lysine": 0,  "nacl": 0,  "proline": -1, "stabilizer": 1, "tween-20": -2, "tween-80": -2},
        "Far":     {"arginine": 0,  "lysine": 0,  "nacl": 0,  "proline": -1, "stabilizer": 1, "tween-20": -2, "tween-80": -2},
    },
    "bsa": {
        "Near-pI": {"arginine": -1, "lysine": -1, "nacl": -1, "proline": 0, "stabilizer": 1, "tween-20": 0, "tween-80": 0},
        "Mixed":   {"arginine": 0,  "lysine": 0,  "nacl": 0,  "proline": -1, "stabilizer": 1, "tween-20": 0, "tween-80": 0},
        "Far":     {"arginine": 0,  "lysine": 0,  "nacl": 0,  "proline": -1, "stabilizer": 1, "tween-20": 0, "tween-80": 0},
    },
    "polyclonal": {
        "Near-pI": {"arginine": -1, "lysine": -1, "nacl": -1, "proline": 0, "stabilizer": 1, "tween-20": 0, "tween-80": 0},
        "Mixed":   {"arginine": 0,  "lysine": 0,  "nacl": 0,  "proline": -1, "stabilizer": 1, "tween-20": 0, "tween-80": 0},
        "Far":     {"arginine": 0,  "lysine": 0,  "nacl": 0,  "proline": -1, "stabilizer": 1, "tween-20": 0, "tween-80": 0},
    },
    "default": {
        "Near-pI": {"arginine": -1, "lysine": -1, "nacl": 0, "proline": 0, "stabilizer": 1, "tween-20": 0, "tween-80": 0},
        "Mixed":   {"arginine": 0,  "lysine": 0,  "nacl": 0, "proline": -1, "stabilizer": 1, "tween-20": 0, "tween-80": 0},
        "Far":     {"arginine": 0,  "lysine": 0,  "nacl": 0, "proline": -1, "stabilizer": 1, "tween-20": 0, "tween-80": 0},
    }
}

# ==========================================
# 2. Model Definition
# ==========================================
class CrossSampleCNP(nn.Module):
    def __init__(self, static_dim, hidden_dim=128, latent_dim=128, dropout=0.0):
        super().__init__()
        # Encoder: (Shear, Visc) + Static -> Latent
        self.encoder = nn.Sequential(
            nn.Linear(2 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder: Query + Latent -> Prediction
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
        r = torch.mean(encoded, dim=1)
        n_queries = query_shear.size(1)
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)
        decoder_input = torch.cat([query_shear, query_static, r_expanded], dim=-1)
        return self.decoder(decoder_input)

    def encode_memory(self, context_tensor):
        encoded = self.encoder(context_tensor)
        return torch.mean(encoded, dim=1)

    def decode_from_memory(self, memory_vector, query_shear, query_static):
        n_queries = query_shear.size(1)
        mem_expanded = memory_vector.unsqueeze(1).repeat(1, n_queries, 1)
        decoder_input = torch.cat([query_shear, query_static, mem_expanded], dim=-1)
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
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")
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
            "Protein_type", "Protein_class_type", "Buffer_type", "Salt_type",
            "Stabilizer_type", "Surfactant_type", "Excipient_type",
        ]

    # ------------------------------------------------------------------
    # Physics Helpers (Internal)
    # ------------------------------------------------------------------
    def _calculate_cci(self, row):
        c_class = row.get('C_Class', 1.0)
        # Assuming missing defaults
        ph = row.get('Buffer_pH', 7.0)
        pi = row.get('PI_mean', 7.0)
        delta_ph = abs(ph - pi)
        tau = 1.5
        return c_class * np.exp(-delta_ph / tau)

    def _get_physics_prior(self, row):
        # 1. Calculate Regime
        cci = self._calculate_cci(row)
        p_type = str(row.get('Protein_class_type', 'default')).lower()
        
        regime = "Far"
        
        # [cite_start]Determine regime based on class logic [cite: 9]
        if 'igg1' in p_type:
            if cci >= 0.90: regime = "Near-pI"
            elif cci >= 0.50: regime = "Mixed"
        elif 'igg4' in p_type:
            if cci >= 0.80: regime = "Near-pI"
            elif cci >= 0.40: regime = "Mixed"
        elif 'fc-fusion' in p_type or 'trispecific' in p_type:
            if cci >= 0.70: regime = "Near-pI"
            elif cci >= 0.40: regime = "Mixed"
        elif 'bispecific' in p_type or 'adc' in p_type:
            if cci >= 0.80: regime = "Near-pI"
            elif cci >= 0.45: regime = "Mixed"
        elif 'bsa' in p_type or 'polyclonal' in p_type:
            if cci >= 0.70: regime = "Near-pI"
            elif cci >= 0.40: regime = "Mixed"
        else: 
            # Default fallback logic
            if cci >= 0.70: regime = "Near-pI"
            elif cci >= 0.40: regime = "Mixed"

        # [cite_start]2. Select Prior Table [cite: 10, 27]
        lookup_key = "default"
        for key in PRIOR_TABLE.keys():
            if key != "default" and key in p_type:
                lookup_key = key
                break
        
        table = PRIOR_TABLE[lookup_key]
        regime_dict = table.get(regime, table["Far"])

        score = 0.0
        
        # [cite_start]3. Calculate Score based on ingredients [cite: 12]
        ingredient_map = {
            'nacl': 'nacl', 'sucrose': 'stabilizer', 'stabilizer': 'stabilizer',
            'arginine': 'arginine', 'arg': 'arginine',
            'lysine': 'lysine', 'lys': 'lysine', 
            'proline': 'proline', 
            'tween-20': 'tween-20', 'tween-80': 'tween-80', 'polysorbate': 'tween-80'
        }

        # A. Stabilizer Logic
        stab_type = str(row.get("Stabilizer_type", "none")).lower()
        stab_conc = float(row.get("Stabilizer_conc", 0.0))
        if stab_type not in ["none", "unknown", "nan"] and stab_conc > 0:
            score += regime_dict.get("stabilizer", 0)

        # B. Surfactant Logic
        surf_type = str(row.get("Surfactant_type", "none")).lower()
        if "tween-20" in surf_type:
            score += regime_dict.get("tween-20", 0)
        elif "tween-80" in surf_type:
            score += regime_dict.get("tween-80", 0)

        # C. Salt / Excipient Logic
        for col in ["Salt_type", "Excipient_type"]:
            val = str(row.get(col, "none")).lower()
            for key, table_key in ingredient_map.items():
                if key in val:
                    # Avoid double counting if 'stabilizer' appears in excipient col
                    if table_key == 'stabilizer' and col == 'Excipient_type':
                        continue 
                    score += regime_dict.get(table_key, 0)

        return score

    def _preprocess(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                
        # 2. Compute Physics Prior (The new Feature)
        # Apply the logic row-by-row
        # Note: We assign it to the dataframe so the preprocessor (StandardScaler) picks it up
        # The preprocessor MUST have been trained with 'Physics_Prior' as the last numeric column
        df_proc['Physics_Prior'] = df_proc.apply(self._get_physics_prior, axis=1)

        # 3. Run Scikit-Learn Pipeline
        try:
            X_static = self.preprocessor.transform(df_proc)
        except ValueError:
            # Fallback for missing numeric columns (fill 0.0)
            feature_names = (
                self.preprocessor.feature_names_in_
                if hasattr(self.preprocessor, "feature_names_in_")
                else []
            )
            for col in feature_names:
                if col not in df_proc.columns:
                    df_proc[col] = 0.0
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
                if val <= 0: val = 1e-6
                
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

    def learn(self, df: pd.DataFrame, fine_tune: bool = True, steps: int = 50, lr: float = 1e-3):
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

    def predict(self, df: pd.DataFrame, context_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Standard prediction returning DataFrame."""
        memory_vector = self.cached_memory
        
        if context_df is not None:
            c_static, c_shear, c_visc = self._preprocess(context_df)
            c_tensor = torch.cat([c_shear, c_visc, c_static], dim=-1)
            with torch.no_grad():
                memory_vector = self.model.encode_memory(c_tensor)

        if memory_vector is None:
            memory_vector = torch.zeros((1, self.config["latent_dim"])).to(self.device)

        q_static, q_shear, _ = self._preprocess(df)

        self.model.eval()
        with torch.no_grad():
            pred_log = self.model.decode_from_memory(memory_vector, q_shear, q_static)

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
    training_file = "data/processed/formulation_data_augmented_no_trast.csv"
    
    print(f"Loading training data from {training_file}...")
    if not os.path.exists(training_file):
        raise FileNotFoundError(f"Could not find training file: {training_file}")
        
    full_train_df = pd.read_csv(training_file)
    
    # Filter for ONLY Adalimumab samples
    history_df = full_train_df[full_train_df['Protein_type'] == 'Adalimumab'].copy()
    
    print(f"Found {len(history_df)} historical 'Adalimumab' samples.")

    # Define Query Data (Sample F1)
    target_data = """ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
F1,Adalimumab,mab-igg1,3.0,150.0,7.6,1.0,220.0,27.5,Histidine,7.4,10,none,0,none,0.0,none,0.0,arginine,50,0.9,0.9,12.5,11.5,9.8,8.8,6.92"""
    
    target_df = pd.read_csv(io.StringIO(target_data))

    predictor = ViscosityPredictorCNP(model_dir)

    print("\nAdapting model to 'Adalimumab' class behavior...")
    predictor.learn(history_df, fine_tune=True, steps=50, lr=1e-3)

    print("\nPredicting Sample F1 based on adapted knowledge...")
    out = predictor.predict(target_df)

    cols_actual = ["Viscosity_100", "Viscosity_1000", "Viscosity_15000000"]
    cols_pred = ["Pred_Viscosity_100", "Pred_Viscosity_1000", "Pred_Viscosity_15000000"]
    
    print("\n--- RESULTS ---")
    print("Actual (Ground Truth):")
    print(out[cols_actual].to_string(index=False))
    
    print("\nPredicted (After Class Fine-Tuning):")
    print(out[cols_pred].to_string(index=False))