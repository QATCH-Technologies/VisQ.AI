import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. Model Definition (Must match training)
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
# 2. The Predictor Class
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

    def _preprocess(
        self, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            X_static = self.preprocessor.transform(df)
        except ValueError:
            feature_names = (
                self.preprocessor.feature_names_in_
                if hasattr(self.preprocessor, "feature_names_in_")
                else []
            )
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            X_static = self.preprocessor.transform(df)

        points_list = []
        static_list = []

        # Order must be preserved: Row 1 (all shears), Row 2 (all shears)
        for i in range(len(df)):
            for col, shear_val in self.shear_map.items():
                val = 1.0
                if col in df.columns and pd.notna(df.iloc[i][col]):
                    val = df.iloc[i][col]
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
        steps: int = 20,
        lr: float = 1e-4,
    ):
        static_t, shear_t, visc_t = self._preprocess(df)
        context_t = torch.cat([shear_t, visc_t, static_t], dim=-1)

        if fine_tune:
            self.model.train()
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

        Returns:
            mean_pred (np.ndarray): Flattened array of mean predictions.
            stats (dict): Dictionary containing 'std', 'lower_ci', 'upper_ci'.
        """
        self.model.train()  # Enable Dropout

        memory_vector = self.cached_memory
        if memory_vector is None:
            memory_vector = torch.zeros((1, self.config["latent_dim"])).to(self.device)

        q_static, q_shear, _ = self._preprocess(df)

        preds_log = []
        with torch.no_grad():
            for _ in range(n_samples):
                # shape: [1, n_points, 1]
                out_log = self.model.decode_from_memory(
                    memory_vector, q_shear, q_static
                )
                preds_log.append(out_log.cpu().numpy())

        self.model.eval()

        # Stack: [n_samples, 1, n_points, 1]
        stack_log = np.stack(preds_log)
        # Squeeze to [n_samples, n_points]
        stack_linear = np.power(10, stack_log).squeeze()
        if stack_linear.ndim == 1:
            # Handle case of single prediction point
            stack_linear = stack_linear[:, None]

        # Statistics
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
