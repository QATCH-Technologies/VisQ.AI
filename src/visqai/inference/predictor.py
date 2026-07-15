"""
predictor.py
============
ViscosityPredictorCNP: loads a trained checkpoint + fitted preprocessor and
serves few-shot predictions (encode-only `.learn()`, `.predict()`,
`.predict_with_uncertainty()` via MC dropout).

Moved from ml/cnp_mk2/inference_o_net.py. Two changes from the original:

1. Row-level feature engineering now goes through
   visqai.preprocessing.pipeline.build_feature_frame instead of a private
   duplicate of the trainer's logic -- this is the fix for the train/inference
   charge-features skew (see pipeline.py's module docstring for the full
   history: charge columns used to be silently zero-filled here).
2. No import-time logging side effect. The original module configured a
   FileHandler (writing a timestamped debug log) at import time, which is why
   ibal_parity_test.py had to monkey-patch logging.basicConfig before
   dynamically importing it. This module just gets a plain
   logging.getLogger(__name__); callers configure handlers if they want them.
"""

from __future__ import annotations

import copy
import os
from typing import Optional, Tuple
import logging

import joblib
import numpy as np
import pandas as pd
import torch

from visqai.models.cnp import CrossSampleCNP
from visqai.preprocessing.pipeline import build_feature_frame, SHEAR_MAP

logger = logging.getLogger(__name__)


class ViscosityPredictorCNP:
    def __init__(self, model_dir: str, verbose: bool = False):
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")
        if not verbose:
            self._logger.setLevel(logging.CRITICAL)
        self._logger.info(f"Initializing ViscosityPredictorCNP with model_dir: {model_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.memory_vector = None  # Stores the calibrated context

        self.preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        self.scaler_path = os.path.join(model_dir, "physics_scaler.pkl")

        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Physics Scaler not found at {self.scaler_path}")

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.physics_scaler = joblib.load(self.scaler_path)

        self.model_path = os.path.join(model_dir, "best_model.pth")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
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

        # Pristine copy of the weights, restored before each learn() call so
        # successive calls for different proteins never contaminate each other.
        self._original_state = copy.deepcopy(self.model.state_dict())

        # Raw context tensor from the last learn() call, used by
        # predict_with_uncertainty for context-subsampling CI.
        self.context_t: Optional[torch.Tensor] = None

        self.shear_map = dict(SHEAR_MAP)

    # ------------------------------------------------------------------
    def _preprocess(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        df_proc = df.copy()

        for col in df_proc.select_dtypes(include=["object"]):
            df_proc[col] = df_proc[col].apply(lambda x: x.value if hasattr(x, "value") else x)

        if "ID" in df_proc.columns:
            df_proc = df_proc.drop(columns=["ID"])

        df_proc, _num_cols, _cat_cols = build_feature_frame(df_proc)

        feature_names = (
            self.preprocessor.feature_names_in_ if hasattr(self.preprocessor, "feature_names_in_") else []
        )
        expected_missing = ["ID"] + list(self.shear_map.keys())
        missing_feats = [c for c in feature_names if c not in df_proc.columns]
        actual_missing = [c for c in missing_feats if c not in expected_missing]
        if actual_missing:
            self._logger.warning(f"Missing static features filled with 0.0: {actual_missing}")
        for c in missing_feats:
            df_proc[c] = 0.0

        X_static = self.preprocessor.transform(df_proc)
        if np.isnan(X_static).any():
            self._logger.warning("NaNs found in X_static after preprocessing! Replacing with 0.")
            X_static = np.nan_to_num(X_static)

        n_rows = len(df_proc)
        n_shears = len(self.shear_map)

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

        scaled_points = self.physics_scaler.transform(raw_points)

        static_t = torch.tensor(np.array(static_list), dtype=torch.float32).unsqueeze(0).to(self.device)
        points_t = torch.tensor(scaled_points.astype(np.float32)).unsqueeze(0).to(self.device)

        shear_t = points_t[:, :, [0]]
        visc_t = points_t[:, :, [1]]
        return static_t, shear_t, visc_t

    # ------------------------------------------------------------------
    def learn(
        self,
        df: pd.DataFrame,
        steps: int = 50,  # kept for API compatibility — no longer used
        lr: float = 1e-3,  # kept for API compatibility — no longer used
        n_draws: int = 20,
        k: int = 8,
    ):
        """Adapts the predictor to a new protein group by encoding its context
        samples into a stable latent memory vector (no weight updates)."""
        if df.empty:
            self._logger.warning("Context DataFrame is empty. Skipping learning.")
            return

        self._logger.info(
            f" > Learn triggered on {len(df)} samples (n_draws={n_draws}, k={k}, no weight updates)."
        )

        self.model.load_state_dict(self._original_state)

        static_t, shear_t, visc_t = self._preprocess(df)
        context_t = torch.cat([shear_t, visc_t, static_t], dim=-1)
        self.context_t = context_t

        n_ctx = context_t.size(1)
        k_eff = min(k, n_ctx)

        self.model.eval()
        memory_draws = []
        with torch.no_grad():
            if n_ctx <= k_eff:
                self.memory_vector = self.model.encode_memory(context_t)
                return

            for _ in range(n_draws):
                idx = torch.randperm(n_ctx, device=self.device)[:k_eff]
                subset = context_t[:, idx, :]
                r = self.model.encode_memory(subset)
                memory_draws.append(r)

        self.memory_vector = torch.stack(memory_draws, dim=0).mean(dim=0)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predicts using the cached memory (calibrated state)."""
        memory_vector = self.memory_vector
        if memory_vector is None:
            self._logger.warning("Memory vector is None. Performing Zero-Shot prediction.")
            memory_vector = torch.zeros((1, self.config["latent_dim"])).to(self.device)

        q_static, q_shear, _ = self._preprocess(df)

        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model.decode_from_memory(memory_vector, q_shear, q_static)

        pred_log10 = self._inverse_to_log(q_shear, y_scaled)
        pred_visc_cp = np.power(10, pred_log10)

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
        n_samples: int = 100,
        ci_range: Tuple[float, float] = (2.5, 97.5),
        k: Optional[int] = None,  # retained for API compatibility — no longer used
    ):
        """Estimates the model's predictive uncertainty via MC Dropout."""
        dropout_val = self.config.get("dropout", 0.0)
        if dropout_val == 0.0:
            self._logger.warning(
                "Model config has dropout=0.0. MC Dropout will produce a zero-width CI."
            )

        q_static, q_shear, _ = self._preprocess(df)

        memory_fixed = (
            self.memory_vector
            if self.memory_vector is not None
            else torch.zeros((1, self.config["latent_dim"]), device=self.device)
        )

        self.model.train()
        preds_log = []
        with torch.no_grad():
            for _ in range(n_samples):
                out_scaled = self.model.decode_from_memory(memory_fixed, q_shear, q_static)
                log_vals = self._inverse_to_log(q_shear, out_scaled)
                preds_log.append(log_vals)
        self.model.eval()

        stack_log = np.stack(preds_log)
        mean_log = np.mean(stack_log, axis=0)
        std_log = np.std(stack_log, axis=0)
        lower_log = np.percentile(stack_log, ci_range[0], axis=0)
        upper_log = np.percentile(stack_log, ci_range[1], axis=0)

        mean_pred = np.power(10, mean_log)
        lower_ci = np.power(10, lower_log)
        upper_ci = np.power(10, upper_log)

        stats = {
            "mean_log10": mean_log,
            "std_log10": std_log,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
        }
        return mean_pred, stats

    def _inverse_to_log(self, q_shear: torch.Tensor, out_scaled: torch.Tensor) -> np.ndarray:
        """Inverse-scales a decoder output tensor to log10 viscosity values."""
        q_shear_np = q_shear.cpu().numpy().reshape(-1, 1)
        out_np = out_scaled.cpu().numpy().reshape(-1, 1)
        combined = np.hstack([q_shear_np, out_np])
        log_vals = self.physics_scaler.inverse_transform(combined)[:, 1]
        return log_vals
