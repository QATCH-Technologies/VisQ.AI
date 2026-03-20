"""
inference_tnp.py
================
Inference API for the trained TransformerNP model.

Provides TNPPredictor: loads model + scalers and runs zero-shot or few-shot
viscosity predictions on new formulation data.

[TNP-ATTN-4] Context is now encoded at sample level (one latent token per
context sample) via encode_context().  Pass the result as context_encoding to
predict_viscosity().  The old context_tensor kwarg (raw point-level concat) is
still accepted for backward compatibility but triggers a deprecation note.

Usage example (few-shot)
------------------------
    predictor = TNPPredictor(model_dir="./models/experiments/tnp_v1")

    # Build context from known samples (list of dicts with 'points' and 'static')
    ctx_enc = predictor.encode_context(context_sample_dicts)

    # Predict for new formulation
    new_formulation = pd.DataFrame([{ ... }])
    viscosities = predictor.predict_viscosity(new_formulation, context_encoding=ctx_enc)

Usage example (zero-shot)
--------------------------
    viscosities = predictor.predict_viscosity(new_formulation)
"""

import os

import joblib
import numpy as np
import pandas as pd
import torch
from tnp.data import _CAT_COLS, _NUM_COLS, _engineer_features, _process_row_features
from tnp.model import TransformerNP


class TNPPredictor:
    def __init__(self, model_dir: str, device: str = None):
        """
        Load the trained TransformerNP model and its data artifacts.

        Args:
            model_dir: Directory containing best_model.pth, preprocessor.pkl,
                       and physics_scaler.pkl.
            device:    'cuda', 'cpu', or None (auto-detect).
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading TNP Predictor on {self.device}...")

        self.preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
        # [TNP-ATTN-6] Query preprocessor excludes protein identity features
        qry_path = os.path.join(model_dir, "query_preprocessor.pkl")
        self.query_preprocessor = (
            joblib.load(qry_path) if os.path.exists(qry_path) else self.preprocessor
        )
        self.physics_scaler = joblib.load(os.path.join(model_dir, "physics_scaler.pkl"))

        self.shear_mean = self.physics_scaler.mean_[0]
        self.shear_scale = self.physics_scaler.scale_[0]
        self.visc_mean = self.physics_scaler.mean_[1]
        self.visc_scale = self.physics_scaler.scale_[1]

        ckpt_path = os.path.join(model_dir, "best_model.pth")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        config = checkpoint.get("config", {})

        self.static_dim = checkpoint["static_dim"]
        self.model = TransformerNP(
            static_dim=self.static_dim,
            hidden_dim=config.get("hidden_dim", 128),
            latent_dim=config.get("latent_dim", 128),
            n_heads=config.get("n_heads", 4),
            dropout=0.0,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        print("Model and preprocessors loaded successfully.")
        print(
            f"  static_dim={self.static_dim} | "
            f"latent_dim={config.get('latent_dim', 128)} | "
            f"n_heads={config.get('n_heads', 4)}"
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_input(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Run raw inputs through the exact feature engineering pipeline from data.py.

        Returns [B, static_dim] on self.device.
        """
        df = df.copy()

        for c in _NUM_COLS:
            df[c] = df[c].fillna(0.0) if c in df.columns else 0.0
        for c in _CAT_COLS:
            df[c] = (
                df[c].astype(str).str.lower().replace("nan", "unknown")
                if c in df.columns
                else "unknown"
            )

        df, _ = _engineer_features(df)
        features_df = df.apply(_process_row_features, axis=1, result_type="expand")
        df = pd.concat([df, features_df], axis=1)

        try:
            X_stat = self.preprocessor.transform(df)
        except ValueError as e:
            raise ValueError(
                f"Preprocessor transform failed. Ensure your input DataFrame has "
                f"the necessary columns. Details: {e}"
            )

        X_stat = np.nan_to_num(X_stat)
        return torch.tensor(X_stat, dtype=torch.float32).to(self.device)

    # ------------------------------------------------------------------
    # [TNP-ATTN-4] Context encoding
    # ------------------------------------------------------------------

    def encode_context(self, context_sample_dicts: list) -> torch.Tensor:
        """
        Encode a list of context samples into a sample-level latent tensor.

        Each dict must have:
            'points'  : torch.Tensor [N_pts, 2]  — (scaled_log_shear, scaled_log_visc)
                        from the data.py pipeline (physics_scaler-normalised)
            'static'  : torch.Tensor [static_dim] — preprocessed static features

        These are exactly the dicts produced by data.load_and_preprocess().

        Returns:
            [1, N_samples, latent_dim] — ready to pass as context_encoding to
            predict_viscosity().
        """
        if not context_sample_dicts:
            raise ValueError("context_sample_dicts must not be empty.")

        ctx_items_list = []
        for s in context_sample_dicts:
            stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
            raw = torch.cat([s["points"], stat], dim=1).to(self.device)
            ctx_items_list.append(raw)

        with torch.no_grad():
            return self.model.encode_context_samples(ctx_items_list)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_viscosity(
        self,
        formulation_df: pd.DataFrame,
        shear_rates: list = None,
        context_encoding: torch.Tensor = None,
        context_tensor: torch.Tensor = None,
        temperature: float = None,
    ) -> pd.DataFrame:
        """
        Predict viscosity for the given formulations at specific shear rates.

        Args:
            formulation_df:    Pandas DataFrame of formulation parameters.
            shear_rates:       Shear rates (s^-1) to predict.
                               Defaults to the canonical 5-point set.
            context_encoding:  [1, N_samples, latent_dim] — pre-encoded context,
                               as returned by encode_context().  Preferred.
            context_tensor:    [1, N_pts, 2+static_dim] — raw point-level context
                               (backward compat; still works but uses point-level
                               attention tokens, not the preferred sample-level).
            temperature:       Attention temperature override.
                               None (default) -> use the model's learned temperature.

        Returns:
            DataFrame with columns [Protein_type, Viscosity_{shear}, ...].
        """
        if shear_rates is None:
            shear_rates = [100.0, 1000.0, 10000.0, 100000.0, 15000000.0]

        X_stat = self._preprocess_input(formulation_df)  # [B, static_dim]
        B = X_stat.shape[0]
        n_q = len(shear_rates)

        log_shears = np.log10(shear_rates)
        scaled_shears = (log_shears - self.shear_mean) / self.shear_scale

        q_shear = torch.tensor(scaled_shears, dtype=torch.float32, device=self.device)
        q_shear = q_shear.view(1, n_q, 1).expand(B, -1, -1)
        q_stat = X_stat.unsqueeze(1).expand(-1, n_q, -1)

        # Resolve context
        if context_encoding is not None:
            # Preferred: sample-level encoded context [1, N_samples, latent_dim]
            ctx = context_encoding.expand(B, -1, -1).to(self.device)
            ctx_is_encoded = True
        elif context_tensor is not None:
            # Backward compat: raw point-level tensor [1, N_pts, 2+static_dim]
            # This still works but attends over N_pts tokens (not N_samples).
            # For best results use encode_context() instead.
            ctx = context_tensor.expand(B, -1, -1).to(self.device)
            ctx_is_encoded = False
        else:
            # Zero-shot: single zero token in latent space
            ctx = torch.zeros(
                B, 1, self.model.latent_dim, dtype=torch.float32, device=self.device
            )
            ctx_is_encoded = True

        with torch.no_grad():
            pred_scaled, _ = self.model(
                ctx,
                q_shear,
                q_stat,
                temperature=temperature,
                ctx_is_encoded=ctx_is_encoded,
            )

        pred_scaled_np = pred_scaled.squeeze(-1).cpu().numpy()  # [B, n_q]
        pred_log = pred_scaled_np * self.visc_scale + self.visc_mean
        pred_visc = 10.0**pred_log

        results = pd.DataFrame(
            pred_visc,
            columns=[f"Viscosity_{int(s)}" for s in shear_rates],
        )
        results.insert(
            0, "Protein_type", formulation_df.get("Protein_type", "Sample").values
        )
        return results


if __name__ == "__main__":
    predictor = TNPPredictor(model_dir="./models/experiments/tnp_v1")

    test_data = pd.DataFrame(
        [
            {
                "ID": 1,
                "Protein_type": "poly-hIgG",
                "Protein_class_type": "Polyclonal",
                "kP": 3.0,
                "MW": 150.0,
                "PI_mean": 7.6,
                "PI_range": 1.0,
                "Protein_conc": 200.0,
                "Temperature": 25.24,
                "Buffer_type": "Histidine",
                "Buffer_pH": 6.0,
                "Buffer_conc": 15.0,
                "Salt_type": "None",
                "Salt_conc": 0.0,
                "Stabilizer_type": "Sucrose",
                "Stabilizer_conc": 0.5,
                "Surfactant_type": "None",
                "Surfactant_conc": 0.0,
                "Excipient_type": "None",
                "Excipient_conc": 0.0,
                "C_Class": 0.9,
                "HCI": 0.9,
            }
        ]
    )

    print("\n--- Zero-Shot Prediction Results ---")
    predictions = predictor.predict_viscosity(test_data)
    print(predictions.to_string(index=False))
