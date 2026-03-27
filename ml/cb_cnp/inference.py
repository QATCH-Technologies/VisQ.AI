"""
inference.py
============
Inference module for the ConceptBottleneckCNP viscosity prediction system.

Design contract
---------------
This module owns *only* inference orchestration:
  - loading serialised artifacts (preprocessor, physics scaler, model weights)
  - wiring the three public endpoints (learn / predict / predict_with_uncertainty)
  - the CBM causal API (get_concept_state / intervene)

Everything else is delegated:
  - feature engineering  ->  ``data_pipeline._engineer_features``
  - pH-regime priors     ->  ``data_pipeline._process_row_features``
  - model architecture   ->  ``models.ConceptBottleneckCNP``
  - column/concept defs  ->  ``constants``

Public API
----------
predictor = InferenceCNP(model_dir)

predictor.learn(df)                           # encode context -> concept memory
predictor.predict(df)                         # decode memory -> cP DataFrame
predictor.predict_with_uncertainty(df, ...)   # MC Dropout CI

predictor.get_concept_state()                 # {concept_name: activation}
predictor.intervene(df, concept_idx, value)   # causal do-intervention
"""

from __future__ import annotations
from scipy.interpolate import PchipInterpolator

import contextlib
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

# ---------------------------------------------------------------------------
# Module imports — all architecture and feature-engineering logic lives here.
# ---------------------------------------------------------------------------
try:
    from models import ConceptBottleneckCNP
    from data_pipeline import _engineer_features, _process_row_features, CONC_THRESHOLDS
    from constants import CONCEPT_NAMES, N_CONCEPTS_SUPERVISED
except ImportError:
    from cb_cnp.models import ConceptBottleneckCNP
    from cb_cnp.data_pipeline import _engineer_features, _process_row_features, CONC_THRESHOLDS
    from cb_cnp.constants import CONCEPT_NAMES, N_CONCEPTS_SUPERVISED


# ---------------------------------------------------------------------------
# Module-level column constants
# Must stay in sync with data_pipeline.load_and_preprocess.
# ---------------------------------------------------------------------------

#: Base numeric input columns expected in any raw formulation DataFrame.
_NUM_COLS: list[str] = [
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

#: Categorical columns normalised to lowercase before the preprocessor sees them.
#: Matches the cat_cols used when the ColumnTransformer was fitted in data_pipeline.
_CAT_COLS: list[str] = [
    "Protein_class_type",
    "Buffer_type",
    "Salt_type",
    "Stabilizer_type",
    "Surfactant_type",
    "Excipient_type",
]

#: Canonical shear-rate -> column-name mapping.  Shared by _preprocess and predict.
_SHEAR_MAP: dict[str, float] = {
    "Viscosity_100": 100.0,
    "Viscosity_1000": 1000.0,
    "Viscosity_10000": 10000.0,
    "Viscosity_100000": 100000.0,
    "Viscosity_15000000": 1.5e7,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure the module logger.

    When *verbose* is False (the default) all output below CRITICAL is
    suppressed — no files are created, no stdout lines appear.
    """
    logger = logging.getLogger("VisQ_Inference")
    logger.handlers.clear()

    if not verbose:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"inference_{stamp}.log")

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class InferenceCNP:
    """
    Inference wrapper for ``ConceptBottleneckCNP`` checkpoints.

    The checkpoint produced by ``train.py`` must contain:
      - ``best_model.pth``   — weights + config + metadata
      - ``preprocessor.pkl`` — fitted sklearn ColumnTransformer
      - ``physics_scaler.pkl``— fitted StandardScaler for (log-shear, log-visc)

    Parameters
    ----------
    model_dir : str
        Directory produced by ``train.py``.
    verbose : bool
        When False (default) all log output below CRITICAL is suppressed.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, model_dir: str, verbose: bool = False) -> None:
        self._logger = _setup_logging(verbose)
        self._logger.info(f"Initialising InferenceCNP from: {model_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir

        # Public state — reset by the caller before each protein group.
        self.memory_vector: Optional[torch.Tensor] = None
        self.context_t: Optional[torch.Tensor] = None

        # Expose shear map for external callers (e.g. smoke-test harness).
        self.shear_map = _SHEAR_MAP

        # --- Load serialised preprocessors ---
        preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
        scaler_path = os.path.join(model_dir, "physics_scaler.pkl")
        for path in (preprocessor_path, scaler_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required artifact not found: {path}")

        self._logger.debug("Loading preprocessor and physics scaler...")
        self.preprocessor = joblib.load(preprocessor_path)
        self.physics_scaler = joblib.load(scaler_path)

        # --- Load model checkpoint ---
        model_path = os.path.join(model_dir, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self._logger.debug(f"Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        self.config: dict = checkpoint["config"]
        self.static_dim: int = checkpoint["static_dim"]
        self._logger.debug(f"config={self.config}")

        # --- Instantiate and restore model ---
        self.model: ConceptBottleneckCNP = self._build_model(checkpoint)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        # Pristine copy — restored before every learn() to prevent
        # cross-protein weight contamination (FIX-2).
        self._original_state = copy.deepcopy(self.model.state_dict())

        # Memory dimensionality is always n_concepts for CBM.
        self._memory_dim: int = checkpoint.get("n_concepts", self.config["latent_dim"])

    # ------------------------------------------------------------------
    # Model construction — delegates entirely to models.py
    # ------------------------------------------------------------------

    def _build_model(self, checkpoint: dict) -> ConceptBottleneckCNP:
        """
        Instantiate ``ConceptBottleneckCNP`` from checkpoint metadata.

        All architecture hyperparameters are sourced from the checkpoint so
        this file contains zero architecture definitions.
        """
        return ConceptBottleneckCNP(
            static_dim=self.static_dim,
            hidden_dim=self.config["hidden_dim"],
            latent_dim=self.config["latent_dim"],
            n_concepts=checkpoint.get("n_concepts", N_CONCEPTS_SUPERVISED),
            concept_names=checkpoint.get("concept_names", CONCEPT_NAMES),
            concept_activations=checkpoint.get("concept_activations"),
            dropout=self.config.get("dropout", 0.0),
        ).to(self.device)

    # ------------------------------------------------------------------
    # Preprocessing — delegates feature engineering to data_pipeline
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        df: pd.DataFrame,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform a raw formulation DataFrame into model-ready tensors.

        Feature engineering is fully delegated to ``data_pipeline``:
          - physics features (unit conversions, interaction terms, volume
            fractions, electrostatics) via ``_engineer_features``
          - pH-regime priors and concentration-split features via
            ``_process_row_features``

        The fitted ``preprocessor`` and ``physics_scaler`` (loaded from disk
        in ``__init__``) are then applied so the inference feature space is
        identical to the training feature space.

        Returns
        -------
        static_t : Tensor [1, N_rows × N_shears, static_dim]
        shear_t  : Tensor [1, N_rows × N_shears, 1]
        visc_t   : Tensor [1, N_rows × N_shears, 1]
        """
        self._logger.debug(f"_preprocess | input shape={df.shape}")

        df_proc = df.copy()

        # Unwrap any enum/object wrappers (e.g. from PyQt combo boxes).
        for col in df_proc.select_dtypes(include=["object"]):
            df_proc[col] = df_proc[col].apply(lambda x: x.value if hasattr(x, "value") else x)

        if "ID" in df_proc.columns:
            df_proc.drop(columns=["ID"], inplace=True)

        # 1. Numeric fill-defaults — must happen before _engineer_features
        #    so derived columns never see NaN inputs.
        for c in _NUM_COLS:
            if c not in df_proc.columns:
                df_proc[c] = 0.0
            else:
                df_proc[c] = df_proc[c].fillna(0.0)

        # 2. Categorical normalisation — must match the encoding seen by the
        #    ColumnTransformer when it was fitted in data_pipeline.
        for c in _CAT_COLS:
            if c in df_proc.columns:
                df_proc[c] = df_proc[c].astype(str).str.lower().replace("nan", "unknown")
            else:
                df_proc[c] = "unknown"

        # 3. Physics feature engineering — fully delegated to data_pipeline.
        #    Stdout is suppressed so the verbose print inside _engineer_features
        #    does not surface during quiet inference.
        self._logger.debug("Delegating to data_pipeline._engineer_features...")
        with contextlib.redirect_stdout(io.StringIO()):
            df_proc, _ = _engineer_features(df_proc)

        # 4. pH-regime priors and concentration-split features — delegated to
        #    data_pipeline._process_row_features (one row at a time).
        self._logger.debug("Delegating to data_pipeline._process_row_features...")
        new_features = df_proc.apply(_process_row_features, axis=1, result_type="expand")
        df_proc = pd.concat([df_proc, new_features], axis=1)

        # 5. Align to the training feature space, then transform.
        feature_names: list[str] = (
            list(self.preprocessor.feature_names_in_)
            if hasattr(self.preprocessor, "feature_names_in_")
            else []
        )
        expected_absent = ["ID"] + list(_SHEAR_MAP.keys())
        missing = [c for c in feature_names if c not in df_proc.columns]
        unexpected = [c for c in missing if c not in expected_absent]
        if unexpected:
            self._logger.warning(f"Filling unexpected missing features with 0: {unexpected}")
        for col in missing:
            df_proc[col] = 0.0

        X_static = self.preprocessor.transform(df_proc)
        if np.isnan(X_static).any():
            self._logger.warning("NaNs in X_static after transform — replacing with 0.")
            X_static = np.nan_to_num(X_static)

        self._logger.debug(f"Static feature matrix shape: {X_static.shape}")

        # 6. Build the (log-shear, log-viscosity) point tensors.
        #    Each formulation row expands to N_shears points; viscosity values
        #    fall back to 1.0 cP (= 0.0 in log10) when the column is absent
        #    (i.e. for query-only rows without ground-truth labels).
        n_rows = len(df_proc)
        n_shears = len(_SHEAR_MAP)

        raw_points = np.empty((n_rows * n_shears, 2), dtype=np.float64)
        static_list: list[np.ndarray] = []

        row_idx = 0
        for i in range(n_rows):
            for col, shear_val in _SHEAR_MAP.items():
                val = 1.0
                if col in df_proc.columns and pd.notna(df_proc.iloc[i][col]):
                    val = float(df_proc.iloc[i][col])
                raw_points[row_idx, 0] = np.log10(shear_val)
                raw_points[row_idx, 1] = np.log10(max(val, 1e-6))
                static_list.append(X_static[i])
                row_idx += 1

        scaled_points = self.physics_scaler.transform(raw_points)

        static_t = (
            torch.tensor(np.array(static_list), dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        points_t = torch.tensor(scaled_points.astype(np.float32)).unsqueeze(0).to(self.device)
        shear_t = points_t[:, :, [0]]
        visc_t = points_t[:, :, [1]]

        self._logger.debug(
            f"Tensors: static={static_t.shape}, shear={shear_t.shape}, visc={visc_t.shape}"
        )
        return static_t, shear_t, visc_t

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inverse_to_log(
        self,
        q_shear: torch.Tensor,
        out_scaled: torch.Tensor,
    ) -> np.ndarray:
        """Inverse-scale a decoder output tensor back to log10 viscosity."""
        q_shear_np = q_shear.cpu().numpy().reshape(-1, 1)
        out_np = out_scaled.cpu().numpy().reshape(-1, 1)
        combined = np.hstack([q_shear_np, out_np])
        return self.physics_scaler.inverse_transform(combined)[:, 1]

    def _zero_shot_memory(self) -> torch.Tensor:
        """Return a zeroed concept vector (zero-shot fallback)."""
        return torch.zeros((1, self._memory_dim), device=self.device)

    def _unpack_predictions(self, pred_visc: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scatter a flat viscosity array back into per-row, per-shear columns.

        Parameters
        ----------
        pred_visc : np.ndarray (n_rows × n_shears,) — linear cP values
        df        : original query DataFrame (used as the result scaffold)

        Returns
        -------
        pd.DataFrame — df with ``Pred_Viscosity_{shear}`` columns appended.
        """
        n_shears = len(_SHEAR_MAP)
        shear_keys = list(_SHEAR_MAP.keys())
        new_cols: dict[str, list] = {k: [] for k in shear_keys}
        for i in range(len(df)):
            start = i * n_shears
            for j, key in enumerate(shear_keys):
                new_cols[key].append(pred_visc[start + j])
        results = df.copy()
        for key, vals in new_cols.items():
            results[f"Pred_{key}"] = vals
        return results

    # ------------------------------------------------------------------
    # Internal helper — dense PCHIP context builder (Option 1 / 2 fix)
    # ------------------------------------------------------------------

    def _build_dense_context_samples(self, df: pd.DataFrame) -> list[torch.Tensor]:
        """
        Build a per-sample list of PCHIP-dense context tensors.

        Mirrors ``data_pipeline._build_samples`` exactly so that inference
        sees the same point density (~40 pts/sample) and intact shear profiles
        that the model was trained on.  Eliminates both distribution mismatches:

          - Mismatch 1: fragmented per-point draws → whole-sample draws
          - Mismatch 2: sparse 5-point profiles  → dense PCHIP profiles

        Returns
        -------
        list of Tensors, one per valid row, each shaped [n_pts_i, 2 + static_dim]
        where the columns are (scaled_log_shear, scaled_log_visc, *static_features).
        """
        # ---- Static feature matrix (same pipeline as _preprocess) ---------
        df_proc = df.copy()
        for col in df_proc.select_dtypes(include=["object"]):
            df_proc[col] = df_proc[col].apply(lambda x: x.value if hasattr(x, "value") else x)
        if "ID" in df_proc.columns:
            df_proc = df_proc.drop(columns=["ID"])

        for c in _NUM_COLS:
            if c not in df_proc.columns:
                df_proc[c] = 0.0
            else:
                df_proc[c] = df_proc[c].fillna(0.0)
        for c in _CAT_COLS:
            if c in df_proc.columns:
                df_proc[c] = df_proc[c].astype(str).str.lower().replace("nan", "unknown")
            else:
                df_proc[c] = "unknown"

        with contextlib.redirect_stdout(io.StringIO()):
            df_proc, _ = _engineer_features(df_proc)
        new_features = df_proc.apply(_process_row_features, axis=1, result_type="expand")
        df_proc = pd.concat([df_proc, new_features], axis=1)

        feature_names: list[str] = (
            list(self.preprocessor.feature_names_in_)
            if hasattr(self.preprocessor, "feature_names_in_")
            else []
        )
        for col in [c for c in feature_names if c not in df_proc.columns]:
            df_proc[col] = 0.0

        X_static = self.preprocessor.transform(df_proc)
        if np.isnan(X_static).any():
            X_static = np.nan_to_num(X_static)

        # ---- Per-row PCHIP interpolation (matches _build_samples exactly) -
        key_logs = np.log10([100.0, 1_000.0, 10_000.0, 100_000.0, 15_000_000.0])
        df_reset = df.reset_index(drop=True)
        sample_tensors: list[torch.Tensor] = []

        for i in range(len(df_reset)):
            raw_x, raw_y = [], []
            for col, shear_val in _SHEAR_MAP.items():
                if col in df_reset.columns and pd.notna(df_reset.iloc[i][col]):
                    v = float(df_reset.iloc[i][col])
                    if v <= 0:
                        v = 1e-6
                    raw_x.append(np.log10(shear_val))
                    raw_y.append(np.log10(v))

            if len(raw_x) < 3:
                self._logger.warning(
                    f"_build_dense_context_samples | row {i}: "
                    f"only {len(raw_x)} visc points — skipping."
                )
                continue

            si = np.argsort(raw_x)
            x_arr = np.array(raw_x)[si]
            y_arr = np.array(raw_y)[si]
            interpolator = PchipInterpolator(x_arr, y_arr)

            # Dense grid — identical construction to _build_samples
            endpoints = np.unique(np.concatenate([x_arr, key_logs]))
            endpoints = endpoints[(endpoints >= x_arr.min()) & (endpoints <= x_arr.max())]
            endpoints.sort()

            dense_x_list = []
            for j in range(len(endpoints) - 1):
                seg = np.linspace(endpoints[j], endpoints[j + 1], 10)
                dense_x_list.append(seg[:-1] if j < len(endpoints) - 2 else seg)
            dense_x = np.concatenate(dense_x_list) if dense_x_list else x_arr
            dense_y = interpolator(dense_x)

            # Scale (log-shear, log-visc) with the fitted physics_scaler
            raw_pts = np.column_stack([dense_x, dense_y])
            scaled_pts = self.physics_scaler.transform(raw_pts)  # [n_pts, 2]

            # Repeat static row for every dense point
            static_row = X_static[i]  # [static_dim]
            static_rep = np.tile(static_row, (len(scaled_pts), 1))  # [n_pts, static_dim]

            # Final layout: [shear | visc | static] — matches encode_memory input
            ctx_np = np.concatenate(
                [scaled_pts[:, [0]], scaled_pts[:, [1]], static_rep], axis=1
            ).astype(np.float32)

            sample_tensors.append(torch.tensor(ctx_np, device=self.device))

        self._logger.debug(
            f"_build_dense_context_samples | {len(sample_tensors)} valid samples built."
        )
        return sample_tensors

    # ------------------------------------------------------------------
    # Public API — context encoding
    # ------------------------------------------------------------------

    def learn(
        self,
        df: pd.DataFrame,
        steps: int = 50,  # kept for API compatibility — no longer used
        lr: float = 1e-3,  # kept for API compatibility — no longer used
        n_draws: int = 20,
        k: int = 8,
        small_pool_threshold: int = 15,
    ) -> None:
        """
        Encode context samples into a stable concept memory vector.

        No weight updates are performed — the CBM adapts purely through the
        concept vector.  Model weights are restored from the pristine checkpoint
        before each call so successive proteins are fully isolated (FIX-2).

        Distribution-mismatch fix (Options 1 + 2):
          - Context samples are built with PCHIP-dense shear profiles matching
            training, and draws select *complete samples* rather than individual
            points (Option 1).
          - When the context pool is small (≤ ``small_pool_threshold``), all
            samples are encoded in a single pass with no subsampling (Option 2).

        Parameters
        ----------
        df                   : Context samples for the target protein.
        steps                : Ignored (API compatibility).
        lr                   : Ignored (API compatibility).
        n_draws              : Number of random context subsets to encode and average.
        k                    : Number of *complete samples* per draw.
        small_pool_threshold : Pool sizes ≤ this skip multi-draw subsampling.
        """
        if df.empty:
            self._logger.warning("Empty context DataFrame — skipping learn().")
            print("Warning: context DataFrame is empty, skipping learn().")
            return

        self._logger.info(f"learn() | n_ctx={len(df)}, n_draws={n_draws}, k={k}")
        print(f" > Encoding context: {len(df)} samples, {n_draws} draws of k={k}…")

        # FIX-2: restore pristine weights before each protein's context.
        self.model.load_state_dict(self._original_state)
        self.model.eval()

        # Build dense PCHIP tensors — one per valid context sample.
        sample_tensors = self._build_dense_context_samples(df)
        n_samples = len(sample_tensors)

        if n_samples == 0:
            self._logger.warning("No valid context samples after PCHIP build — aborting.")
            print("Warning: no valid context samples (each row needs ≥ 3 viscosity points).")
            return

        n_pts_avg = int(np.mean([s.shape[0] for s in sample_tensors]))
        print(f" > Built {n_samples} dense samples (~{n_pts_avg} pts/sample).")

        def _encode_indices(indices: list[int]) -> torch.Tensor:
            """Concatenate the selected complete samples and encode."""
            pts = torch.cat([sample_tensors[i] for i in indices], dim=0)  # [total_pts, C]
            return self.model.encode_memory(pts.unsqueeze(0))  # [1, memory_dim]

        with torch.no_grad():
            # Option 2 — small pool: single full-context pass, no subsampling.
            if n_samples <= small_pool_threshold:
                self._logger.debug(
                    f"Small pool ({n_samples} ≤ {small_pool_threshold}): "
                    "single full-context encode."
                )
                self.memory_vector = _encode_indices(list(range(n_samples)))
                # context_t: flat concatenation for backward compatibility.
                self.context_t = torch.cat([s.unsqueeze(0) for s in sample_tensors], dim=1)
                norm = self.memory_vector.norm().item()
                self._logger.info(f"Encoding complete (full pool). norm={norm:.3f}")
                print(f" > Encoding complete (full pool). Memory norm: {norm:.3f}")
                return

            # Option 1 — large pool: draw k *complete* samples per draw.
            k_eff = min(k, n_samples)
            draws: list[torch.Tensor] = []
            for draw_i in range(n_draws):
                idx = torch.randperm(n_samples, device=self.device)[:k_eff].tolist()
                mem = _encode_indices(idx)
                draws.append(mem)
                self._logger.debug(
                    f"  Draw {draw_i + 1}/{n_draws}: "
                    f"samples={idx}, norm={mem.norm().item():.3f}"
                )

        self.memory_vector = torch.stack(draws, dim=0).mean(dim=0)
        # context_t: flat concatenation of all samples for backward compatibility.
        self.context_t = torch.cat([s.unsqueeze(0) for s in sample_tensors], dim=1)
        norm = self.memory_vector.norm().item()
        self._logger.info(
            f"Encoding complete ({n_draws} draws of k={k_eff}). "
            f"norm={norm:.3f}, shape={self.memory_vector.shape}"
        )
        print(f" > Encoding complete. Memory norm: {norm:.3f}")

    # ------------------------------------------------------------------
    # Public API — prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict viscosity at all canonical shear rates from the cached memory.

        Falls back to a zero concept vector (zero-shot) if ``learn()`` has not
        been called.  Output columns are ``Pred_Viscosity_{shear}``.

        Parameters
        ----------
        df : Query formulations.

        Returns
        -------
        pd.DataFrame — input DataFrame with prediction columns appended.
        """
        self._logger.info(f"predict() | n_queries={len(df)}")

        memory = self.memory_vector if self.memory_vector is not None else self._zero_shot_memory()
        if self.memory_vector is None:
            self._logger.warning("No memory vector — falling back to zero-shot.")

        q_static, q_shear, _ = self._preprocess(df)

        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model.decode_from_memory(memory, q_shear, q_static)

        pred_visc = np.power(10, self._inverse_to_log(q_shear, y_scaled))
        self._logger.info("predict() complete.")
        return self._unpack_predictions(pred_visc, df)

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        n_samples: int = 100,
        ci_range: Tuple[float, float] = (2.5, 97.5),
        k: Optional[int] = None,  # retained for API compatibility — ignored
    ) -> Tuple[np.ndarray, dict]:
        """
        Estimate predictive uncertainty via MC Dropout.

        The concept memory is held fixed; the decoder is called ``n_samples``
        times in ``train()`` mode so each pass draws a different dropout mask.
        All statistics are computed in log10 space to avoid right-tail distortion.

        Requires ``dropout > 0`` in the model config; if ``dropout == 0.0`` the
        CI will be zero-width and a warning is emitted.

        Parameters
        ----------
        df        : Query formulations.
        n_samples : Stochastic decoder passes (100 recommended).
        ci_range  : Percentile bounds for the confidence interval.
        k         : Deprecated — accepted for compatibility, ignored.

        Returns
        -------
        mean_pred : np.ndarray (n_queries,) — mean in linear cP
        stats     : dict with keys
                    ``mean_log10``, ``std_log10``, ``lower_ci``, ``upper_ci``
        """
        dropout_val = self.config.get("dropout", 0.0)
        self._logger.info(
            f"predict_with_uncertainty() | n_samples={n_samples}, "
            f"ci_range={ci_range}, dropout={dropout_val}"
        )
        if dropout_val == 0.0:
            msg = (
                "dropout=0.0 in checkpoint — CI will be zero-width. "
                "Retrain with dropout > 0.0 for meaningful uncertainty estimates."
            )
            self._logger.warning(msg)
            print(f"WARNING: {msg}")

        q_static, q_shear, _ = self._preprocess(df)

        memory = self.memory_vector if self.memory_vector is not None else self._zero_shot_memory()
        if self.memory_vector is None:
            self._logger.warning("No memory vector — zero-shot; CI reflects decoder noise only.")

        # Activate dropout by switching to train() mode.
        self.model.train()
        preds_log: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(n_samples):
                out_scaled = self.model.decode_from_memory(memory, q_shear, q_static)
                preds_log.append(self._inverse_to_log(q_shear, out_scaled))
                if (i + 1) % 25 == 0:
                    self._logger.debug(f"  MC pass {i + 1}/{n_samples}")
        self.model.eval()

        stack_log = np.stack(preds_log)  # (n_samples, n_queries)
        mean_log = stack_log.mean(axis=0)
        std_log = stack_log.std(axis=0)

        stats = {
            "mean_log10": mean_log,
            "std_log10": std_log,
            "lower_ci": np.power(10, np.percentile(stack_log, ci_range[0], axis=0)),
            "upper_ci": np.power(10, np.percentile(stack_log, ci_range[1], axis=0)),
        }

        self._logger.info(f"Uncertainty complete. Mean std_log10: {std_log.mean():.4f}")
        return np.power(10, mean_log), stats

    # ------------------------------------------------------------------
    # CBM causal API
    # ------------------------------------------------------------------

    def get_concept_state(self) -> Optional[dict[str, float]]:
        """
        Return the current concept memory as a named dictionary.

        Only meaningful after ``learn()`` has been called.

        Returns
        -------
        dict {concept_name: activation_value}, or None if learn() not yet called.
        """
        if self.memory_vector is None:
            self._logger.debug("get_concept_state() called before learn() — returning None.")
            return None

        c_np = self.memory_vector.squeeze(0).cpu().numpy()
        return {name: float(val) for name, val in zip(self.model.concept_names, c_np)}

    def intervene(
        self,
        df: pd.DataFrame,
        concept_idx: int | list[int],
        concept_value: float,
    ) -> pd.DataFrame:
        """
        Causal do-intervention: clamp concept dimension(s) and re-decode.

        Implements do(c_i = v) in the Pearl causal sense.  The stored memory
        vector is cloned — the original is never mutated.  Falls back to a
        zero memory vector if ``learn()`` has not been called.

        Parameters
        ----------
        df            : Query formulations.
        concept_idx   : Concept index (int) or list of indices to clamp.
        concept_value : Value to clamp to; should be in the activation range
                        of that concept ([-1, 1] for tanh, [0, 1] for sigmoid).

        Returns
        -------
        pd.DataFrame with ``Pred_Viscosity_{shear}`` columns.
        """
        base_memory = (
            self.memory_vector if self.memory_vector is not None else self._zero_shot_memory()
        )
        if self.memory_vector is None:
            self._logger.warning("intervene() called before learn() — using zero-shot memory.")
            print("Warning: intervene() called before learn(). Using zero-shot memory.")

        # Clamp on a clone so the stored concept state is never modified.
        c_mod = base_memory.clone()
        if isinstance(concept_idx, int):
            concept_idx = [concept_idx]
        for idx in concept_idx:
            c_mod[:, idx] = concept_value

        q_static, q_shear, _ = self._preprocess(df)

        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model.decode_from_memory(c_mod, q_shear, q_static)

        pred_visc = np.power(10, self._inverse_to_log(q_shear, y_scaled))
        applied = {self.model.concept_names[i]: concept_value for i in concept_idx}
        self._logger.info(f"intervene() complete. Applied: {applied}")
        return self._unpack_predictions(pred_visc, df)


# ---------------------------------------------------------------------------
# Smoke-test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_dir = "models/experiments/cbm_cnp_v4"
    training_file = "data/raw/formulation_data_03042026.csv"

    # 1. Initialise
    try:
        predictor = InferenceCNP(model_dir, verbose=False)
        print(f"Model loaded: ConceptBottleneckCNP | {predictor.model.n_concepts} concepts")
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        raise SystemExit(1)

    # 2. Context pool
    if not os.path.exists(training_file):
        print(f"Error: training file not found at {training_file}")
        raise SystemExit(1)

    print(f"Loading context pool from {training_file}…")
    full_train_df = pd.read_csv(training_file)
    for col in full_train_df.select_dtypes(include=["int", "int64", "int32"]).columns:
        if col != "ID":
            full_train_df[col] = full_train_df[col].astype(float)
    full_train_df["ID"] = full_train_df["ID"].astype(str)

    # 3. Target samples
    target_data = """\
ID,Protein_type,Protein_class_type,kP,MW,PI_mean,PI_range,Protein_conc,Temperature,\
Buffer_type,Buffer_pH,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,\
Surfactant_type,Surfactant_conc,Excipient_type,Excipient_conc,C_Class,HCI,\
Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000
511,poly-hIgG,Polyclonal,3.0,150.0,7.6,1.0,83.0,25.0,Histidine,6.0,15.0,None,0.0,\
Sucrose,0.25,None,0.0,None,0.0,0.9,0.9,4.15,4.24,4.22,3.89,1.72
630,Adalimumab,mAb_IgG1,3.0,148.0,8.7,0.3,206.0,25.35,Histidine,6.0,15.0,NaCl,70.0,\
Sucrose,0.2,tween-80,0.05,None,0.0,1.0,1.0,36.4,38.0,39.6,40.9,6.36"""

    import io as _io

    target_df = pd.read_csv(_io.StringIO(target_data))
    target_df["ID"] = target_df["ID"].astype(str)

    all_results: list[pd.DataFrame] = []
    shear_cols = ["100", "1000", "10000", "100000", "15000000"]

    # 4. Per-protein predict
    for protein in target_df["Protein_type"].unique():
        print(f"\n--- Processing: {protein} ---")
        prot_target = target_df[target_df["Protein_type"] == protein].copy()
        target_ids = prot_target["ID"].tolist()

        history_df = full_train_df[
            (full_train_df["Protein_type"] == protein) & (~full_train_df["ID"].isin(target_ids))
        ].copy()

        predictor.memory_vector = None
        predictor.context_t = None

        if not history_df.empty:
            print(f"Adapting to {protein} ({len(history_df)} context samples)…")
            predictor.learn(history_df)
        else:
            print(f"No history for {protein} — zero-shot.")

        results_df = predictor.predict(prot_target)
        all_results.append(results_df)

        concept_state = predictor.get_concept_state()
        if concept_state:
            print(f"\n  Concept activations for {protein}:")
            for cname, cval in concept_state.items():
                bar = "█" * int(abs(cval) * 20)
                sign = "+" if cval >= 0 else "-"
                print(f"    {cname:<28} {sign}{abs(cval):.3f}  {bar}")

    # 5. Results table
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv("debug_predict_results.csv", index=False)

    print("\n" + "=" * 60)
    print("PREDICTED vs ACTUAL VISCOSITY (cP)")
    print("=" * 60)
    for _, row in final_results.iterrows():
        print(f"\nSample ID: {row['ID']} | Protein: {row['Protein_type']}")
        print(f"{'Shear Rate':>12} | {'Actual cP':>10} | {'Pred cP':>10} | {'% Error':>10}")
        print("-" * 52)
        for shear in shear_cols:
            actual_col = f"Viscosity_{shear}"
            pred_col = f"Pred_Viscosity_{shear}"
            actual_val = row.get(actual_col, np.nan)
            pred_val = row.get(pred_col, np.nan)
            if pd.notna(actual_val) and pd.notna(pred_val) and actual_val > 0:
                error = abs(pred_val - actual_val) / actual_val * 100
                print(f"{shear:>12} | {actual_val:10.2f} | {pred_val:10.2f} | {error:9.1f}%")
            else:
                print(f"{shear:>12} | {'N/A':>10} | {pred_val:10.2f} | {'N/A':>10}")

    # 6. Uncertainty
    print("\n" + "=" * 60)
    print("UNCERTAINTY ESTIMATES (MC Dropout, 95% CI)")
    print("=" * 60)
    for protein in target_df["Protein_type"].unique():
        prot_target = target_df[target_df["Protein_type"] == protein].copy()
        target_ids = prot_target["ID"].tolist()
        history_df = full_train_df[
            (full_train_df["Protein_type"] == protein) & (~full_train_df["ID"].isin(target_ids))
        ].copy()

        predictor.memory_vector = None
        predictor.context_t = None

        if not history_df.empty:
            predictor.learn(history_df)
            mean_pred, stats = predictor.predict_with_uncertainty(prot_target, n_samples=100)
            pred_ids = prot_target["ID"].tolist()
            n_shears = len(predictor.shear_map)

            print(f"\n  {protein}")
            for i, sid in enumerate(pred_ids):
                print(f"    Sample {sid}:")
                for j, shear in enumerate(shear_cols):
                    q_idx = i * n_shears + j
                    if q_idx < len(mean_pred):
                        lo = stats["lower_ci"][q_idx]
                        hi = stats["upper_ci"][q_idx]
                        sl = stats["std_log10"][q_idx]
                        print(
                            f"      {shear:>12} s⁻¹ | "
                            f"mean={mean_pred[q_idx]:7.2f} cP | "
                            f"95% CI [{lo:7.2f}, {hi:7.2f}] | "
                            f"sigma={sl:.3f} log₁₀"
                        )
        else:
            print(f"\n  {protein}: No context — zero-shot, uncertainty not available.")

    # 7. Concept intervention demo
    print("\n" + "=" * 60)
    print("CONCEPT INTERVENTION DEMO")
    print("=" * 60)

    demo_protein = target_df["Protein_type"].iloc[0]
    demo_target = target_df[target_df["Protein_type"] == demo_protein].copy()
    demo_history = full_train_df[
        (full_train_df["Protein_type"] == demo_protein)
        & (~full_train_df["ID"].isin(demo_target["ID"].tolist()))
    ].copy()

    predictor.memory_vector = None
    predictor.context_t = None
    if not demo_history.empty:
        predictor.learn(demo_history)

    concept_names = predictor.model.concept_names
    concept_acts = predictor.model._concept_activations
    baseline_df = predictor.predict(demo_target.head(1))
    baseline_100 = float(baseline_df["Pred_Viscosity_100"].iloc[0])

    print(f"\n  Protein: {demo_protein}  |  Baseline η(100 s⁻¹) = {baseline_100:.2f} cP")
    print(f"\n  {'Concept':<28} {'Act':>7} {'-> η(100) cP':>13} {'Δ cP':>10}")
    print("  " + "-" * 62)

    for ci, (cname, act_type) in enumerate(zip(concept_names, concept_acts)):
        sweep_val = 1.0 if act_type == "sigmoid" else -1.0
        int_df = predictor.intervene(demo_target.head(1), concept_idx=ci, concept_value=sweep_val)
        int_val = float(int_df["Pred_Viscosity_100"].iloc[0])
        delta = int_val - baseline_100
        sign = "▲" if delta > 0 else "▼"
        print(f"  {cname:<28} {act_type:>7} {int_val:>13.2f} {sign}{abs(delta):>9.2f}")
