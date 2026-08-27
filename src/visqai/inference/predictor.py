"""
ViscosityPredictorCNP: inference, few-shot adaptation, and uncertainty estimation
for the Cross-Sample Conditional Neural Process viscosity model.

The predictor loads a trained model checkpoint together with the fitted feature
preprocessor and physics scaler, preprocesses formulation-level inputs through
the shared feature-engineering pipeline, and produces viscosity predictions at
the configured shear rates.

Few-shot adaptation uses a query-conditioned local residual corrector rather
than updating model weights. The deployed `linear` corrector estimates a
shrunk formulation-level residual offset and concentration-dependent term,
followed by a separately shrunk shear-dependent residual slope. A
leave-one-formulation-out transfer check prevents correction when the observed
context does not demonstrate within-context generalization. An optional
`kernel` corrector provides a research alternative that estimates local
residuals from Gaussian similarity in a restricted physicochemical feature
space.

The predictor maintains the legacy neural context representation solely for
MC-dropout uncertainty estimation. Point predictions do not depend on the
neural memory representation; instead, they use the exact zero-shot decoder
path as their prior and add the validated local residual correction.

Descriptor out-of-distribution protection is applied after preprocessing by
clipping scaled numeric features to a configured standard-deviation range.
The same policy is expected to be applied during model training so that
training and inference feature handling remain symmetric.

Notes:
    The fitted `preprocessor.pkl` may have been serialized with a different
    scikit-learn version. A compatibility implementation of the removed
    `_RemainderColsList` wrapper is installed before loading the artifact
    when necessary.

    The empirical-Bayes constants and correction thresholds in this module
    are calibrated model artifacts rather than general-purpose hyperparameters.
    They should be recalibrated when the training data, feature representation,
    or zero-shot model architecture changes materially.
"""

from __future__ import annotations

import copy
import os
import logging

import joblib
import numpy as np
import pandas as pd
import torch

from visqai.constants import DESCRIPTOR_OOD_CLIP_SIGMA, SHEAR_MAP
from visqai.models.cnp import CrossSampleCNP
from visqai.features.dataprocessor import build_feature_frame

logger = logging.getLogger(__name__)

# preprocessor.pkl was fit under scikit-learn ~1.5.x, whose ColumnTransformer
# stores its "remainder" columns in `_RemainderColsList` (a UserList subclass
# that only exists to emit a one-time FutureWarning on access -- see
# sklearn.compose._column_transformer). That class was removed in later
# sklearn releases, so joblib.load(preprocessor.pkl) on a newer/older sklearn
# install (e.g. this repo's `pytorch` conda env, currently 1.7.2) fails with
# AttributeError: Can't get attribute '_RemainderColsList'. Since the class is
# purely a cosmetic warning wrapper around a plain list, restoring it as a
# UserList subclass is a functionally identical stand-in for unpickling.
try:
    import sklearn.compose._column_transformer as _sklearn_ct
    from collections import UserList as _UserList

    if not hasattr(_sklearn_ct, "_RemainderColsList"):

        class _RemainderColsList(_UserList):
            def __init__(
                self,
                columns=(),
                *,
                future_dtype=None,
                warning_was_emitted=False,
                warning_enabled=True,
            ):
                super().__init__(columns)
                self.future_dtype = future_dtype
                self.warning_was_emitted = warning_was_emitted
                self.warning_enabled = warning_enabled

        _sklearn_ct._RemainderColsList = _RemainderColsList
except ImportError:
    pass

# Empirical-Bayes shrinkage constants for the delta corrector's offset_hat
# (T-R3.2/T-R3.6/T-R3.7), calibrated ONCE from models/experiments/
# rung1_context_learnability/residuals.csv via the same one-way-ANOVA
# decomposition analyze.py's T2 uses, but on PER-FORMULATION MEAN residuals
# (one number per context row) rather than raw points -- see module
# docstring's "FORMULATION-LEVEL VARIANCE" section for why: raw points from
# one formulation are correlated, not independent samples, and pooling them
# understated offset_hat's true uncertainty. SIGMA2_WITHIN = within-protein,
# across-formulation MSW; TAU2_BETWEEN = (MSB-MSW)/n0, between-protein
# variance of true offsets (Fisher's unbalanced-design correction).
# Recalibrate from a fresh Rung 1 run if the training data or prior
# architecture changes enough to shift these substantially.
SIGMA2_WITHIN: float = 0.031106
TAU2_BETWEEN: float = 0.034323

# T-R3.7: minimum number of distinct context FORMULATIONS (not points)
# before offset_hat/slope_hat can be nonzero. A single formulation cannot
# distinguish "this protein has a real offset" from "this one formulation
# is atypical" -- see module docstring.
MIN_CONTEXT_FORMULATIONS: int = 2

# T-R3.8: minimum fraction of within-context leave-one-formulation-out folds
# that must show improvement before offset_hat/slope_hat fire at all --
# REPLACES T-R3.6's magnitude/SE confidence gate (see module docstring's
# "WITHIN-CONTEXT TRANSFER CHECK" section for why: a magnitude test can't
# see that an offset is real but formulation-unstable, which is exactly
# nivolumab's failure mode). Calibrated via offline simulation against
# Rung 1's residual table -- 2/3 was the smallest (most permissive) fraction
# that still cleared the -0.005 worst-k8-lift bar.
TRANSFER_CHECK_FRAC: float = 2.0 / 3.0

# Empirical-Bayes shrinkage constant for the delta corrector's slope_hat
# (T-R3.4): variance of the per-protein OLS slopes fit in Rung 1's T3
# (t3_offset_slope.csv), i.e. the population prior on "how steep can a
# genuine per-protein shear-dependent bias be."
TAU2_SLOPE: float = 0.000735

# Empirical-Bayes shrinkage constant for the query-conditioned corrector's
# conc_hat (Task 1.1): population prior on "how steep can a genuine
# per-protein Protein_conc-dependent bias be." Calibrated the same way as
# TAU2_SLOPE -- population variance (ddof=1) of per-protein OLS slopes of
# (per-formulation-mean resid) vs. Protein_conc, computed from Rung 1's
# leave-one-protein-out zero-shot residual table
# (models/experiments/rung1_context_learnability/residuals.csv) merged with
# the raw dataset's Protein_conc column, restricted to proteins with >=4
# formulations and >=3 distinct concentrations (12/12 proteins qualified):
# slope mean=+0.00041, var(ddof=1)=5.4728e-06. See module docstring's
# "QUERY-CONDITIONED LOCAL RESIDUAL" section for why this is fit on
# per-formulation means (unlike TAU2_SLOPE, which is fit on raw points)
# and why this fold is disjoint from the condition-shift acceptance eval.
TAU2_CONC: float = 5.473e-06

# Descriptor-OOD down-weighting (prior-side, not corrector work): caps every
# SCALED numeric feature to +/- DESCRIPTOR_OOD_CLIP_SIGMA (visqai.constants;
# shared, single-sourced with visqai.training.data's identical use of it)
# standard deviations before it reaches the network. StandardScaler's output
# IS already in standard-deviation units, so this is a plain np.clip -- no
# extra stats needed. Matches visqai.eval.logo_eval.FOLD_RANGE_N_SIGMA (5.0),
# the existing diagnostic threshold that LOGS a held-out group's OOD
# descriptor values without acting on them; this constant is what turns that
# diagnostic into an actual behavior change. A held-out group whose real
# feature value sits past 5 sigma of what the training fold ever represented
# gets capped at the fold's own edge instead of injecting a raw, unbounded
# activation the network never learned to handle -- see PREDICTOR docstring
# section "DESCRIPTOR-OOD DOWN-WEIGHTING".

# Task 1.2: candidate Gaussian kernel bandwidths (in SCALED-feature units --
# same standard-deviation units DESCRIPTOR_OOD_CLIP_SIGMA already clips to)
# the kernel corrector picks from via leave-one-formulation-out on the
# CONTEXT itself (ground rule #1 -- never the acceptance eval). A log-spaced
# grid spanning "very local" (0.25 sigma -- only near-duplicate formulations
# contribute) to "nearly flat/global" (4 sigma -- close to an unweighted
# mean, similar to Task 1.1's offset_hat).
#
# SHIP DECISION (Task A.2/A.3 addendum, issue1_query_conditioned_correction_
# plan.md): corrector_mode defaults to "linear" (Task 1.1 + the Task A.3
# context-support clamp, see _fit_local_residual/predict) -- that is the
# DEPLOYED corrector. "kernel" is a non-default RESEARCH BRANCH, not shipped:
# a three-way ablation against corrector_mode="offset_only" (the pre-Task-
# 1.1 corrector) on the bad-prior stratum -- the stratum with a measurable
# effect; the good-prior stratum's differences are all inside this eval's
# ~0.017 log-MAE minimum detectable effect at n=11 protein clusters, see
# the (now-removed) condition-shift eval's module docstring -- showed post-A.3 kernel
# captures only ~27% of linear's bad-prior gain (offset-only +0.054, linear
# +0.075, kernel +0.020) in exchange for a good-prior "gain" (+0.008) that
# is not distinguishable from noise at this sample size. Never pick a
# corrector based on the good-prior number alone -- that stratum currently
# has no statistical power to tell correctors apart.
KERNEL_BANDWIDTH_CANDIDATES: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)


class ViscosityPredictorCNP:
    def __init__(self, model_dir: str, verbose: bool = False) -> None:
        """Initialize a trained CNP viscosity predictor.

        Loads the fitted preprocessing artifacts and model checkpoint from
        `model_dir`, restores the model to its pristine evaluation state, and
        initializes the state used by zero-shot prediction, few-shot residual
        correction, and uncertainty estimation.

        The predictor automatically selects CUDA when available and otherwise
        falls back to CPU. No model weights are modified during initialization;
        a deep copy of the loaded state is retained so that subsequent calls to
        :meth:`learn` can restore the original model before adapting to a new
        context.

        Args:
            model_dir: Directory containing the fitted preprocessing artifacts
                (`preprocessor.pkl` and `physics_scaler.pkl`) and the trained
                model checkpoint (`best_model.pth`).
            verbose: Whether to enable informational logging for this predictor
                instance. When `False`, the instance logger is restricted to
                critical messages.

        Raises:
            FileNotFoundError: If the preprocessor, physics scaler, or model
                checkpoint is missing from `model_dir`.
            RuntimeError: If the checkpoint cannot be loaded or its state
                dictionary is incompatible with the reconstructed model.
            KeyError: If the checkpoint does not contain the expected `config`
                or `static_dim` metadata.
        """
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")
        if not verbose:
            self._logger.setLevel(logging.CRITICAL)
        self._logger.info(f"Initializing ViscosityPredictorCNP with model_dir: {model_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self._memory_vector = None  # Stores the calibrated context
        self.offset_hat = 0.0  # Delta corrector's shrunk offset
        self.conc_hat = 0.0  # Query-conditioned corrector's shrunk Protein_conc coefficient
        self.conc_center = 0.0  # Protein_conc pivot conc_hat is fit around
        self.conc_support_min = 0.0  # Context's own min Protein_conc
        self.conc_support_max = 0.0  # Context's own max Protein_conc
        self.slope_hat = 0.0  # Delta corrector's shrunk shear-slope
        self.slope_center = 0.0  # log_shear pivot point slope_hat is fit around

        # Real measured points offset_hat/slope_hat were estimated from
        self.n_context_points = 0

        # Kernel-weighted local residual correction. This provides an
        # alternative to the deployed linear offset-plus-concentration correction
        # for feature relationships that are not well represented by a linear term,
        # such as additive or buffer identity effects. The behavior is selected by
        # `corrector_mode`: "linear" is the default deployed path, while "kernel"
        # enables the experimental kernel-based correction described in the module
        # documentation.
        self.corrector_mode = "linear"
        self._kernel_ctx_phi = (
            None  # (n_formulations, n_kernel_features) context similarity vectors
        )
        self._kernel_ctx_resid = None  # (n_formulations,) per-formulation mean residuals
        self.kernel_bandwidth = None  # chosen via LOO on context (_fit_local_residual_kernel)
        self._kernel_feat_idx = None  # cached column indices into the preprocessor's numeric block

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
        self._original_state = copy.deepcopy(self.model.state_dict())
        self.context_t: torch.Tensor | None = None
        self.shear_map = dict(SHEAR_MAP)

    @property
    def memory_vector(self) -> torch.Tensor | None:
        """Return the encoded context memory used by the legacy neural path.

        The memory vector is populated by :meth:`learn` from the current context
        and is retained primarily for MC-dropout uncertainty estimation. It is
        not used by the main point-prediction path, which applies the zero-shot
        decoder prior together with the local residual correction.

        Returns:
            The encoded context memory tensor, or `None` when no context has
            been learned or the predictor has been reset to zero-shot operation.
        """
        return self._memory_vector

    @memory_vector.setter
    def memory_vector(self, value: torch.Tensor | None) -> None:
        """Set the encoded context memory and synchronize zero-shot state.

        Assigning `None` is the established mechanism for resetting the
        predictor to zero-shot operation. In addition to clearing the legacy
        neural memory, this setter clears all few-shot residual-correction state,
        including the concentration correction, shear-dependent correction, and
        kernel-correction context. This keeps existing callers that reset only
        `memory_vector` from retaining stale few-shot corrections.

        Args:
            value: Encoded context memory tensor to retain for subsequent
                uncertainty estimation, or `None` to clear the learned context
                and restore zero-shot correction state.
        """
        self._memory_vector = value
        if value is None:
            self.offset_hat = 0.0
            self.conc_hat = 0.0
            self.conc_center = 0.0
            self.conc_support_min = 0.0
            self.conc_support_max = 0.0
            self.slope_hat = 0.0
            self.slope_center = 0.0
            self.n_context_points = 0
            self._kernel_ctx_phi = None
            self._kernel_ctx_resid = None
            self.kernel_bandwidth = None

    def _clip_descriptor_ood(self, x_static: np.ndarray) -> np.ndarray:
        """Clip scaled numeric descriptors to the configured OOD boundary.

        The numeric portion of the preprocessed feature matrix is produced by a
        standard scaler, so each value is expressed in standard-deviation units.
        Values outside `±DESCRIPTOR_OOD_CLIP_SIGMA` are therefore clipped to
        the nearest boundary before being passed to the model. This limits the
        effect of descriptor values that are substantially outside the training
        distribution while preserving the preprocessor's categorical one-hot
        features unchanged.

        Args:
            x_static: Preprocessed static feature matrix whose leading columns
                correspond to the numeric feature block.

        Returns:
            The same feature matrix with scaled numeric features clipped to the
            configured OOD range. The input array is modified in place.
        """
        n_num = len(self.preprocessor.transformers_[0][2])
        x_static[:, :n_num] = np.clip(
            x_static[:, :n_num], -DESCRIPTOR_OOD_CLIP_SIGMA, DESCRIPTOR_OOD_CLIP_SIGMA
        )
        return x_static

    def _preprocess(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        df_proc = df.copy()

        for col in df_proc.select_dtypes(include=["object"]):
            df_proc[col] = df_proc[col].apply(lambda x: x.value if hasattr(x, "value") else x)

        if "ID" in df_proc.columns:
            df_proc = df_proc.drop(columns=["ID"])

        df_proc, _num_cols, _cat_cols = build_feature_frame(df_proc)
        return self._preprocess_built(df_proc)

    def _preprocess_built(
        self, df_proc: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess an already feature-engineered formulation DataFrame.

        Completes the preprocessing pipeline after :func:`build_feature_frame` by
        filling missing expected features, applying the fitted static-feature
        preprocessor, replacing invalid numeric values, clipping descriptor
        out-of-distribution values, and constructing the scaled shear-rate and
        viscosity tensors expected by the CNP model.

        This method is intentionally separate from :meth:`_preprocess` so callers
        that have deliberately modified engineered features, such as permutation
        feature-importance evaluations, can bypass feature reconstruction without
        having their modifications overwritten by a second call to
        :func:`build_feature_frame`.

        Args:
            df_proc: DataFrame that has already passed through
                :func:`build_feature_frame`. Its engineered feature values are used
                as provided and are not recomputed.

        Returns:
            A tuple containing:
                static_t: Static formulation features as a batched float32 tensor.
                shear_t: Scaled log10 shear-rate values as a batched float32 tensor.
                visc_t: Scaled log10 viscosity values as a batched float32 tensor.
        """
        feature_names = (
            self.preprocessor.feature_names_in_
            if hasattr(self.preprocessor, "feature_names_in_")
            else []
        )
        expected_missing = ["ID"] + list(self.shear_map.keys())
        missing_feats = [c for c in feature_names if c not in df_proc.columns]
        actual_missing = [c for c in missing_feats if c not in expected_missing]
        if actual_missing:
            self._logger.warning(f"Missing static features filled with 0.0: {actual_missing}")
        for c in missing_feats:
            df_proc[c] = 0.0

        x_static = self.preprocessor.transform(df_proc)
        if np.isnan(x_static).any():
            self._logger.warning("NaNs found in X_static after preprocessing! Replacing with 0.")
            x_static = np.nan_to_num(x_static)
        x_static = self._clip_descriptor_ood(x_static)

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
                static_list.append(x_static[i])
                row_idx += 1

        scaled_points = self.physics_scaler.transform(raw_points)

        static_t = (
            torch.tensor(np.array(static_list), dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        points_t = torch.tensor(scaled_points.astype(np.float32)).unsqueeze(0).to(self.device)

        shear_t = points_t[:, :, [0]]
        visc_t = points_t[:, :, [1]]
        return static_t, shear_t, visc_t

    def learn(
        self,
        df: pd.DataFrame,
        steps: int = 50,  # kept for API compatibility — no longer used
        lr: float = 1e-3,  # kept for API compatibility — no longer used
        n_draws: int = 20,
        k: int = 8,
    ) -> None:
        """Adapt the predictor to a new protein context without updating weights.

        The context is first encoded into the legacy neural memory representation,
        which is retained for :meth:`predict_with_uncertainty`. The primary
        few-shot point-prediction path instead estimates a local residual correction
        from the context, consisting of a formulation-level offset, an optional
        concentration-dependent term, and a shear-dependent slope. These corrections
        are fit using the configured transfer checks and empirical-Bayes shrinkage.

        When `corrector_mode` is `"kernel"`, an additional kernel-weighted local
        residual model is fitted from the context for use by :meth:`predict`. The
        kernel path is gated by within-context transfer performance and is otherwise
        left inactive. The default `"linear"` mode uses the query-conditioned
        linear residual corrector.

        Repeated calls restore the model to its original checkpoint state before
        encoding the new context, preventing context-specific state from
        contaminating subsequent learning calls.

        Args:
            df: DataFrame containing measured formulation samples for the protein
                context. Each row represents a formulation and may contain measured
                viscosity values at one or more configured shear rates.
            steps: Retained for API compatibility with earlier implementations that
                performed iterative adaptation. It is no longer used because learning
                does not update model weights.
            lr: Retained for API compatibility with earlier implementations. It is
                no longer used because learning does not perform gradient-based
                optimization.
            n_draws: Number of random context subsets used to estimate the legacy
                neural memory representation when the context contains more than
                `k` available points.
            k: Maximum number of context points sampled for each neural memory
                encoding draw.

        Returns:
            None. The predictor is updated in place with the encoded context and
            fitted few-shot correction state.

        Notes:
            If `df` is empty, learning is skipped and the existing predictor state
            is left unchanged.
        """
        if df.empty:
            self._logger.warning("Context DataFrame is empty. Skipping learning.")
            return

        self._logger.info(
            f" > Learn triggered on {len(df)} samples (n_draws={n_draws}, k={k}, no weight updates)"
        )

        self.model.load_state_dict(self._original_state)

        static_t, shear_t, visc_t = self._preprocess(df)
        context_t = torch.cat([shear_t, visc_t, static_t], dim=-1)
        self.context_t = context_t

        n_ctx = context_t.size(1)
        k_eff = min(k, n_ctx)

        # Legacy neural memory
        self.model.eval()
        with torch.no_grad():
            if n_ctx <= k_eff:
                self._memory_vector = self.model.encode_memory(context_t)
            else:
                memory_draws = []
                for _ in range(n_draws):
                    idx = torch.randperm(n_ctx, device=self.device)[:k_eff]
                    subset = context_t[:, idx, :]
                    r = self.model.encode_memory(subset)
                    memory_draws.append(r)
                self._memory_vector = torch.stack(memory_draws, dim=0).mean(dim=0)

        # Few-shot residual correction: the primary adaptation mechanism. The
        # formulation-level offset, concentration-dependent term, and shear-dependent
        # slope are estimated from context residuals and empirically shrunk to
        # distinguish transferable protein-level signal from formulation-level noise.
        ctx_formulations, ctx_shear, ctx_resid, ctx_conc, ctx_static = self._context_residuals(df)

        # Ablation mode: "offset_only" provides a controlled baseline for evaluating
        # whether the newer local-correction components provide measurable benefit
        # over the preceding correction strategy. The conditioning variable is
        # neutralized before fitting, which makes its coefficient unidentifiable and
        # therefore leaves that component exactly zero while preserving the shared
        # offset/slope fitting path. Reusing the production fitting implementation
        # ensures the baseline remains definitionally aligned with the corresponding
        # legacy correction rather than maintaining a separate implementation that
        # could drift over time.
        ctx_conc_for_fit = (
            ctx_conc if self.corrector_mode != "offset_only" else np.zeros_like(ctx_conc)
        )
        (
            self.offset_hat,
            self.conc_hat,
            self.conc_center,
            self.slope_hat,
            self.slope_center,
        ) = self._fit_local_residual(ctx_formulations, ctx_shear, ctx_resid, ctx_conc_for_fit)
        self.n_context_points = len(ctx_resid)

        # Constrain concentration-dependent correction to the range represented by
        # the context. Query concentrations are clipped to the observed context
        # interval before evaluating the fitted concentration term, preventing a
        # short-range or noisy trend estimate from producing unbounded extrapolation
        # outside the evidence used to fit it. When the concentration coefficient is
        # zero, this clipping has no effect on the resulting prediction.
        if len(ctx_conc) > 0:
            self.conc_support_min = float(np.min(ctx_conc))
            self.conc_support_max = float(np.max(ctx_conc))
        else:
            self.conc_support_min = 0.0
            self.conc_support_max = 0.0

        # Optional kernel-based local residual correction. Provides an alternative
        # to the linear correction model for feature relationships that may be
        # nonlinear or localized, such as additive or buffer identity effects. The
        # kernel path is fitted only when explicitly selected via `corrector_mode`;
        # the default "linear" mode leaves this state inactive and ignores it during
        # prediction.
        self._kernel_ctx_phi = None
        self._kernel_ctx_resid = None
        self.kernel_bandwidth = None
        if self.corrector_mode == "kernel":
            ctx_phi, ctx_form_resid, bandwidth, gate_passed = self._fit_local_residual_kernel(
                ctx_formulations, ctx_resid, ctx_static
            )
            if gate_passed:
                self._kernel_ctx_phi = ctx_phi
                self._kernel_ctx_resid = ctx_form_resid
                self.kernel_bandwidth = bandwidth

    def _prior_log10(self, q_shear: torch.Tensor, q_static: torch.Tensor) -> np.ndarray:
        """Generate the zero-shot log10-viscosity prior for query inputs.

        Uses the predictor's current zero-shot decoder path with an explicitly zero
        latent memory vector. This ensures the prior used by the few-shot residual
        corrector is identical to the baseline prediction path rather than relying
        on a separate or isolated prior head.

        Args:
            q_shear: Batched, scaled log10 shear-rate tensor for the query points.
            q_static: Batched static formulation-feature tensor for the query points.

        Returns:
            NumPy array containing the predicted log10 viscosity values in the
            original, unscaled physical space.
        """
        zero_mem = torch.zeros((1, self.config["latent_dim"]), device=self.device)
        self.model.eval()
        with torch.no_grad():
            prior_scaled = self.model.decode_from_memory(zero_mem, q_shear, q_static)
        return self._inverse_to_log(q_shear, prior_scaled)

    def _context_residuals(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build residual-learning targets from the real measurements in a context set.

        For each observed formulation/shear-rate pair, computes the residual between
        the measured log10 viscosity and the model's zero-shot prior prediction.
        Missing viscosity measurements are excluded rather than replaced with the
        placeholder values used by query-side preprocessing.

        Args:
            df: Context DataFrame containing formulation descriptors and measured
                viscosity values at the shear rates defined by `self.shear_map`.

        Returns:
            A tuple containing:
                - `formulation_idx`: Integer formulation indices corresponding to
                  the original row positions in `df`. Each formulation index is
                  repeated for every valid measured shear point belonging to that
                  row.
                - `shear_logs`: Unscaled `log10` shear-rate values.
                - `residuals`: Measured minus zero-shot prior `log10` viscosity
                  values for each valid measurement.
                - `concs`: Unscaled `Protein_conc` values associated with each
                  measurement, or `0.0` when unavailable.
                - `statics`: Scaled static feature vectors corresponding to each
                  measurement. These contain the full preprocessed numeric and
                  categorical feature representation used by the model.

            If `df` is empty or contains no valid measured viscosity points, each
            returned array is empty.
        """
        if df.empty:
            return np.empty(0, dtype=int), np.empty(0), np.empty(0), np.empty(0), np.empty((0, 0))

        df_proc = df.copy()
        for col in df_proc.select_dtypes(include=["object"]):
            df_proc[col] = df_proc[col].apply(lambda x: x.value if hasattr(x, "value") else x)
        if "ID" in df_proc.columns:
            df_proc = df_proc.drop(columns=["ID"])
        df_proc, _num_cols, _cat_cols = build_feature_frame(df_proc)

        feature_names = (
            self.preprocessor.feature_names_in_
            if hasattr(self.preprocessor, "feature_names_in_")
            else []
        )
        for c in feature_names:
            if c not in df_proc.columns:
                df_proc[c] = 0.0
        x_static = self.preprocessor.transform(df_proc)
        if np.isnan(x_static).any():
            x_static = np.nan_to_num(x_static)
        x_static = self._clip_descriptor_ood(x_static)

        has_conc = "Protein_conc" in df_proc.columns
        formulation_idx, shear_logs, true_logs, statics, concs = [], [], [], [], []
        for i in range(len(df_proc)):
            conc_val = 0.0
            if has_conc and pd.notna(df_proc.iloc[i]["Protein_conc"]):
                conc_val = float(df_proc.iloc[i]["Protein_conc"])
            for col, shear_val in self.shear_map.items():
                if col not in df_proc.columns:
                    continue
                v = df_proc.iloc[i][col]
                if pd.isna(v) or v <= 0:
                    continue
                formulation_idx.append(i)
                shear_logs.append(np.log10(shear_val))
                true_logs.append(np.log10(float(v)))
                statics.append(x_static[i])
                concs.append(conc_val)

        if not shear_logs:
            return np.empty(0, dtype=int), np.empty(0), np.empty(0), np.empty(0), np.empty((0, 0))

        statics_arr = np.array(statics)
        raw_points = np.column_stack([shear_logs, true_logs])
        scaled_points = self.physics_scaler.transform(raw_points)
        shear_t = (
            torch.tensor(scaled_points[:, [0]], dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        static_t = torch.tensor(statics_arr, dtype=torch.float32).unsqueeze(0).to(self.device)

        prior_log = self._prior_log10(shear_t, static_t)
        return (
            np.array(formulation_idx, dtype=int),
            np.array(shear_logs),
            np.array(true_logs) - prior_log,
            np.array(concs),
            statics_arr,
        )

    def _fit_formulation_level(
        self, form_conc: np.ndarray, form_means: np.ndarray
    ) -> tuple[float, float, float]:
        """Fit a formulation-level residual correction using empirical-Bayes shrinkage.

        Fits the formulation means to an intercept plus a Protein_conc-centered
        linear term. The intercept and concentration coefficient are jointly
        regularized according to their respective between-formulation variance
        scales, providing a two-parameter generalization of scalar offset shrinkage.

        When the context contains fewer than two distinct concentration values, the
        concentration effect is not identifiable. In that case, the method falls back
        to the scalar formulation-level offset estimator and returns a zero
        concentration coefficient.

        Args:
            form_conc: One-dimensional array containing the Protein_conc value for
                each formulation. Values are centered internally around their context
                mean before fitting the concentration effect.
            form_means: One-dimensional array containing the mean residual for each
                formulation. Must have the same length as `form_conc`.

        Returns:
            A tuple containing:
                - `offset`: Shrunk formulation-level residual offset evaluated at
                  the context mean concentration.
                - `conc_coeff`: Shrunk linear residual coefficient with respect to
                  centered Protein_conc. Returns `0.0` when the context does not
                  contain sufficient concentration variation to identify the effect.
                - `conc_center`: Mean Protein_conc of the context, used as the
                  reference point for the fitted concentration effect.

        Notes:
            The regularization combines the estimated per-formulation residual
            variance with the configured between-formulation and concentration
            variance scales. With no identifiable concentration variation, the
            estimator reduces algebraically to the scalar shrinkage used by the
            formulation-level offset fallback.
        """
        k = len(form_means)
        cbar = float(np.mean(form_conc))
        sigma2 = max(float(np.var(form_means, ddof=1)), SIGMA2_WITHIN * 0.5)
        conc_centered = form_conc - cbar
        if np.ptp(conc_centered) < 1e-9:
            raw = float(np.mean(form_means))
            shrink = (k * TAU2_BETWEEN) / (k * TAU2_BETWEEN + sigma2)
            return raw * shrink, 0.0, cbar

        basis = np.column_stack([np.ones(k), conc_centered])
        lam = sigma2 * np.array([1.0 / TAU2_BETWEEN, 1.0 / TAU2_CONC])
        beta = np.linalg.solve(basis.T @ basis + np.diag(lam), basis.T @ form_means)
        return float(beta[0]), float(beta[1]), cbar

    def _fit_formulation_level_raw(
        self, form_conc: np.ndarray, form_means: np.ndarray
    ) -> tuple[float, float, float]:
        """Fit an unregularized formulation-level residual model.

        Computes the ordinary least-squares fit of formulation-level mean residuals
        to an intercept and a Protein_conc-centered linear term. This unshrunk
        estimator is intended for transfer-validation checks, where applying the
        production shrinkage would make leave-one-out estimates unnecessarily
        conservative and could obscure whether the learned correction transfers
        across formulations.

        If fewer than two formulations are available, or if the context contains no
        meaningful variation in Protein_conc, the concentration effect is treated as
        unidentifiable and the method falls back to the raw mean residual with a
        zero concentration coefficient.

        Args:
            form_conc: One-dimensional array containing the Protein_conc value for
                each formulation.
            form_means: One-dimensional array containing the mean residual for each
                formulation. Must correspond element-wise to `form_conc`.

        Returns:
            A tuple containing:
                - `offset`: Unshrunk residual offset evaluated at the mean context
                  concentration.
                - `conc_coeff`: Unshrunk linear residual coefficient with respect
                  to centered Protein_conc, or `0.0` when the concentration effect
                  is not identifiable.
                - `conc_center`: Mean Protein_conc of the formulations, used as the
                  reference point for the concentration term.
        """
        k = len(form_means)
        if k == 0:
            return 0.0, 0.0, 0.0
        cbar = float(np.mean(form_conc))
        conc_centered = form_conc - cbar
        if k < 2 or np.ptp(conc_centered) < 1e-9:
            return float(np.mean(form_means)), 0.0, cbar
        basis = np.column_stack([np.ones(k), conc_centered])
        beta, *_ = np.linalg.lstsq(basis, form_means, rcond=None)
        return float(beta[0]), float(beta[1]), cbar

    def _transfer_check_passes(
        self, ctx_formulations: np.ndarray, ctx_resid: np.ndarray, ctx_conc: np.ndarray
    ) -> bool:
        """Evaluate whether the learned local residual correction transfers across formulations.

        Performs a leave-one-formulation-out validation of the formulation-level
        residual model. For each held-out formulation, the model is fit on the
        remaining formulations using the unshrunk formulation-level estimator, then
        evaluated at the held-out formulation's own Protein_conc value. The learned
        correction is considered transferable when it reduces the held-out mean
        absolute residual relative to applying no correction for the required
        fraction of folds.

        This validation is performed without production shrinkage so that the
        transfer test measures whether the underlying local correction generalizes,
        rather than whether regularization suppresses it. When Protein_conc is
        constant across the context, the concentration coefficient is necessarily
        zero and the procedure reduces to the scalar offset leave-one-out check.

        Args:
            ctx_formulations: One-dimensional integer array identifying the
                formulation associated with each measured residual. Multiple
                measurements may share the same formulation identifier.
            ctx_resid: One-dimensional array of measured-minus-prior residuals,
                corresponding element-wise to `ctx_formulations`.
            ctx_conc: One-dimensional array of Protein_conc values corresponding to
                the residual measurements. Values are aggregated to the
                formulation level before fitting and evaluating the correction.

        Returns:
            `True` if the learned formulation-level correction reduces mean
            absolute residual for at least `TRANSFER_CHECK_FRAC` of the
            leave-one-formulation-out folds; otherwise `False`.
        """
        unique_forms = np.unique(ctx_formulations)
        k = len(unique_forms)
        form_means_all = pd.Series(ctx_resid).groupby(ctx_formulations).mean()
        form_conc_all = pd.Series(ctx_conc).groupby(ctx_formulations).mean()
        successes = 0
        for held_back in unique_forms:
            train_forms = [f for f in unique_forms if f != held_back]
            if not train_forms:
                continue
            train_means = form_means_all.loc[train_forms].values
            train_conc = form_conc_all.loc[train_forms].values
            intercept, conc_coeff, cbar = self._fit_formulation_level_raw(train_conc, train_means)
            held_conc = float(form_conc_all.loc[held_back])
            held_pred = intercept + conc_coeff * (held_conc - cbar)
            test_mask = ctx_formulations == held_back
            test_resid = ctx_resid[test_mask]
            raw_err = float(np.mean(np.abs(test_resid)))
            corrected_err = float(np.mean(np.abs(test_resid - held_pred)))
            if corrected_err < raw_err:
                successes += 1
        return (successes / k) >= TRANSFER_CHECK_FRAC

    def _fit_local_residual(
        self,
        ctx_formulations: np.ndarray,
        ctx_shear: np.ndarray,
        ctx_resid: np.ndarray,
        ctx_conc: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        """Fit the local residual correction model from context measurements.

        Builds a query-conditioned residual model consisting of a formulation-level
        offset, a Protein_conc-dependent correction, and a shear-dependent slope.
        The formulation-level terms are estimated first using empirical-Bayes
        shrinkage, after which the shear slope is estimated from residuals remaining
        after the full formulation-level correction has been removed.

        The fit is accepted only when the context contains the minimum required
        number of formulations and the leave-one-formulation-out transfer check
        demonstrates sufficient out-of-formulation predictive benefit. This prevents
        unstable or non-transferable context corrections from being applied.

        The formulation-level correction is evaluated relative to the context's
        mean Protein_conc, while the shear term is centered at the context's mean
        log10 shear rate. The resulting parameters therefore represent corrections
        around the empirical center of the context rather than an arbitrary origin.

        For compatibility with the pre-concentration-conditioned correction, the
        method has a hard fallback: when the context does not contain sufficient
        concentration variation, the concentration coefficient is zero and the
        offset/slope estimation reduces to the corresponding scalar
        offset-plus-slope procedure.

        Args:
            ctx_formulations: One-dimensional array identifying the formulation
                associated with each measured residual. Multiple shear measurements
                may share the same formulation identifier.
            ctx_shear: One-dimensional array of unscaled `log10` shear-rate values
                corresponding to each measured residual.
            ctx_resid: One-dimensional array of measured-minus-prior
                `log10`-viscosity residuals.
            ctx_conc: One-dimensional array of unscaled Protein_conc values
                corresponding to each residual measurement.

        Returns:
            A tuple containing:
                - `offset_hat`: Shrunk formulation-level residual offset evaluated
                  at `conc_center`.
                - `conc_hat`: Shrunk residual coefficient for centered
                  Protein_conc.
                - `conc_center`: Mean Protein_conc of the context used as the
                  concentration pivot.
                - `slope_hat`: Shrunk residual slope with respect to centered
                  log10 shear rate.
                - `slope_center`: Mean log10 shear rate of the context used as the
                  shear pivot.

            If the context is empty, contains too few formulations, or fails the
            transfer check, all returned parameters are zero.
        """
        if len(ctx_resid) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        form_means = pd.Series(ctx_resid).groupby(ctx_formulations).mean().values
        if len(form_means) < MIN_CONTEXT_FORMULATIONS:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        if not self._transfer_check_passes(ctx_formulations, ctx_resid, ctx_conc):
            return 0.0, 0.0, 0.0, 0.0, 0.0

        form_conc = pd.Series(ctx_conc).groupby(ctx_formulations).mean().values
        offset_hat, conc_hat, conc_center = self._fit_formulation_level(form_conc, form_means)

        if len(np.unique(ctx_shear)) < 2:
            return offset_hat, conc_hat, conc_center, 0.0, 0.0

        n = len(ctx_shear)
        xbar = float(np.mean(ctx_shear))
        form_correction = offset_hat + conc_hat * (ctx_conc - conc_center)
        resid_after_form = ctx_resid - form_correction
        sxx = float(np.sum((ctx_shear - xbar) ** 2))
        slope_raw = float(np.sum((ctx_shear - xbar) * resid_after_form) / sxx)

        if n > 2:
            fitted = slope_raw * (ctx_shear - xbar)
            sigma2_slope = max(
                float(np.sum((resid_after_form - fitted) ** 2) / (n - 2) / sxx), TAU2_SLOPE * 0.1
            )
        else:
            sigma2_slope = SIGMA2_WITHIN / sxx

        shrink = TAU2_SLOPE / (TAU2_SLOPE + sigma2_slope)
        return offset_hat, conc_hat, conc_center, slope_raw * shrink, xbar

    def _kernel_feature_indices(self) -> list[int]:
        """Return the preprocessed numeric feature indices used by the kernel metric.

        Identifies the physically meaningful subset of the preprocessor's scaled
        numeric feature block for use by the kernel-based local residual corrector.
        The selected features include `Protein_conc` and the ingredient/buffer
        physicochemical property columns defined by the feature-processing pipeline.

        Restricting the similarity metric to these features prevents unrelated
        numeric descriptors, engineered interactions, or other dimensions from
        diluting the formulation similarity used by the kernel corrector.

        The resulting indices are cached after the first call because the
        preprocessor's feature layout is fixed for the lifetime of the predictor.

        Returns:
            A list of integer column indices into the preprocessor's leading scaled
            numeric feature block corresponding to the features used by the kernel
            similarity metric.
        """
        if self._kernel_feat_idx is None:
            from visqai.features.dataprocessor import _all_property_columns

            num_cols = list(self.preprocessor.transformers_[0][2])
            wanted = {"Protein_conc"} | _all_property_columns()
            self._kernel_feat_idx = [i for i, c in enumerate(num_cols) if c in wanted]
        return self._kernel_feat_idx

    def _kernel_loo_scan(
        self, form_phi: np.ndarray, form_resid: np.ndarray, ell: float
    ) -> tuple[float, float]:
        """Evaluate a Gaussian-kernel residual corrector using leave-one-formulation-out validation.

        For each formulation in the context, predicts its mean residual from the
        remaining formulations using a Gaussian kernel over their similarity
        features. The held-out formulation is never included in its own prediction,
        providing an out-of-formulation estimate of how well the selected bandwidth
        transfers to unseen context formulations.

        The leave-one-out predictions are intentionally unshrunk so that bandwidth
        selection and transfer gating measure the underlying local similarity signal
        rather than the effect of regularization. The resulting error and improvement
        fraction can be used both to select the kernel bandwidth and to determine
        whether the kernel corrector provides sufficient transfer benefit.

        Args:
            form_phi: Two-dimensional array of formulation-level similarity feature
                vectors. Each row represents one formulation in the context and must
                correspond to the same row in `form_resid`.
            form_resid: One-dimensional array of formulation-level mean residuals
                associated with `form_phi`.
            ell: Positive Gaussian kernel bandwidth controlling how rapidly similarity
                weights decrease with feature-space distance.

        Returns:
            A tuple containing:
                - `mean_abs_loo_error`: Mean absolute residual error across the
                  leave-one-formulation-out predictions.
                - `frac_improved`: Fraction of held-out formulations for which the
                  kernel correction reduces absolute residual error relative to
                  applying no correction.
        """
        k = len(form_resid)
        abs_errs = []
        n_improved = 0
        for i in range(k):
            mask = np.arange(k) != i
            d2 = np.sum((form_phi[mask] - form_phi[i]) ** 2, axis=1)
            w = np.exp(-d2 / (2.0 * ell**2))
            wsum = float(w.sum())
            pred = float(np.sum(w * form_resid[mask]) / wsum) if wsum > 1e-12 else 0.0
            raw_err = abs(float(form_resid[i]))
            corrected_err = abs(float(form_resid[i]) - pred)
            abs_errs.append(corrected_err)
            if corrected_err < raw_err:
                n_improved += 1
        return float(np.mean(abs_errs)), n_improved / k

    def _fit_local_residual_kernel(
        self, ctx_formulations: np.ndarray, ctx_resid: np.ndarray, ctx_static: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None, bool]:
        """Fit and validate a formulation-level Gaussian-kernel residual corrector.

        Constructs a local residual model that predicts a query formulation's
        correction from context formulations with similar physically meaningful
        descriptors. Residuals are first collapsed to one mean value per formulation
        to prevent multiple shear-rate measurements from causing pseudoreplication.
        Similarity is computed only over the feature subset selected by
        `_kernel_feature_indices`.

        The kernel bandwidth is selected using leave-one-formulation-out validation
        within the context. This validation serves both as bandwidth selection and
        as the transfer gate: the kernel corrector is enabled only when a candidate
        bandwidth reduces absolute residual error for at least
        `TRANSFER_CHECK_FRAC` of the held-out formulations. The acceptance test
        therefore uses only context data and does not consume an external evaluation
        set.

        Bandwidth candidates are evaluated in ascending order, and the first
        candidate that satisfies the transfer criterion is selected. This favors the
        most localized correction supported by the context rather than selecting the
        bandwidth with the lowest raw leave-one-out error, reducing the risk of
        overly broad corrections being applied outside the region represented by the
        context.

        Args:
            ctx_formulations: One-dimensional array identifying the formulation
                associated with each measured residual. Multiple measurements may
                share the same formulation identifier.
            ctx_resid: One-dimensional array of residuals between measured and
                zero-shot predicted log10 viscosity values, corresponding
                element-wise to `ctx_formulations`.
            ctx_static: Two-dimensional array of scaled static feature vectors,
                corresponding to the context measurements. The physically meaningful
                subset used for kernel similarity is selected internally.

        Returns:
            A tuple containing:
                - `form_phi`: Formulation-level similarity feature vectors, with
                  exactly one row per unique context formulation.
                - `form_resid`: Mean residual for each formulation in `form_phi`.
                - `bandwidth`: Selected Gaussian kernel bandwidth, or `None` when
                  the transfer gate fails or too few formulations are available.
                - `gate_passed`: `True` when a candidate bandwidth satisfies the
                  required leave-one-formulation-out transfer criterion; otherwise
                  `False`.

            When the context does not contain the minimum required number of
            formulations, all returned model arrays and the bandwidth are `None`
            and `gate_passed` is `False`.
        """
        form_means = pd.Series(ctx_resid).groupby(ctx_formulations).mean()
        unique_forms = form_means.index.values
        k = len(unique_forms)
        if k < MIN_CONTEXT_FORMULATIONS:
            return None, None, None, False

        kernel_idx = self._kernel_feature_indices()
        phi_full = ctx_static[:, kernel_idx]
        form_phi = np.array([phi_full[ctx_formulations == f][0] for f in unique_forms])
        form_resid = np.asarray(form_means.values, dtype=float)

        best_ell = float(KERNEL_BANDWIDTH_CANDIDATES[0])
        gate_passed = False
        for ell in KERNEL_BANDWIDTH_CANDIDATES:  # ascending
            _mae, frac = self._kernel_loo_scan(form_phi, form_resid, ell)
            if frac >= TRANSFER_CHECK_FRAC:
                best_ell = float(ell)
                gate_passed = True
                break

        return form_phi, form_resid, best_ell, gate_passed

    def _predict_kernel_correction(self, q_phi: np.ndarray) -> np.ndarray:
        """Predict kernel-weighted local residual corrections for query descriptors.

        Args:
            q_phi: Restricted kernel-feature representations for the query rows,
                with shape `(n_queries, n_kernel_features)`.

        Returns:
            A NumPy array of shape `(n_queries,)` containing the shrunk
            kernel-weighted residual correction for each query row. Returns zeros
            when the kernel corrector is inactive because its context state is
            unavailable or the transfer gate did not pass.

        Notes:
            The correction is computed from formulation-level context residuals,
            avoiding pseudoreplication across multiple shear measurements from the
            same formulation. Gaussian kernel weights determine the local residual
            estimate, while empirical-Bayes shrinkage uses the kernel weights'
            Kish effective sample size to reduce corrections supported by little
            effective context evidence. Consequently, queries distant from the
            calibrated context are shrunk more strongly toward zero.
        """
        if self._kernel_ctx_phi is None or self._kernel_ctx_resid is None:
            return np.zeros(len(q_phi))

        ell = self.kernel_bandwidth
        sigma2 = max(
            (
                float(np.var(self._kernel_ctx_resid, ddof=1))
                if len(self._kernel_ctx_resid) > 1
                else SIGMA2_WITHIN
            ),
            SIGMA2_WITHIN * 0.5,
        )
        out = np.zeros(len(q_phi))
        for j in range(len(q_phi)):
            d2 = np.sum((self._kernel_ctx_phi - q_phi[j]) ** 2, axis=1)
            w = np.exp(-d2 / (2.0 * ell**2))
            wsum = float(w.sum())
            if wsum <= 1e-12:
                continue
            raw = float(np.sum(w * self._kernel_ctx_resid) / wsum)
            n_eff = (wsum**2) / float(np.sum(w**2))
            shrink = (n_eff * TAU2_BETWEEN) / (n_eff * TAU2_BETWEEN + sigma2)
            out[j] = raw * shrink
        return out

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate viscosity predictions for a query DataFrame.

        Args:
            df: Query formulations to predict. Each row is preprocessed into static
                formulation features and shear-rate inputs before prediction.

        Returns:
            A DataFrame containing the model's viscosity predictions, with the
            query-conditioned local residual correction applied when a calibrated
            context is available.

        Notes:
            Predictions are formed from the zero-shot prior plus the learned local
            correction terms for formulation offset, protein concentration, and
            shear-rate dependence. The concentration term uses each query row's own
            `Protein_conc` rather than the context concentration.

            When the predictor has been reset to zero-shot mode, all learned
            correction terms are zero and prediction reduces to the unadapted model
            path.

            When concentration correction is active, each query concentration is
            clamped to the range observed in the calibration context before the
            concentration-dependent correction is evaluated. This prevents the
            local linear fit from extrapolating beyond the concentration range
            supported by the context.
        """
        q_static, q_shear, _ = self._preprocess(df)
        return self._predict_from_tensors(q_static, q_shear, df)

    def _predict_from_tensors(
        self, q_static: torch.Tensor, q_shear: torch.Tensor, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate predictions from preprocessed static and shear tensors.

        Args:
            q_static: Preprocessed static formulation features with the same layout
                produced by :meth:`_preprocess`.
            q_shear: Preprocessed shear-rate inputs corresponding to `q_static`.
            df: Original query DataFrame used for result-frame assembly and, for the
                linear corrector, the raw `Protein_conc` values. Its row order must
                match the formulation order represented by the input tensors.

        Returns:
            A copy of `df` with one `Pred_<shear>` column added for each shear
            condition in :attr:`shear_map`. Predictions are returned in viscosity
            units after combining the zero-shot prior with the active local
            correction terms and transforming from log10 space.

        Notes:
            This method contains the prediction logic shared by :meth:`predict` and
            evaluation paths that already have a built feature frame. The zero-shot
            prior is computed first, followed by the shear-dependent correction.

            When `corrector_mode` is `"kernel"` and a calibrated kernel context
            is available, the kernel-weighted formulation-level correction replaces
            the linear offset and concentration correction. Otherwise, the linear
            correction uses the query formulation's own `Protein_conc` value,
            clamped to the concentration range observed during calibration.

            The shear correction is evaluated from the raw log10 shear rates and the
            calibration context's fitted shear center. All correction terms are
            combined in log10-viscosity space before conversion back to viscosity.
        """
        prior_log10 = self._prior_log10(q_shear, q_static)

        n_shears = len(self.shear_map)
        raw_log_shears = np.tile(np.array([np.log10(v) for v in self.shear_map.values()]), len(df))
        slope_term = self.slope_hat * (raw_log_shears - self.slope_center)

        if self.corrector_mode == "kernel" and self._kernel_ctx_phi is not None:
            # Kernel mode replaces the linear offset/concentration correction with a
            # single formulation-level correction derived from context similarity.
            # `_preprocess` repeats each formulation's static feature vector once per
            # shear condition, so retain the first entry of each formulation's repeated
            # block before selecting the feature subset used by the kernel metric.
            q_static_np = q_static.detach().cpu().numpy()[0]
            q_static_per_row = q_static_np[::n_shears]
            q_phi = q_static_per_row[:, self._kernel_feature_indices()]
            level_term = np.repeat(self._predict_kernel_correction(q_phi), n_shears)
        else:
            if "Protein_conc" in df.columns:
                conc_vals = pd.to_numeric(df["Protein_conc"], errors="coerce").fillna(0.0).values
            else:
                conc_vals = np.zeros(len(df))
            raw_concs = np.repeat(conc_vals, n_shears)
            clamped_concs = np.clip(raw_concs, self.conc_support_min, self.conc_support_max)
            conc_term = self.conc_hat * (clamped_concs - self.conc_center)
            level_term = self.offset_hat + conc_term

        pred_log10 = prior_log10 + level_term + slope_term
        pred_visc_cp = np.power(10, pred_log10)

        results = df.copy()
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
        ci_range: tuple[float, float] = (2.5, 97.5),
        k: int | None = None,  # retained for API compatibility no longer used
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Estimate predictive uncertainty using Monte Carlo dropout.

        Args:
            df: Query formulations for which predictions and uncertainty estimates
                are required.
            n_samples: Number of stochastic forward passes used to approximate the
                predictive distribution. Larger values generally provide more stable
                uncertainty estimates at the cost of additional inference time.
            ci_range: Lower and upper percentile bounds for the reported confidence
                interval, expressed as percentages. For example, `(2.5, 97.5)`
                produces a nominal 95% interval.
            k: Retained for API compatibility with earlier context-subsampling
                implementations. It is currently unused.

        Returns:
            A tuple `(mean_pred, stats)` where `mean_pred` contains the mean
            predicted viscosity values in the original viscosity units and `stats`
            is a dictionary containing the mean and standard deviation in log10
            space together with the lower and upper confidence-interval bounds in
            viscosity units.

        Notes:
            The model is evaluated with dropout enabled during inference while its
            learned parameters remain fixed. If a calibrated memory vector is
            available, it is held fixed across all stochastic forward passes;
            otherwise, a zero latent memory is used for the zero-shot path.

            Uncertainty is estimated from the distribution of stochastic
            log10-viscosity predictions. Confidence bounds are computed as
            percentiles in log10 space and then transformed back to viscosity space.
            If the configured dropout rate is zero, all stochastic passes are
            deterministic and the resulting interval has zero width.
        """
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
        """Convert scaled decoder outputs back to log10 viscosity values.

        Args:
            q_shear: Scaled shear-rate values corresponding to `out_scaled`.
            out_scaled: Scaled decoder predictions representing log10 viscosity.

        Returns:
            A one-dimensional NumPy array containing the predicted log10 viscosity
            values in the original, unscaled physical space.

        Notes:
            The physics scaler was fitted jointly on log10 shear rate and log10
            viscosity. Both quantities are therefore reconstructed together before
            extracting the inverse-scaled viscosity component.
        """
        q_shear_np = q_shear.cpu().numpy().reshape(-1, 1)
        out_np = out_scaled.cpu().numpy().reshape(-1, 1)
        combined = np.hstack([q_shear_np, out_np])
        log_vals = self.physics_scaler.inverse_transform(combined)[:, 1]
        return log_vals
