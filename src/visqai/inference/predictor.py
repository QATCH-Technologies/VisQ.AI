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

DELTA CORRECTOR (T-R3.1/T-R3.2): the few-shot mechanism
---------------------------------------------------------
`.predict()` no longer routes few-shot through the neural correction_head at
all. models/experiments/rung1_context_learnability's Rung 1 analysis (T2-T6)
established, on the zero-shot LOGO prior's real held-out residuals, that: the
signal is ~44% between-protein variance (ICC), mostly a flat per-protein
offset rather than a shear-dependent shape (T3), worth a real +0.05 log MAE
ceiling (T4), and -- critically -- that a DUMB closed-form estimator (prior
+ mean of context residuals) already recovers most of that ceiling at k>=4
(T5). The neural correction_head, by contrast, measured near-zero real lift
in production LOGO runs. So the few-shot path here is now exactly that dumb
estimator, shrunk (empirical Bayes / James-Stein) to fix the T5-identified
k=1 regression:

    prediction = prior(x) + offset_hat
    offset_hat = shrink(mean_{context}(y - prior(x)))

`prior(x)` is deliberately the CURRENT zero-shot prediction path (decode_from_
memory at literal r=0), not an isolated prior_head call -- Rung 1's entire
residual table was built by scoring predictor.predict() with
memory_vector=None (see build_residuals.py), so the delta corrector has to be
defined against the exact same baseline the T5 estimator was validated
against for the "matches the T5 curve" pass criterion to mean anything.
offset_hat defaults to 0.0 whenever memory_vector is reset to None (see the
`memory_vector` property below), so an empty/absent context is bit-for-bit
identical to the existing zero-shot path -- it cannot regress it.

`.learn()` still also populates `self.memory_vector` via the old neural
encode_memory() path; that's kept ONLY for predict_with_uncertainty's MC-
dropout sampling (a genuinely different, still-useful tool: epistemic
uncertainty over the full decoder, orthogonal to the point-estimate few-shot
mechanism above), not because `.predict()` still needs it.

SLOPE TERM (T-R3.4)
--------------------
Offset-first: T-R3.1/T-R3.2 shipped and were validated alone before this was
added. For ~4 proteins (vudalimab, etanercept, pembrolizumab, belatacept)
Rung 1's T3 found the within-protein residual isn't purely a flat offset --
a shear-dependent slope explains a further 12-25% of variance beyond it --
and confirmed (via a real LOGO run) the offset-only corrector was already
matching its own offset-only oracle for these groups while still leaving a
consistent ~0.011-0.018 log MAE gap versus the offset+slope oracle. So:

    prediction = prior(x) + offset_hat + slope_hat * (log_shear(x) - x_bar)

`slope_hat` is fit on (log_shear, resid - offset_hat) pairs from context --
i.e. AFTER the already-validated offset_hat is subtracted, so the offset
mechanism is completely unchanged; slope is a pure additive term on top, and
`x_bar` (the context's own mean log_shear) is the point the line pivots
around so the offset and slope corrections don't fight each other at
prediction time. Both offset_hat and slope_hat are independently
empirical-Bayes shrunk (see _shrink_offset / _fit_offset_slope). Offline
simulation against the Rung 1 residual table (200 repeats/k, both the 4
flagged proteins and the full 12) confirmed this beats offset-only at every
k>=1 for the flagged proteins (99.9% oracle recovery at k=8) and is neutral
to slightly positive for the offset-dominated majority.

FORMULATION-LEVEL VARIANCE + CONFIDENCE GATE (T-R3.6/T-R3.7)
----------------------------------------------------------------
The full 29-group LOGO re-eval that validated T-R3.1-3.4 still showed a real
outlier: nivolumab at -0.028 lift. Root cause, found by comparing point-level
vs formulation-level ANOVA on the Rung 1 table: _shrink_offset's sigma2 was
being estimated from up to 5 raw SHEAR POINTS per context formulation, but
those points are NOT independent samples -- they share that formulation's own
bias (see T6's stability_ratio; nivolumab's is 0.91, the highest of the
board, meaning its residual is driven far more by WHICH formulation than by
protein identity). Pooling them as if independent inflated the effective
sample size ~5x and understated offset_hat's true uncertainty by roughly the
same factor, so shrinkage was systematically too weak in exactly the noisy
cases it exists to protect against.

The fix: sigma2/tau2 and the sample size `n` used for shrinkage are now
computed from PER-FORMULATION MEAN residuals (one number per context row),
not raw points -- SIGMA2_WITHIN/TAU2_BETWEEN below are recalibrated on that
basis. Two additional gates layer on top:
  - T-R3.7 (hard k>=2): offset_hat is 0.0 whenever context has fewer than 2
    distinct formulations -- a single formulation can't distinguish "this
    protein has a real offset" from "this one formulation is atypical," so
    k=1 no longer corrects at all. This trivially guarantees k=1 lift >= 0
    (identical to zero-shot) rather than relying on shrinkage magnitude to
    get there approximately.
  - T-R3.6 (confidence gate): even at k>=2, offset_hat is zeroed unless
    |raw_mean| >= its own standard error (mean / SE >= 1) -- i.e. the context
    residuals have to be consistently non-zero, not just averaging to a
    number that shrinkage hasn't fully squashed. This is what turns
    marginal/noisy groups into deliberate abstention instead of a small
    wrong correction.
Offline simulation (300 repeats/k) confirmed: k=1 negative-lift count across
all 12 proteins drops from 6 to 0 (exactly 0.0 lift everywhere, by
construction), and nivolumab's k=8 worst-case lift improves from -0.011 to
-0.0096 -- a real but partial improvement, since nivolumab's instability
looks like genuine formulation-level noise (T6 stability_ratio 0.91) rather
than an under-shrunk estimate, and no amount of shrinkage sophistication
recovers signal that isn't consistently there.

WITHIN-CONTEXT TRANSFER CHECK (T-R3.8)
-----------------------------------------
T-R3.6's confidence gate tests "is this number distinguishable from zero" --
but nivolumab's offset largely WAS distinguishable from zero (it has a real,
non-trivial mean), it just doesn't reliably describe the protein: T6's
stability_ratio (0.91, the highest of the whole board) means the "offset"
varies substantially by which formulation you're looking at. A magnitude
test can't see that; it only sees a mean and a variance. So T-R3.6's
confidence gate is REPLACED (not supplemented) by a direct generalization
test: leave-one-formulation-out within the context itself. For each context
formulation held back in turn, estimate the offset from the OTHER k-1
formulations (raw mean, unshrunk -- shrinking a k-1 LOO estimate would make
it too conservative to discriminate transfer from non-transfer) and check
whether applying it actually reduces error on the held-back formulation
versus doing nothing. Only if >= TRANSFER_CHECK_FRAC (2/3) of the k LOO
folds show improvement does the corrector fire at all -- using the FULL,
properly shrunk (T-R3.2) k-formulation offset, not the LOO estimates
themselves. This directly answers "does this offset transfer to a formulation
it wasn't estimated from," which is exactly what a real per-protein constant
must do and what nivolumab's noisy, formulation-dependent residual can't.
Both offset_hat and slope_hat are gated together -- if the flat offset
doesn't even transfer, a slope on top of it isn't more trustworthy.

Offline simulation (300 repeats/k) at frac=2/3: nivolumab's k=8 lift improves
from -0.0096 (T-R3.6) to -0.0017; the worst k=8 group across all 12 proteins
becomes belatacept at -0.0024, clearing the -0.005 bar. Mean k=8 lift drops
from +0.0355 to +0.0178 -- a real cost (more groups abstain more often) for
eliminating the worst-case failure.

QUERY-CONDITIONED LOCAL RESIDUAL (Task 1.1, issue1_query_conditioned_correction_plan.md)
------------------------------------------------------------------------------------------
Everything above (T-R3.1-3.8) is a corrector that is CONSTANT over the
formulation feature space: offset_hat/slope_hat depend only on the
within-curve shear axis, so context can shift a protein's curve up/down and
tilt its shear-thinning, but it cannot correct how that protein responds to
a *new condition* (e.g. a query at a higher concentration than anything in
context). That is Weakness #1 the plan names first. The fix generalizes the
scalar offset to a shrunk linear model on context residuals with basis
`b = [1, (log_shear - sbar), (conc - cbar)]`:

    prediction = prior(x) + offset_hat + conc_hat*(conc(x) - conc_center)
                 + slope_hat*(log_shear(x) - slope_center)

`offset_hat`/`conc_hat`/`conc_center` are identified TOGETHER at the
FORMULATION level (see `_fit_formulation_level`): one number per context
formulation (pseudoreplication fix, rule 3) regressed on `[1, conc-cbar]`,
ridge-shrunk with `Î = sigma2 * diag(1/TAU2_BETWEEN, 1/TAU2_CONC)` -- the
same empirical-Bayes construction `_shrink_offset` used, just 2-dimensional.
`slope_hat` is fit exactly as T-R3.4 did, but on residuals AFTER subtracting
the FULL formulation-level correction (intercept + conc term), not just the
old scalar offset, so it never competes with the new conc term.

FALLBACK (hard guarantee): when context spans <2 distinct `Protein_conc`
values, `_fit_formulation_level`'s conc branch never fires -- conc_hat is
exactly 0.0 and offset_hat collapses to the OLD scalar `_shrink_offset`
formula bit-for-bit, so a context that doesn't vary concentration reproduces
the pre-Task-1.1 offset+slope corrector exactly (see
test_predictor_local_residual.py's fallback test).

T-R3.8's within-context transfer check is likewise generalized (not just
supplemented): each LOO fold now fits the WHOLE formulation-level model
(`_fit_formulation_level_raw`, unshrunk -- same reasoning as before, a
shrunk k-1 estimate would be too conservative to discriminate transfer from
non-transfer) and tests whether it reduces the held-back formulation's
error, gating offset_hat/conc_hat/slope_hat together exactly like before.

TAU2_CONC is calibrated the same way TAU2_SLOPE was -- population variance
(ddof=1) of PER-PROTEIN OLS slopes of (formulation-mean resid) vs.
Protein_conc, computed from the same leave-one-protein-out zero-shot
residual table (models/experiments/rung1_context_learnability/residuals.csv)
TAU2_SLOPE/TAU2_BETWEEN/SIGMA2_WITHIN were calibrated from -- a fold
structure disjoint from the condition-shift acceptance eval (Task 0.1),
per the plan's ground rule against training/calibrating on the acceptance
metric itself. Unlike TAU2_SLOPE (fit on raw per-shear points, since shear
varies within a formulation), the conc slope is fit on PER-FORMULATION MEAN
residuals -- Protein_conc doesn't vary within a formulation, so this is
already the correct unit of replication.

DESCRIPTOR-OOD DOWN-WEIGHTING (parallel prior-side track, not corrector work)
------------------------------------------------------------------------------
Separate from the few-shot mechanism above: visqai.eval.cnp_logo's
FOLD RANGE GUARD (_check_fold_feature_range, FOLD_RANGE_N_SIGMA=5.0) has
been LOGGING, for every LOGO fold, which engineered feature columns a
held-out group's real values exceed 5 standard deviations of the training
fold's fitted distribution on -- e.g. nivolumab's prior_lysine/conc_sq_x_kP/
nearpI_x_conc, all repeatedly flagged. That guard only had visibility; it
never changed a prediction. DESCRIPTOR_OOD_CLIP_SIGMA turns it into an actual
behavior change: _preprocess and _context_residuals now clip every SCALED
numeric feature to +/- that many standard deviations (StandardScaler's
output is already in those units, so this is a plain np.clip) before it
reaches the network, both for query points (zero-shot AND few-shot
predictions) and for context residual estimation. A held-out group's
genuinely out-of-distribution descriptor value is capped at the training
fold's own edge instead of injecting a raw, unbounded activation the network
never learned to handle -- this is exactly the same idea as
visqai.training.data's ZERO_VARIANCE_FALLBACK_SCALE (bounding a different
failure mode of the same underlying problem: OOD activations at held-out
time), just applied to every numeric column instead of only zero-variance
ones. training.data.load_and_preprocess applies the identical clip at fit
time for train/inference symmetry.
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
            def __init__(self, columns=(), *, future_dtype=None,
                         warning_was_emitted=False, warning_enabled=True):
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
# SCALED numeric feature to +/- this many standard deviations before it
# reaches the network. StandardScaler's output IS already in
# standard-deviation units, so this is a plain np.clip -- no extra stats
# needed. Matches visqai.eval.cnp_logo.FOLD_RANGE_N_SIGMA (5.0), the
# existing diagnostic threshold that LOGS a held-out group's OOD descriptor
# values without acting on them; this constant is what turns that diagnostic
# into an actual behavior change. A held-out group whose real feature value
# sits past 5 sigma of what the training fold ever represented gets capped
# at the fold's own edge instead of injecting a raw, unbounded activation
# the network never learned to handle -- see PREDICTOR docstring section
# "DESCRIPTOR-OOD DOWN-WEIGHTING".
DESCRIPTOR_OOD_CLIP_SIGMA: float = 5.0

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
# visqai.eval.condition_shift's module docstring -- showed post-A.3 kernel
# captures only ~27% of linear's bad-prior gain (offset-only +0.054, linear
# +0.075, kernel +0.020) in exchange for a good-prior "gain" (+0.008) that
# is not distinguishable from noise at this sample size. Never pick a
# corrector based on the good-prior number alone -- that stratum currently
# has no statistical power to tell correctors apart.
KERNEL_BANDWIDTH_CANDIDATES: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)


class ViscosityPredictorCNP:
    def __init__(self, model_dir: str, verbose: bool = False):
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")
        if not verbose:
            self._logger.setLevel(logging.CRITICAL)
        self._logger.info(f"Initializing ViscosityPredictorCNP with model_dir: {model_dir}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self._memory_vector = None  # Stores the calibrated context (legacy neural path)
        self.offset_hat = 0.0  # Delta corrector's shrunk offset (T-R3.1/T-R3.2) -- the actual few-shot signal
        self.conc_hat = 0.0  # Query-conditioned corrector's shrunk Protein_conc coefficient (Task 1.1)
        self.conc_center = 0.0  # Protein_conc pivot conc_hat is fit around (context's own mean conc)
        self.conc_support_min = 0.0  # Task A.3: context's own min Protein_conc -- clamps predict()'s query conc
        self.conc_support_max = 0.0  # Task A.3: context's own max Protein_conc -- see predict()'s docstring
        self.slope_hat = 0.0  # Delta corrector's shrunk shear-slope (T-R3.4), additive on top of offset_hat
        self.slope_center = 0.0  # log_shear pivot point slope_hat is fit around (context's own mean log_shear)
        self.n_context_points = 0  # Real measured points offset_hat/slope_hat were estimated from

        # Task 1.2: kernel-weighted local residual corrector, an alternative
        # to Task 1.1's linear (offset+conc) model for axes where a linear
        # term is the wrong shape (additive/buffer identity). Selectable via
        # `corrector_mode` ("linear" is the default/Task-1.1 behavior,
        # unchanged; "kernel" opts into this path). See module docstring.
        self.corrector_mode = "linear"
        self._kernel_ctx_phi = None  # (n_formulations, n_kernel_features) context similarity vectors
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

        # Pristine copy of the weights, restored before each learn() call so
        # successive calls for different proteins never contaminate each other.
        self._original_state = copy.deepcopy(self.model.state_dict())

        # Raw context tensor from the last learn() call, used by
        # predict_with_uncertainty for context-subsampling CI.
        self.context_t: Optional[torch.Tensor] = None

        self.shear_map = dict(SHEAR_MAP)

    # ------------------------------------------------------------------
    @property
    def memory_vector(self):
        return self._memory_vector

    @memory_vector.setter
    def memory_vector(self, value):
        """`predictor.memory_vector = None` is the established "reset to
        zero-shot" idiom every existing caller uses (cnp_logo._shot_metrics,
        cli.parity_eval, eval.predictor_harness.reset_memory) -- keep
        offset_hat (the delta corrector's actual few-shot signal) in sync so
        those callers get a real reset without needing to know it exists."""
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

    # ------------------------------------------------------------------
    def _clip_descriptor_ood(self, X_static: np.ndarray) -> np.ndarray:
        """Cap every SCALED numeric feature to +/- DESCRIPTOR_OOD_CLIP_SIGMA
        standard deviations -- see that constant's docstring. Only the "num"
        block (StandardScaler output) is meaningful to clip this way; the
        "cat" one-hot block is already in {0,1} and clipping it would be a
        no-op anyway, so this only touches the leading n_num columns."""
        n_num = len(self.preprocessor.transformers_[0][2])
        X_static[:, :n_num] = np.clip(
            X_static[:, :n_num], -DESCRIPTOR_OOD_CLIP_SIGMA, DESCRIPTOR_OOD_CLIP_SIGMA
        )
        return X_static

    def _preprocess(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        df_proc = df.copy()

        for col in df_proc.select_dtypes(include=["object"]):
            df_proc[col] = df_proc[col].apply(lambda x: x.value if hasattr(x, "value") else x)

        if "ID" in df_proc.columns:
            df_proc = df_proc.drop(columns=["ID"])

        df_proc, _num_cols, _cat_cols = build_feature_frame(df_proc)
        return self._preprocess_built(df_proc)

    def _preprocess_built(self, df_proc: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The back half of _preprocess: everything AFTER build_feature_frame
        (missing-column fill, scale, clip, tensor construction). Split out so
        visqai.eval.predictor_harness.predict_from_built can hand in a
        DataFrame that's already been through build_feature_frame with one
        ENGINEERED column deliberately mutated (permutation feature
        importance on e.g. conc_sq or whole_charge) -- calling the full
        _preprocess on such a frame would silently re-derive that column from
        its raw inputs via another build_feature_frame pass and overwrite the
        mutation before it ever reached the model."""
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
        X_static = self._clip_descriptor_ood(X_static)

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

        # Legacy neural memory (kept only for predict_with_uncertainty --
        # see module docstring). Not used by predict()'s main path anymore.
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

        # Delta corrector (T-R3.1/T-R3.2/T-R3.4/T-R3.6/T-R3.7, generalized by
        # Task 1.1 to be query-conditioned on Protein_conc): the actual
        # few-shot mechanism.
        ctx_formulations, ctx_shear, ctx_resid, ctx_conc, ctx_static = self._context_residuals(df)

        # Task A.2 (addendum): corrector_mode="offset_only" is the ABLATION
        # arm for measuring whether Task 1.1/1.2 add anything over the
        # pre-Task-1.1 corrector -- zeroing ctx_conc before the fit forces
        # `_fit_formulation_level`'s conc branch to never fire (constant
        # input -> zero variance -> conc_hat stays exactly 0.0, the same
        # code path the Fallback guarantee already proves is bit-for-bit the
        # old scalar-offset+slope corrector). This reuses the exact fitting
        # code rather than a separate implementation, so there is no risk of
        # the ablation arm silently drifting from what "pre-Task-1.1" means.
        ctx_conc_for_fit = ctx_conc if self.corrector_mode != "offset_only" else np.zeros_like(ctx_conc)
        (
            self.offset_hat,
            self.conc_hat,
            self.conc_center,
            self.slope_hat,
            self.slope_center,
        ) = self._fit_local_residual(ctx_formulations, ctx_shear, ctx_resid, ctx_conc_for_fit)
        self.n_context_points = len(ctx_resid)

        # Task A.3: the context's own Protein_conc range -- predict() clamps
        # every query's conc to this range before evaluating conc_hat's
        # linear term, so the fit only ever INTERPOLATES within what it was
        # estimated from. Real-run evidence for why this is needed: a linear
        # trend fit on a context's lower/upper half, evaluated at a query
        # OUTSIDE that half (the corrector's whole reason to exist), was
        # extrapolating the line arbitrarily far -- harmless when the trend
        # happens to be real and the query isn't too far out, but with no
        # guardrail at all a noisy/short-range slope estimate can blow up
        # for a query several times further from context than the context's
        # own span. Harmless when conc_hat is 0 (clamping a term that's
        # multiplied by zero changes nothing).
        if len(ctx_conc) > 0:
            self.conc_support_min = float(np.min(ctx_conc))
            self.conc_support_max = float(np.max(ctx_conc))
        else:
            self.conc_support_min = 0.0
            self.conc_support_max = 0.0

        # Task 1.2: kernel-weighted local residual, an alternative to the
        # linear model above for axes (additive/buffer identity) where a
        # linear term is the wrong shape. Only fit when explicitly selected
        # -- corrector_mode defaults to "linear", so this is a no-op (and
        # predict() ignores kernel state entirely) unless a caller opts in.
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
        """prior(x) from the delta-corrector spec: the CURRENT zero-shot
        prediction path (decode_from_memory at literal r=0), in real
        log10-viscosity space. See module docstring for why this -- not an
        isolated prior_head call -- is the correct "prior"."""
        zero_mem = torch.zeros((1, self.config["latent_dim"]), device=self.device)
        self.model.eval()
        with torch.no_grad():
            prior_scaled = self.model.decode_from_memory(zero_mem, q_shear, q_static)
        return self._inverse_to_log(q_shear, prior_scaled)

    def _context_residuals(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(formulation_idx, log_shear_raw, true_log10 - prior_log10,
        Protein_conc_raw, static_features) for every REAL measured point in
        `df`, masked exactly like Rung 1's T1
        (models/experiments/rung1_context_learnability/build_residuals.py)
        -- a context row missing a value at some shear rate must NOT
        contribute a fabricated residual through _preprocess's
        placeholder-fill (val=1.0) path. That placeholder is harmless on the
        query side of predict() (there's no truth to compare against, we're
        just predicting there); it would silently corrupt offset_hat/
        slope_hat/conc_hat here. log_shear_raw is UNSCALED log10(shear rate)
        -- matches Rung 1's T3 convention. Protein_conc_raw is UNSCALED (not
        centered -- callers center on the context's own mean, mirroring how
        log_shear is handled). static_features is the FULL scaled/one-hot
        feature row (same block _preprocess produces) per point -- Task
        1.2's kernel corrector selects its similarity subset from this via
        `_kernel_feature_indices`. formulation_idx is `df`'s own row
        position (0..len(df)-1, one integer per point, repeated for every
        real shear measurement that row has) -- callers group by it to
        collapse to per-formulation means for T-R3.6/T-R3.7/Task-1.1/Task-1.2
        (see module docstring's "FORMULATION-LEVEL VARIANCE" section for why
        raw points can't be pooled as if independent)."""
        if df.empty:
            return np.empty(0, dtype=int), np.empty(0), np.empty(0), np.empty(0), np.empty((0, 0))

        df_proc = df.copy()
        for col in df_proc.select_dtypes(include=["object"]):
            df_proc[col] = df_proc[col].apply(lambda x: x.value if hasattr(x, "value") else x)
        if "ID" in df_proc.columns:
            df_proc = df_proc.drop(columns=["ID"])
        df_proc, _num_cols, _cat_cols = build_feature_frame(df_proc)

        feature_names = (
            self.preprocessor.feature_names_in_ if hasattr(self.preprocessor, "feature_names_in_") else []
        )
        for c in feature_names:
            if c not in df_proc.columns:
                df_proc[c] = 0.0
        X_static = self.preprocessor.transform(df_proc)
        if np.isnan(X_static).any():
            X_static = np.nan_to_num(X_static)
        X_static = self._clip_descriptor_ood(X_static)

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
                statics.append(X_static[i])
                concs.append(conc_val)

        if not shear_logs:
            return np.empty(0, dtype=int), np.empty(0), np.empty(0), np.empty(0), np.empty((0, 0))

        statics_arr = np.array(statics)
        raw_points = np.column_stack([shear_logs, true_logs])
        scaled_points = self.physics_scaler.transform(raw_points)
        shear_t = torch.tensor(scaled_points[:, [0]], dtype=torch.float32).unsqueeze(0).to(self.device)
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
    ) -> Tuple[float, float, float]:
        """Shrunk (empirical-Bayes) formulation-level fit of [offset,
        conc_coeff] on basis [1, conc-cbar] (Task 1.1) -- generalizes
        T-R3.2's `_shrink_offset` scalar shrinkage to 2 dimensions. Caller
        (`_fit_local_residual`) already enforces T-R3.7's k>=2 formulation
        gate and the LOO transfer check before calling this. `sigma2` is the
        SAME per-formulation noise estimate the old `_shrink_offset` used
        (blended with SIGMA2_WITHIN so a lucky low-variance draw at small k
        can't collapse shrinkage to ~none). `Î = sigma2 * diag(1/TAU2_BETWEEN,
        1/TAU2_CONC)` makes this exactly a 2D ridge/Bayesian-linear-
        regression posterior mean; with k formulations and a constant conc
        column it collapses ALGEBRAICALLY to the old scalar
        `raw * (k*TAU2_BETWEEN)/(k*TAU2_BETWEEN+sigma2)` formula (see module
        docstring's FALLBACK guarantee) -- when context spans <2 distinct
        Protein_conc values, conc_coeff can't be separated from noise, so it
        is returned as exactly 0.0 rather than fit."""
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
    ) -> Tuple[float, float, float]:
        """Unshrunk formulation-level [offset, conc_coeff] fit used ONLY by
        the LOO transfer check (`_transfer_check_passes`) -- shrinking a k-1
        LOO estimate would make it too conservative to discriminate transfer
        from non-transfer (same reasoning T-R3.8 always used for the scalar
        offset). Plain least-squares, falling back to the raw per-formulation
        mean (conc_coeff=0.0) when there are <2 formulations or the LOO
        subset's conc column is degenerate."""
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
        """T-R3.8, generalized by Task 1.1 to test the WHOLE local model
        (offset + conc term), not just the scalar offset -- leave-one-
        formulation-out transfer check. For each context formulation held
        back in turn, fit the formulation-level model from the OTHER k-1
        formulations (`_fit_formulation_level_raw`, unshrunk) and check
        whether applying its prediction (evaluated at the held-back
        formulation's own conc) actually reduces error there versus doing
        nothing. Requires TRANSFER_CHECK_FRAC of the k folds to show
        improvement. When context conc is constant this reduces exactly to
        the old scalar-offset LOO check (conc_coeff is always 0.0 in that
        case), preserving the Task-1.1 fallback guarantee. See module
        docstring."""
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
    ) -> Tuple[float, float, float, float, float]:
        """Task 1.1: query-conditioned local residual model. Generalizes the
        old `_fit_offset_slope` (scalar offset, then shear slope) to a
        shrunk multi-basis fit on [1, (log_shear-sbar), (conc-cbar)], so
        context that spans a concentration RANGE can correct a query at a
        DIFFERENT concentration -- not just shift/tilt the curve uniformly
        (see module docstring). Order: T-R3.7 formulation-count gate ->
        generalized T-R3.8 whole-model LOO transfer check -> formulation-
        level [offset, conc_coeff] fit (shrunk, `_fit_formulation_level`) ->
        shear-slope fit on raw points, AFTER subtracting the FULL
        formulation-level correction (offset + conc_coeff term), same
        empirical-Bayes shrinkage as before. Returns (offset_hat, conc_hat,
        conc_center, slope_hat, slope_center).

        FALLBACK (hard): with <2 distinct context Protein_conc values,
        conc_hat is exactly 0.0 and offset_hat/slope_hat are bit-for-bit the
        old offset+slope corrector's output (see `_fit_formulation_level`
        and `_transfer_check_passes` docstrings for why each step
        individually collapses to its pre-Task-1.1 form)."""
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

    def _kernel_feature_indices(self) -> list:
        """Task 1.2: column indices (into the preprocessor's leading SCALED
        numeric block, same block `_clip_descriptor_ood` clips) used for the
        kernel corrector's similarity metric -- Protein_conc plus every
        ingredient/buffer physicochemical property column
        (visqai.preprocessing.pipeline's property-space columns). This is
        the "physically-meaningful subset" the plan calls for: restricting
        to it keeps similarity from being diluted by inert dimensions (MW,
        temperature, engineered cross-terms, etc. all sit in the same scaled
        block but say nothing about which additive/buffer is present).
        Cached per instance -- the preprocessor's column layout never
        changes after __init__."""
        if self._kernel_feat_idx is None:
            from visqai.preprocessing.pipeline import _all_property_columns

            num_cols = list(self.preprocessor.transformers_[0][2])
            wanted = {"Protein_conc"} | _all_property_columns()
            self._kernel_feat_idx = [i for i, c in enumerate(num_cols) if c in wanted]
        return self._kernel_feat_idx

    def _kernel_loo_scan(self, form_phi: np.ndarray, form_resid: np.ndarray, ell: float) -> Tuple[float, float]:
        """For bandwidth `ell`: leave-one-formulation-out over the context,
        predicting each held-back formulation's residual from a Gaussian-
        kernel-weighted mean of the OTHER k-1 formulations' residuals (raw,
        unshrunk -- same reasoning T-R3.8 always used: shrinking a k-1 LOO
        estimate would make it too conservative to discriminate transfer
        from non-transfer). Returns (mean_abs_loo_error, frac_improved) --
        frac_improved is the fraction of formulations where the LOO
        kernel prediction reduced error vs. doing nothing, used both to pick
        `ell` (minimize mean_abs_loo_error) and to gate the corrector
        (TRANSFER_CHECK_FRAC of frac_improved, mirroring T-R3.8)."""
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
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], bool]:
        """Task 1.2: kernel-weighted local residual corrector -- an
        alternative to `_fit_local_residual`'s linear model for axes
        (additive/buffer identity) where a linear term is the wrong shape.
        Weights each context FORMULATION's mean residual (pseudoreplication
        fix, rule 3 -- one number per formulation) by Gaussian similarity, in
        the physically-meaningful feature subset only (`_kernel_feature_
        indices`), to the query. Bandwidth is chosen by leave-one-
        formulation-out on the CONTEXT ITSELF (ground rule #1 -- never the
        acceptance eval); the SAME LOO scan also serves as the transfer-check
        gate (T-R3.7/T-R3.8's philosophy, generalized here): fires only if
        some bandwidth's LOO reconstruction beats doing nothing on
        >= TRANSFER_CHECK_FRAC of formulations. T-R3.7's k>=2 formulation
        gate applies identically. Returns (form_phi, form_resid, bandwidth,
        gate_passed) -- `predict()` uses form_phi/form_resid/bandwidth only
        when gate_passed.

        Task A.3: candidates are tried SMALLEST-first
        (KERNEL_BANDWIDTH_CANDIDATES is ascending) and the FIRST one that
        clears TRANSFER_CHECK_FRAC wins -- NOT whichever minimizes LOO MAE.
        A wider bandwidth can win on raw within-context reconstruction error
        while still being too wide to decay to ~0 for a query genuinely
        outside context support (the kernel's whole safety mechanism against
        extrapolation); preferring the smallest bandwidth that still clears
        the transfer-check bar keeps the correction as localized as the
        evidence allows, which is the more conservative (and more honestly
        "this is what the context actually supports") choice whenever
        several bandwidths are about equally good at LOO reconstruction."""
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
        for ell in KERNEL_BANDWIDTH_CANDIDATES:  # ascending -- most localized first
            _mae, frac = self._kernel_loo_scan(form_phi, form_resid, ell)
            if frac >= TRANSFER_CHECK_FRAC:
                best_ell = float(ell)
                gate_passed = True
                break

        return form_phi, form_resid, best_ell, gate_passed

    def _predict_kernel_correction(self, q_phi: np.ndarray) -> np.ndarray:
        """Task 1.2: shrunk kernel-weighted correction for each query row's
        restricted feature vector `q_phi` (n_queries, n_kernel_features).
        Returns 0.0 for every row when the kernel corrector hasn't fired
        (`_kernel_ctx_phi` is None -- gate failed or memory_vector reset),
        so this is a no-op exactly like offset_hat/conc_hat default to 0.
        Empirical-Bayes shrinkage reuses TAU2_BETWEEN/SIGMA2_WITHIN (same
        population priors as the linear corrector) with the kernel weights'
        EFFECTIVE sample size (Kish's formula, (sum w)^2 / sum(w^2)) in
        place of a plain formulation count -- a query far from every context
        point (small effective n) gets shrunk hard toward 0, a query near a
        cluster of context formulations (large effective n) gets to trust
        the local mean more."""
        if self._kernel_ctx_phi is None or self._kernel_ctx_resid is None:
            return np.zeros(len(q_phi))

        ell = self.kernel_bandwidth
        sigma2 = max(
            float(np.var(self._kernel_ctx_resid, ddof=1)) if len(self._kernel_ctx_resid) > 1 else SIGMA2_WITHIN,
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
        """Query-conditioned delta-corrector prediction (Task 1.1):
        prior(x) + offset_hat + conc_hat*(Protein_conc(x) - conc_center) +
        slope_hat*(log_shear(x) - slope_center) (see module docstring).
        offset_hat/conc_hat/slope_hat are all 0.0 whenever memory_vector has
        been reset to None (the memory_vector property setter keeps them in
        sync), so this is bit-for-bit the existing zero-shot path when
        there's no context. Uses each QUERY row's own Protein_conc -- not
        the context's -- so a query at a different concentration than any
        context formulation gets a real, distinct correction rather than the
        flat per-protein constant the pre-Task-1.1 corrector applied
        everywhere.

        Task A.3: the query's Protein_conc is CLAMPED to
        [conc_support_min, conc_support_max] -- the context's own observed
        range -- before evaluating conc_hat's linear term, so the fit only
        ever interpolates within evidence it was actually estimated from. A
        linear trend estimated from a handful of context formulations is not
        trustworthy extrapolated arbitrarily far past them; clamping means a
        query beyond context support gets exactly the correction at the
        NEAREST edge of that support, not a runaway linear projection."""
        q_static, q_shear, _ = self._preprocess(df)
        return self._predict_from_tensors(q_static, q_shear, df)

    def _predict_from_tensors(
        self, q_static: torch.Tensor, q_shear: torch.Tensor, df: pd.DataFrame
    ) -> pd.DataFrame:
        """The back half of predict(): prior + delta-corrector terms +
        results-frame assembly, given already-preprocessed static/shear
        tensors. Split out so visqai.eval.predictor_harness.predict_from_built
        can supply tensors built from a DataFrame that skipped predict()'s own
        _preprocess (see _preprocess_built's docstring) while sharing this
        exact corrector logic rather than re-implementing it. `df` is only
        used for its raw Protein_conc column and row count -- same role it
        plays in predict()'s own call."""
        prior_log10 = self._prior_log10(q_shear, q_static)

        n_shears = len(self.shear_map)
        raw_log_shears = np.tile(np.array([np.log10(v) for v in self.shear_map.values()]), len(df))
        slope_term = self.slope_hat * (raw_log_shears - self.slope_center)

        if self.corrector_mode == "kernel" and self._kernel_ctx_phi is not None:
            # Task 1.2: kernel-weighted level correction replaces the linear
            # offset+conc term entirely (never both -- see module docstring).
            # q_static repeats each row's static features once per shear (see
            # _preprocess), so every n_shears-th entry is one row's own
            # vector -- pull those out before restricting to the kernel's
            # similarity subset.
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
