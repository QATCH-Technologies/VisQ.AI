"""
data.py
=======
Training-time data loading: load_and_preprocess builds a fitted
ColumnTransformer + physics_scaler and resamples each formulation's viscosity
curve into a fixed set of key-shear-rate points via PCHIP interpolation.

Moved from ml/cnp_mk2/train_o_net_v4_rung1.py. The row-level feature
engineering (categorical/charge featurization, unit conversion, physics
priors) now goes through visqai.preprocessing.pipeline.build_feature_frame --
previously this logic was duplicated inline here and, independently and
divergently, in inference_o_net.py::_preprocess (see pipeline.py's module
docstring for the bug that duplication caused).

Not carried forward: the original wrote `df.to_csv("pembro_data.csv")` as a
debug side effect on every call (writes a fixed filename into whatever the
current working directory happens to be). That was leftover debug output,
not documented behavior -- dropped rather than ported.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from visqai.preprocessing.pipeline import build_feature_frame, protected_feature_indices, SHEAR_MAP

# Lowercase keys matching the "group" field produced by load_and_preprocess.
# Groups in the same class are "hard negatives" for the contrastive loss --
# they share similar static features and are the most important pairs for it
# to distinguish.
PROTEIN_CLASS_MAP = {
    "adalimumab": "igg1",
    "bevacizumab": "igg1",
    "trastuzumab": "igg1",
    "pembrolizumab": "igg4",
    "ibalizumab": "igg4",
    "nivolumab": "igg4",
    "belatacept": "fc_fusion",
    "etanercept": "fc_fusion",
    "vudalimab": "bispecific",
    "poly-higg": "polyclonal",
    "bgg": "polyclonal",
    "bsa": "other",
    "risankizumab": "igg1",
}
# Groups excluded from contrastive/consistency sampling -- they don't
# represent a specific protein type and would produce trivially easy
# negative examples.
NON_PROTEIN_GROUPS = {"none"}

KEY_SHEARS = [100.0, 1000.0, 10000.0, 100000.0, 15000000.0]

# When a numeric column has zero variance in a training fold (e.g. a
# leave-one-ingredient-out fold where the held-out ingredient's engineered
# columns are identically zero/constant in every training row), sklearn's
# StandardScaler sets scale_=1 for that column (see
# sklearn.preprocessing._data._handle_zeros_in_scale) -- which means
# .transform() on a later OOD value is just (raw - mean), i.e. the raw
# magnitude passes through almost untouched. This fixed a-priori scale
# replaces that degenerate 1.0 so a constant-in-training column can never
# inject a raw-unit activation regardless of what a held-out fold's value
# turns out to be. It is deliberately NOT derived from this fold's own data
# (that would just reproduce the degenerate case) -- it's a fixed, generous
# guess at "typical" spread for an engineered feature of this pipeline, which
# is all that's needed once the raw feature magnitudes themselves are already
# bounded at construction (see categorical.py's *_mw_ratio fields and
# pipeline.py's SALT_MG_ML_CAP).
ZERO_VARIANCE_FALLBACK_SCALE: float = 10.0

# Descriptor-OOD down-weighting (prior-side, not corrector work): caps every
# SCALED numeric feature to +/- this many standard deviations, both here (at
# fit time -- a single outlier training row shouldn't inject a raw, unbounded
# activation either) and in visqai.inference.predictor's _preprocess/
# _context_residuals (at inference time, where a held-out group's real value
# can legitimately exceed anything the training fold represented). Matches
# both DESCRIPTOR_OOD_CLIP_SIGMA in predictor.py and
# visqai.eval.cnp_logo.FOLD_RANGE_N_SIGMA, the existing diagnostic that logs
# this exact condition without acting on it -- keep all three in sync.
DESCRIPTOR_OOD_CLIP_SIGMA: float = 5.0


def _fix_zero_variance_scale(preprocessor: ColumnTransformer, X_matrix: np.ndarray, num_cols: list[str]) -> np.ndarray:
    """Patch the fitted 'num' StandardScaler in place: any column that came
    out zero-variance in this fit gets ZERO_VARIANCE_FALLBACK_SCALE instead
    of sklearn's degenerate scale_=1. Patches both the already-computed
    X_matrix (fit_transform used the old scale) and the persisted scaler
    object, so every future .transform() call -- including at inference time
    via the joblib-dumped preprocessor.pkl -- inherits the same bounded
    behaviour for that column."""
    scaler = preprocessor.named_transformers_["num"]
    zero_var = np.where(scaler.var_ <= 1e-12)[0]
    if len(zero_var) == 0:
        return X_matrix
    print(
        f"  [load_and_preprocess] zero-variance training columns "
        f"{[num_cols[i] for i in zero_var]} -> using fixed a-priori scale "
        f"{ZERO_VARIANCE_FALLBACK_SCALE} instead of sklearn's degenerate scale_=1."
    )
    for i in zero_var:
        # fit_transform already produced (raw - mean_) / 1.0 for these
        # columns; rescale in place by the fixed factor rather than
        # refitting, then persist the same fix into the scaler itself.
        X_matrix[:, i] = X_matrix[:, i] / ZERO_VARIANCE_FALLBACK_SCALE
        scaler.scale_[i] = ZERO_VARIANCE_FALLBACK_SCALE
    return X_matrix


def _drop_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-2.4 data-hygiene finding (issue1_query_conditioned_correction_
    plan.md Finding C): a handful of rows in the raw export are entirely
    blank (no Protein_type, no Protein_conc -- trailing/malformed CSV rows,
    not real measurements; two even carry a stray leftover Viscosity_100
    value with every other field NaN). Unlike the eval/LOGO harness path
    (which always runs visqai.eval.data_prep.prepare_df first),
    load_and_preprocess is the real production training entrypoint and had
    no such filter -- these rows were reaching build_feature_frame's
    fillna(0.0), then preprocessor.fit_transform(df), contaminating the
    fitted StandardScaler's mean/scale for every numeric column with
    all-zero(ish) rows before the later per-sample `len(raw_x) < 3` check
    drops them from actual training samples. Drop them before anything is
    fit."""
    return df[df["Protein_type"].notna() | df["Protein_conc"].notna()].reset_index(drop=True)


def load_and_preprocess(csv_path, save_dir=None):
    df = pd.read_csv(csv_path)
    df = _drop_blank_rows(df)

    df, num_cols, cat_cols = build_feature_frame(df)
    protected_indices = protected_feature_indices(num_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    X_matrix = preprocessor.fit_transform(df)
    X_matrix = _fix_zero_variance_scale(preprocessor, X_matrix, num_cols)
    if np.isnan(X_matrix).any():
        print("WARNING: NaNs found in X_matrix after preprocessing! Replacing with 0.")
        X_matrix = np.nan_to_num(X_matrix)
    X_matrix[:, : len(num_cols)] = np.clip(
        X_matrix[:, : len(num_cols)], -DESCRIPTOR_OOD_CLIP_SIGMA, DESCRIPTOR_OOD_CLIP_SIGMA
    )

    all_shear, all_visc = [], []
    for i in range(len(df)):
        for col, shear_val in SHEAR_MAP.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = df.iloc[i][col]
                if v <= 0:
                    v = 1e-6
                all_shear.append(np.log10(shear_val))
                all_visc.append(np.log10(v))

    physics_scaler = StandardScaler()
    physics_scaler.fit(np.column_stack([all_shear, all_visc]))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.pkl"))
        joblib.dump(physics_scaler, os.path.join(save_dir, "physics_scaler.pkl"))
        joblib.dump(protected_indices, os.path.join(save_dir, "protected_indices.pkl"))

    # --- Target resampling: PCHIP-interpolate each curve, densely sample it,
    # and re-express every point relative to the fixed KEY_SHEARS grid. ---
    samples = []
    key_logs = np.log10(KEY_SHEARS)

    for i in range(len(df)):
        raw_x, raw_y = [], []
        for col, shear_val in SHEAR_MAP.items():
            if col in df.columns and pd.notna(df.iloc[i][col]):
                v = df.iloc[i][col]
                if v <= 0:
                    v = 1e-6
                raw_x.append(np.log10(shear_val))
                raw_y.append(np.log10(v))

        if len(raw_x) < 3:
            continue

        si = np.argsort(raw_x)
        x_arr, y_arr = np.array(raw_x)[si], np.array(raw_y)[si]
        interpolator = PchipInterpolator(x_arr, y_arr)

        interval_endpoints = np.unique(np.concatenate([x_arr, key_logs]))
        interval_endpoints = interval_endpoints[
            (interval_endpoints >= x_arr.min()) & (interval_endpoints <= x_arr.max())
        ]
        interval_endpoints.sort()

        dense_x_list = []
        for j in range(len(interval_endpoints) - 1):
            interval_pts = np.linspace(interval_endpoints[j], interval_endpoints[j + 1], 10)
            if j < len(interval_endpoints) - 2:
                dense_x_list.append(interval_pts[:-1])
            else:
                dense_x_list.append(interval_pts)

        dense_x = np.concatenate(dense_x_list) if dense_x_list else x_arr
        dense_y = interpolator(dense_x)
        pts = [physics_scaler.transform(np.array([[dx, dy]]))[0] for dx, dy in zip(dense_x, dense_y)]

        if pts:
            pts_np = np.stack(pts)
            samples.append(
                {
                    "static": torch.tensor(X_matrix[i], dtype=torch.float32),
                    "points": torch.tensor(pts_np, dtype=torch.float32),
                    "group": df.iloc[i]["Protein_type"],
                    "id": df.iloc[i]["ID"],
                }
            )

    return samples, X_matrix.shape[1], physics_scaler, protected_indices


def _build_ctx_tensor(task_samples, indices, device):
    """Build a context tensor [1, N_points, 2+static_dim] from sample indices."""
    ctx_items = []
    for i in indices:
        s = task_samples[i]
        stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
        ctx_items.append(torch.cat([s["points"], stat], dim=1))
    return torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)


def _build_tgt_tensors(task_samples, indices, device):
    """Build query tensors for target samples."""
    shear_list, y_list, stat_list = [], [], []
    for i in indices:
        s = task_samples[i]
        n = s["points"].shape[0]
        shear_list.append(s["points"][:, [0]])
        y_list.append(s["points"][:, [1]])
        stat_list.append(s["static"].unsqueeze(0).repeat(n, 1))
    if not shear_list:
        return None, None, None
    q_x = torch.cat(shear_list, dim=0).unsqueeze(0).to(device)
    q_stat = torch.cat(stat_list, dim=0).unsqueeze(0).to(device)
    q_y = torch.cat(y_list, dim=0).unsqueeze(0).to(device)
    return q_x, q_stat, q_y


def compute_viscosity_weights(
    qy_scaled,
    visc_mean,
    visc_scale,
    threshold=2.0,
    max_weight=4.0,
    steepness=3.0,
):
    """
    Compute per-point loss weights based on viscosity magnitude.

    Points above `threshold` (log10 cP) are upweighted via a smooth sigmoid
    ramp. Weights are mean-normalised to 1.0 so the total loss scale is
    preserved and the balance between all loss terms stays the same as in the
    unweighted baseline.
    """
    log_visc = qy_scaled * visc_scale + visc_mean
    raw_w = 1.0 + (max_weight - 1.0) * torch.sigmoid(steepness * (log_visc - threshold))
    normalised_w = raw_w / (raw_w.mean() + 1e-8)
    return normalised_w.detach()
