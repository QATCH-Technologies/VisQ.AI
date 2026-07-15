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
}
# Groups excluded from contrastive/consistency sampling -- they don't
# represent a specific protein type and would produce trivially easy
# negative examples.
NON_PROTEIN_GROUPS = {"none"}

KEY_SHEARS = [100.0, 1000.0, 10000.0, 100000.0, 15000000.0]


def load_and_preprocess(csv_path, save_dir=None):
    df = pd.read_csv(csv_path)

    df, num_cols, cat_cols = build_feature_frame(df)
    protected_indices = protected_feature_indices(num_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    X_matrix = preprocessor.fit_transform(df)
    if np.isnan(X_matrix).any():
        print("WARNING: NaNs found in X_matrix after preprocessing! Replacing with 0.")
        X_matrix = np.nan_to_num(X_matrix)

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
