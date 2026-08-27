"""
data.py
=======
Training-time data loading: load_and_preprocess builds a fitted
ColumnTransformer + physics_scaler and resamples each formulation's viscosity
curve into a fixed set of key-shear-rate points via PCHIP interpolation.

Moved from ml/cnp_mk2/train_o_net_v4_rung1.py. The row-level feature
engineering (categorical/charge featurization, unit conversion, physics
priors) now goes through visqai.features.dataprocessor.build_feature_frame --
previously this logic was duplicated inline here and, independently and
divergently, in inference_o_net.py::_preprocess (see dataprocessor.py's module
docstring for the bug that duplication caused).

Not carried forward: the original wrote `df.to_csv("pembro_data.csv")` as a
debug side effect on every call (writes a fixed filename into whatever the
current working directory happens to be). That was leftover debug output,
not documented behavior -- dropped rather than ported.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from visqai import paths
from visqai.constants import DESCRIPTOR_OOD_CLIP_SIGMA, SHEAR_MAP
from visqai.features.dataprocessor import build_feature_frame, protected_feature_indices

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

NON_PROTEIN_GROUPS = {"none"}

KEY_SHEARS = [100.0, 1000.0, 10000.0, 100000.0, 15000000.0]
ZERO_VARIANCE_FALLBACK_SCALE: float = 10.0


def _fix_zero_variance_scale(
    preprocessor: ColumnTransformer, x_matrix: np.ndarray, num_cols: list[str]
) -> np.ndarray:
    """Patch zero-variance numeric features with a fixed fallback scale.

    Replaces the unit scale assigned by :class:`sklearn.preprocessing.StandardScaler`
    to numeric columns with effectively zero training variance. The same
    fallback scale is applied to the already-transformed feature matrix and
    persisted into the fitted scaler so subsequent transformations use the
    same bounded behavior.

    Args:
        preprocessor: Fitted column transformer containing the numeric
            :class:`~sklearn.preprocessing.StandardScaler` under the `"num"`
            transformer.
        x_matrix: Feature matrix produced by the preprocessor's
            `fit_transform` call. Zero-variance columns are modified in
            place to reflect the fallback scale.
        num_cols: Names of the numeric columns, in the same order as the
            fitted numeric transformer. Used for diagnostic reporting.

    Returns:
        The corrected feature matrix. If no numeric columns have effectively
        zero variance, the original matrix is returned unchanged.

    Notes:
        Scikit-learn uses `scale_=1` for zero-variance features. For
        engineered features that are constant in a training fold but may
        receive non-constant values at inference time, that behavior can
        allow raw-unit magnitudes to pass through with insufficient scaling.
        Applying `ZERO_VARIANCE_FALLBACK_SCALE` provides a fixed,
        data-independent bound for those features.
    """
    scaler = preprocessor.named_transformers_["num"]
    zero_var = np.where(scaler.var_ <= 1e-12)[0]
    if len(zero_var) == 0:
        return x_matrix
    print(
        f"  [load_and_preprocess] zero-variance training columns "
        f"{[num_cols[i] for i in zero_var]} -> using fixed a-priori scale "
        f"{ZERO_VARIANCE_FALLBACK_SCALE} instead of sklearn's degenerate scale_=1."
    )
    for i in zero_var:
        x_matrix[:, i] = x_matrix[:, i] / ZERO_VARIANCE_FALLBACK_SCALE
        scaler.scale_[i] = ZERO_VARIANCE_FALLBACK_SCALE
    return x_matrix


def _drop_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove malformed rows with no formulation-defining fields.

    Drops rows that contain neither `Protein_type` nor `Protein_conc`.
    Such rows are treated as blank or malformed export records rather than
    valid formulations and must be removed before feature construction and
    preprocessor fitting so they cannot contaminate fitted feature statistics.

    Args:
        df: Raw formulation dataframe.

    Returns:
        A dataframe containing only rows with a non-null `Protein_type` or
        `Protein_conc` value. The returned dataframe is reindexed from
        zero.
    """
    return df[df["Protein_type"].notna() | df["Protein_conc"].notna()].reset_index(drop=True)


def load_and_preprocess(
    csv_path: str | Path, save_dir: str | Path | None = None
) -> tuple[list[dict], int, StandardScaler, list[int]]:
    """Load formulation data and build the fitted training representation.

    Loads formulation viscosity measurements, constructs the shared static
    feature representation, fits the static-feature and physics scalers, and
    converts each formulation's measured viscosity curve into a dense,
    fixed-format set of physics-scaled training points using PCHIP
    interpolation.

    The fitted preprocessing artifacts can optionally be persisted for use
    by inference. Static features are generated through
    :func:`visqai.features.dataprocessor.build_feature_frame` so training and
    inference share the same feature-engineering implementation.

    Args:
        csv_path: Path to the input formulation table.
        save_dir: Optional directory in which to save the fitted
            `preprocessor.pkl`, `physics_scaler.pkl`, and
            `protected_indices.pkl` artifacts. If `None`, artifacts are
            not persisted.

    Returns:
        A tuple containing:

        * `samples`: List of formulation samples. Each sample contains the
          transformed static features, physics-scaled curve points, protein
          group, and source identifier.
        * `static_dim`: Number of columns in the transformed static-feature
          matrix.
        * `physics_scaler`: Fitted
          :class:`sklearn.preprocessing.StandardScaler` for log10 shear rate
          and log10 viscosity.
        * `protected_indices`: Indices of static features protected by the
          feature-processing pipeline.

    Notes:
        Formulations with fewer than three valid viscosity measurements are
        excluded from the returned training samples because PCHIP
        interpolation requires sufficient distinct support points.

        Non-positive viscosity values are replaced with a small positive
        floor before logarithmic transformation. Static numeric features are
        protected against zero-variance scaling, NaN values, and descriptor
        out-of-distribution magnitudes before being used by the model.
    """
    df = paths.load_table(csv_path)
    df = _drop_blank_rows(df)

    df, num_cols, cat_cols = build_feature_frame(df)
    protected_indices = protected_feature_indices(num_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    x_matrix = np.asarray(preprocessor.fit_transform(df))

    x_matrix = _fix_zero_variance_scale(preprocessor, x_matrix, num_cols)
    if np.isnan(x_matrix).any():
        print("WARNING: NaNs found in x_matrix after preprocessing! Replacing with 0.")
        x_matrix = np.nan_to_num(x_matrix)
    x_matrix[:, : len(num_cols)] = np.clip(
        x_matrix[:, : len(num_cols)], -DESCRIPTOR_OOD_CLIP_SIGMA, DESCRIPTOR_OOD_CLIP_SIGMA
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

    # Target resampling: PCHIP-interpolate each curve, densely sample it,
    # and re-express every point relative to the fixed KEY_SHEARS grid.
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
        pts = [
            physics_scaler.transform(np.array([[dx, dy]]))[0] for dx, dy in zip(dense_x, dense_y)
        ]

        if pts:
            pts_np = np.stack(pts)
            samples.append(
                {
                    "static": torch.tensor(x_matrix[i], dtype=torch.float32),
                    "points": torch.tensor(pts_np, dtype=torch.float32),
                    "group": df.iloc[i]["Protein_type"],
                    "id": df.iloc[i]["ID"],
                }
            )

    return samples, x_matrix.shape[1], physics_scaler, protected_indices


def _build_ctx_tensor(
    task_samples: list[dict], indices: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Construct the context tensor for selected formulation samples.

    Expands each formulation's static feature vector across all of its
    physics-scaled measurement points and concatenates those features with
    the corresponding scaled log10 shear and viscosity values.

    Args:
        task_samples: Collection of preprocessed formulation samples, where
            each sample provides `"points"` and `"static"` tensors.
        indices: Indices of the samples to include in the context set.
        device: Torch device on which the resulting tensor should reside.

    Returns:
        A tensor of shape `[1, N, 2 + static_dim]`, where `N` is the
        total number of context measurement points across the selected
        formulations. The leading dimension is the batch dimension expected
        by the CNP encoder.
    """
    ctx_items = []
    for i in indices:
        s = task_samples[i]
        stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
        ctx_items.append(torch.cat([s["points"], stat], dim=1))
    return torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)


def _build_tgt_tensors(
    task_samples: list[dict], indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
    """Construct query tensors for selected target formulations.

    Collects the physics-scaled shear values, viscosity targets, and repeated
    static feature vectors for the requested target samples into tensors
    suitable for CNP prediction and loss computation.

    Args:
        task_samples: Collection of preprocessed formulation samples, where
            each sample provides `"points"` and `"static"` tensors.
        indices: Indices of the samples to use as targets.
        device: Torch device on which the resulting tensors should reside.

    Returns:
        A tuple `(q_x, q_stat, q_y)` containing:

        * `q_x`: Query shear tensor of shape `[1, N, 1]`.
        * `q_stat`: Repeated static-feature tensor of shape
          `[1, N, static_dim]`.
        * `q_y`: Target viscosity tensor of shape `[1, N, 1]`.

        If `indices` is empty, returns `(None, None, None)`.
    """
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
    qy_scaled: torch.Tensor,
    visc_mean: float,
    visc_scale: float,
    threshold: float = 2.0,
    max_weight: float = 4.0,
    steepness: float = 3.0,
) -> torch.Tensor:
    """Compute normalized loss weights from viscosity magnitude.

    Converts scaled log10-viscosity targets back to log10 viscosity, then
    applies a smooth sigmoid weighting function that increases the
    contribution of points above the specified viscosity threshold. The
    resulting weights are normalized to have mean one so introducing the
    weighting does not change the overall scale of the loss.

    Args:
        qy_scaled: Torch tensor containing physics-scaled log10-viscosity
            target values.
        visc_mean: Mean of log10 viscosity used by the physics scaler.
        visc_scale: Scale of log10 viscosity used by the physics scaler.
        threshold: Log10-viscosity value at which the weighting ramp is
            centered.
        max_weight: Maximum asymptotic weight assigned to high-viscosity
            points.
        steepness: Sigmoid steepness controlling how quickly weights increase
            around `threshold`.

    Returns:
        A detached torch tensor of per-point weights with the same shape as
        `qy_scaled` and approximately unit mean.

    Notes:
        The weights are detached from the computation graph so the weighting
        scheme affects optimization of the model parameters without creating
        gradients through the weighting calculation itself.
    """
    log_visc = qy_scaled * visc_scale + visc_mean
    raw_w = 1.0 + (max_weight - 1.0) * torch.sigmoid(steepness * (log_visc - threshold))
    normalised_w = raw_w / (raw_w.mean() + 1e-8)
    return normalised_w.detach()
