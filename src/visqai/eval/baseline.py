"""
baseline.py
===========
Phase 0 reference baseline: a plain feature-only regressor (sklearn
HistGradientBoostingRegressor) trained on the same leave-one-GROUP-out
splits as the CNP, predicting log10(viscosity) from the row's engineered
static features plus log10(shear rate) as an extra input -- long format,
mirroring how the CNP treats shear rate as a query input rather than five
independent targets.

Success rule (per the Phase 0 spec this implements): CNP zero-shot must
reach <= this baseline's held-out log10 MAE; CNP few-shot must beat it.
Feature-only LOGO log10 MAE in the ~0.15-0.17 range is the expected ballpark
for this kind of model on this data -- if a CNP change doesn't move it
toward that line on held-out groups, revert it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from visqai.eval.constants import SHEAR_COLS, SHEAR_RATES
from visqai.eval.logo_splits import LogoGroup, build_groups, zero_ingredient_properties
from visqai.eval.metrics import calc_metrics
from visqai.preprocessing.pipeline import build_feature_frame


def _melt_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (sample, shear rate): every engineered static feature,
    plus log10(shear_rate) as a numeric input and log10(viscosity) as the
    regression target."""
    feat_df, num_cols, cat_cols = build_feature_frame(df)
    blocks = []
    for shear_col, shear_val in zip(SHEAR_COLS, SHEAR_RATES):
        if shear_col not in feat_df.columns:
            continue
        v = pd.to_numeric(feat_df[shear_col], errors="coerce")
        valid = v.notna() & (v > 0)
        if not valid.any():
            continue
        block = feat_df.loc[valid, num_cols + cat_cols].copy()
        block["log10_shear"] = np.log10(shear_val)
        block["_log10_visc"] = np.log10(v[valid].values)
        blocks.append(block)
    if not blocks:
        return pd.DataFrame(), [], []
    return pd.concat(blocks, ignore_index=True), num_cols + ["log10_shear"], cat_cols


def _make_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    model = HistGradientBoostingRegressor(max_depth=6, max_iter=300, learning_rate=0.05, random_state=0)
    return Pipeline([("pre", pre), ("model", model)])


def fit_baseline(train_df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """Fit the reference regressor on train_df. Returns (pipeline, feat_cols)
    so the same fitted pipeline can be reused for both the real held-out
    frame and an ingredient-ablated counterfactual."""
    train_long, num_cols, cat_cols = _melt_long(train_df)
    feat_cols = num_cols + cat_cols
    pipe = _make_pipeline(num_cols, cat_cols)
    pipe.fit(train_long[feat_cols], train_long["_log10_visc"])
    return pipe, feat_cols


def score_baseline(pipe: Pipeline, feat_cols: list[str], held_df: pd.DataFrame) -> dict:
    held_long, _, _ = _melt_long(held_df)
    if held_long.empty:
        return dict(n=0, mae=np.nan, mape=np.nan, rmse=np.nan, r2=np.nan, log_mae=np.nan, log_rmse=np.nan, log_bias=np.nan, within_2x=np.nan)
    pred_log = pipe.predict(held_long[feat_cols])
    true_cp = np.power(10, held_long["_log10_visc"].values)
    pred_cp = np.power(10, pred_log)
    return calc_metrics(true_cp, pred_cp)


def run_baseline_fold(train_df: pd.DataFrame, held_df: pd.DataFrame, group: LogoGroup | None = None) -> dict:
    """Fit+score one LOGO fold. If `group` is an ingredient group, also
    scores the ablation counterfactual (that ingredient's rows re-featurized
    with the property vector zeroed, i.e. as if it were unknown) through the
    SAME fitted pipeline, for the "does the property vector actually buy
    extrapolation" check."""
    pipe, feat_cols = fit_baseline(train_df)
    m = score_baseline(pipe, feat_cols, held_df)
    m["n_train_rows"] = len(train_df)
    m["n_held_rows"] = len(held_df)

    if group is not None and group.axis == "ingredient":
        ablated_held = zero_ingredient_properties(held_df, group)
        m_ablated = score_baseline(pipe, feat_cols, ablated_held)
        m["ablation_log_mae"] = m_ablated["log_mae"]
        m["ablation_delta"] = m_ablated["log_mae"] - m["log_mae"]

    return m


def run_baseline_logo(df: pd.DataFrame, axis: str, min_rows: int = 2, groups=None) -> pd.DataFrame:
    """Run the baseline over every LOGO group for `axis` (or a caller-supplied
    subset via `groups`, for smoke tests). Returns one row per held-out group."""
    fold_groups = groups if groups is not None else build_groups(df, axis, min_rows=min_rows)
    rows = []
    for g in fold_groups:
        train_df, held_df = g.split(df)
        if held_df.empty or train_df.empty:
            continue
        m = run_baseline_fold(train_df, held_df, group=g)
        rows.append({"axis": g.axis, "group": g.key, **m})
    return pd.DataFrame(rows)
