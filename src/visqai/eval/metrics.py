"""
metrics.py
==========
Two metric entrypoints, moved from two different files. They are NOT
interchangeable -- diffing them surfaced a real masking difference, so they
are kept separate rather than one delegating to the other:

- calc_metrics(true, pred): array-level. From ibal_parity_test.py. MASKS to
  finite, strictly-positive (true>0, pred>0) pairs before computing anything
  -- non-finite/non-positive points are dropped entirely (excluded from N).
  Returns the richer set: mae/mape/rmse/r2/log_mae/log_rmse/log_bias/
  within_2x/n.

- compute_metrics(results_df, truth_df): dataframe-level, scoped to
  eval.constants.PRED_COLS/VISC_COLS. From learning_curve_ibal.py. Does NOT
  mask non-positive/non-finite values out -- it clips them to a 1e-6 floor
  before taking log10 (via _log10_safe) and includes them in the mean. If
  any prediction/truth pair is non-positive, calc_metrics and compute_metrics
  will disagree on N and on every derived statistic. Callers that need the
  scoped, learning-curve-specific 5-key summary (mae/rmse/mape/mae_log10/
  rmse_log10) should keep using compute_metrics; callers that need the fuller
  per-shear breakdown (parity plots, R², within-2x) should use calc_metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from visqai.eval.constants import PRED_COLS, VISC_COLS


def calc_metrics(true, pred) -> dict:
    """Linear-cP MAE/MAPE/RMSE/R2 plus log10 MAE/RMSE/bias/within-2x.
    Masks to finite, strictly-positive (true>0, pred>0) pairs."""
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
    t, p = t[mask], p[mask]
    if t.size == 0:
        return dict(
            mae=np.nan, mape=np.nan, rmse=np.nan, r2=np.nan,
            log_mae=np.nan, log_rmse=np.nan, log_bias=np.nan,
            within_2x=np.nan, n=0,
        )
    mae = float(np.mean(np.abs(t - p)))
    mape = float(np.mean(np.abs((t - p) / np.clip(t, 1e-6, None))) * 100)
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    log_err = np.log10(p) - np.log10(t)
    return dict(
        mae=mae,
        mape=mape,
        rmse=rmse,
        r2=r2,
        log_mae=float(np.mean(np.abs(log_err))),
        log_rmse=float(np.sqrt(np.mean(log_err**2))),
        log_bias=float(np.mean(log_err)),
        within_2x=float(np.mean(np.abs(log_err) < np.log10(2)) * 100),
        n=int(t.size),
    )


def _log10_safe(arr: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(arr, 1e-6, None))


def compute_metrics(results_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    """Linear-cP and log10-space metrics scoped to eval.constants.PRED_COLS/
    VISC_COLS (rmse_log10/mae_log10 match the model's training objective).
    Does not mask non-positive values -- see module docstring."""
    pred_all, true_all = [], []
    for pc, vc in zip(PRED_COLS, VISC_COLS):
        if pc in results_df.columns and vc in truth_df.columns:
            pred_all.append(results_df[pc].values)
            true_all.append(truth_df[vc].values)
    if not pred_all:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "mae_log10": np.nan, "rmse_log10": np.nan}

    pred = np.concatenate(pred_all)
    true = np.concatenate(true_all)

    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mape = float(np.mean(np.abs((true - pred) / np.clip(true, 1e-6, None)))) * 100.0

    t_log = _log10_safe(true)
    p_log = _log10_safe(pred)
    mae_log10 = float(np.mean(np.abs(t_log - p_log)))
    rmse_log10 = float(np.sqrt(np.mean((t_log - p_log) ** 2)))

    return {"mae": mae, "rmse": rmse, "mape": mape, "mae_log10": mae_log10, "rmse_log10": rmse_log10}
