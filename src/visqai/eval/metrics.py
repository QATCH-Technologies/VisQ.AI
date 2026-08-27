from __future__ import annotations

import numpy as np
import pandas as pd

from visqai.constants import PRED_COLS, VISC_COLS


def calc_metrics(true, pred) -> dict:
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
    t, p = t[mask], p[mask]
    if t.size == 0:
        return dict(
            mae=np.nan,
            mape=np.nan,
            rmse=np.nan,
            r2=np.nan,
            log_mae=np.nan,
            log_rmse=np.nan,
            log_bias=np.nan,
            within_2x=np.nan,
            n=0,
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
    pred_all, true_all = [], []
    for pc, vc in zip(PRED_COLS, VISC_COLS):
        if pc in results_df.columns and vc in truth_df.columns:
            pred_all.append(results_df[pc].values)
            true_all.append(truth_df[vc].values)
    if not pred_all:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "mae_log10": np.nan,
            "rmse_log10": np.nan,
        }

    pred = np.concatenate(pred_all)
    true = np.concatenate(true_all)

    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mape = float(np.mean(np.abs((true - pred) / np.clip(true, 1e-6, None)))) * 100.0

    t_log = _log10_safe(true)
    p_log = _log10_safe(pred)
    mae_log10 = float(np.mean(np.abs(t_log - p_log)))
    rmse_log10 = float(np.sqrt(np.mean((t_log - p_log) ** 2)))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mae_log10": mae_log10,
        "rmse_log10": rmse_log10,
    }


AGGREGATE_MDE: float = 0.017
PER_FOLD_RUN_SD: float = 0.0185
PER_FOLD_RUN_SD_PROVENANCE: str = (
    "5-seed re-run, 5 charge-unchanged 'control' proteins (belatacept, bsa, etanercept, "
    "poly-higg, vudalimab), pooled within-condition across-seed sd of zero_shot_log_mae. "
    "See artifacts/logo_multiseed_report.json / artifacts/fv_charge_followup_report.md Task C."
)


def check_against_noise_band(
    delta: float, *, scope: str, is_difference: bool, n_seeds: int = 1
) -> bool:
    if scope == "aggregate":
        return abs(delta) <= AGGREGATE_MDE
    if scope != "per_fold":
        raise ValueError(f"scope must be 'per_fold' or 'aggregate', got {scope!r}")
    if is_difference:
        band = PER_FOLD_RUN_SD * (2.0 / n_seeds) ** 0.5
    else:
        band = PER_FOLD_RUN_SD
    return abs(delta) <= band
