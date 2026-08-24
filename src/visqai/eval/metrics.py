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


# ---------------------------------------------------------------------------
# Follow-on plan (Fv charge, Tasks C then G) -- noise/threshold constants for
# DIFFERENT kinds of number. Conflating them is exactly what produced the
# predecessor plan's C1 error (AGGREGATE_MDE, derived across ~11-12
# clusters, applied as a per-fold threshold) and then, one level down, the
# follow-on plan's C3 error: PER_FOLD_RUN_SD (formerly misleadingly named
# PER_FOLD_NOISE_SD) is the sd of a SINGLE run under a fixed protein/
# condition/seed -- it is not itself a band, and nothing this module gets
# compared against is a single run. A DIFFERENCE of two such runs has sd
# sigma*sqrt(2) (one seed each) or sigma*sqrt(2/k) (k-seed means each,
# independent noise averaging down as usual). Storing one number labelled
# "the band" silently applied that single-run sd to differences too --
# check_against_noise_band now derives the band at call time instead.
#
# AGGREGATE_MDE: the minimum detectable effect for a claim about many
# clusters at once (e.g. "does this change move the average across all
# proteins"). Unchanged from the predecessor plan. Not derived from
# PER_FOLD_RUN_SD and not on the same footing -- keep them separate rather
# than deriving one from the other (e.g. do not assume AGGREGATE_MDE ==
# PER_FOLD_RUN_SD / sqrt(n_clusters); it wasn't computed that way and pinning
# a relationship between them that isn't actually true is its own version of
# this same conflation risk).
#
# PER_FOLD_RUN_SD: the run-to-run standard deviation of a SINGLE fold's own
# zero_shot_log_mae under a fixed protein/condition, re-trained with a
# different seed. Measured from the 5-seed re-run of the 5 "control"
# proteins in the Fv/whole-antibody charge swap (net_charge bit-for-bit
# unchanged by the swap, so any variation is pure training noise, not a
# real effect): pooled within-condition across-seed sd = 0.0185 (10
# protein-condition cells, analysis/aggregate_logo_multiseed.py ->
# artifacts/logo_multiseed_report.json). Re-derive if the training recipe
# (architecture, epochs, context-sampling) changes materially -- this is an
# empirical measurement of THIS harness's noise, not a universal constant.
#
# IMPORTANT SCOPE LIMIT: PER_FOLD_RUN_SD measures stochasticity in CNP
# TRAINING (visqai.cli.logo_eval, different random seeds -> different
# trained weights). It has NOTHING to do with computations that involve no
# model training at all -- e.g. fv_regime_real_data_eval.py's criterion-3
# LOBO probe is a closed-form linear regression (np.linalg.lstsq) fit on
# fixed data, with zero seed-dependent randomness anywhere in it. Comparing
# that computation's deltas against a training-noise band is the Task C4
# category error: a quantity measuring one source of variation applied to a
# result that has a completely different (in that specific case, zero)
# source of variation. Check whether the number you have came from a
# stochastic process before reaching for this at all.
AGGREGATE_MDE: float = 0.017
PER_FOLD_RUN_SD: float = 0.0185
PER_FOLD_RUN_SD_PROVENANCE: str = (
    "5-seed re-run, 5 charge-unchanged 'control' proteins (belatacept, bsa, etanercept, "
    "poly-higg, vudalimab), pooled within-condition across-seed sd of zero_shot_log_mae. "
    "See artifacts/logo_multiseed_report.json / artifacts/fv_charge_followup_report.md Task C."
)


def check_against_noise_band(delta: float, *, scope: str, is_difference: bool, n_seeds: int = 1) -> bool:
    """Route a delta to the threshold appropriate for its scope, deriving
    the band at call time rather than comparing against a single stored
    number -- the fix for the Task C3 error (PER_FOLD_RUN_SD is a
    single-run sd, not a band).

    scope: 'per_fold' (compare against PER_FOLD_RUN_SD-derived quantities)
        or 'aggregate' (compare against AGGREGATE_MDE, unchanged, always a
        single fixed threshold regardless of is_difference/n_seeds).
    is_difference: True if `delta` is itself a difference between two
        independently-measured stochastic quantities (e.g. after-before);
        False if it's a single measurement's own deviation from a fixed
        reference. Required for scope='per_fold' -- no default, so a
        caller must state which kind of number it has rather than get one
        assumed for it.
    n_seeds: number of seeds each side of the comparison was averaged over
        (only meaningful when is_difference=True). Noise on a k-seed mean
        scales as 1/sqrt(k); the band on a difference of two such means
        scales as PER_FOLD_RUN_SD * sqrt(2/k).

    Raises ValueError if `scope` isn't recognized, or if scope='per_fold'
    and `is_difference` wasn't explicitly passed.
    """
    if scope == "aggregate":
        return abs(delta) <= AGGREGATE_MDE
    if scope != "per_fold":
        raise ValueError(f"scope must be 'per_fold' or 'aggregate', got {scope!r}")
    if is_difference:
        band = PER_FOLD_RUN_SD * (2.0 / n_seeds) ** 0.5
    else:
        band = PER_FOLD_RUN_SD
    return abs(delta) <= band
