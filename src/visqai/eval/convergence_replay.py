"""
convergence_replay.py
=======================
Step-by-step context-addition replay: at each step, encode the growing
context set and record held-out metrics + uncertainty + shape fidelity.

Moved from ml/cnp_mk2/learning_curve_ibal.py. The module-level "CONFIG ← edit
these paths" constants (N_DRAWS, K_CONTEXT, MAX_CTX_POOL, N_UNC_SAMPLES,
PLOT_STEP_PROFILES, PROFILE_MAX_STEPS) are now function parameters with the
same defaults, rather than hardcoded globals meant to be hand-edited --
visqai.cli.learning_curve exposes them as argparse flags.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from visqai.eval.metrics import compute_metrics
from visqai.eval.shape_metrics import _aggregate_shape
from visqai.eval.predictor_harness import reset_memory, has_nan_weights
from visqai.eval.plotting.convergence import predict_profiles, plot_sample_profile

logger = logging.getLogger(__name__)

_NULL_SHAPE_AGG = {
    "shape_rmse_log10": np.nan,
    "slope_sign_agree": np.nan,
    "plateau_err_log10": np.nan,
    "thin_ratio_log_err": np.nan,
}


def encode_context(
    predictor,
    context_df: pd.DataFrame,
    learn_steps: int = 50,  # kept for API compat with predictor.learn — ignored by the encode-only engine
    learn_lr: float = 1e-3,  # kept for API compat with predictor.learn — ignored by the encode-only engine
    n_draws: int = 20,
    k_context: int = 8,
    max_ctx_pool: int = 15,
):
    """Reset memory, apply diverse context selection (if the engine exposes
    it), then encode with multi-draw averaging."""
    reset_memory(predictor)
    if context_df.empty:
        return

    if hasattr(predictor, "_select_diverse_context"):
        context_df = predictor._select_diverse_context(context_df, max_k=max_ctx_pool)

    predictor.learn(context_df, steps=learn_steps, lr=learn_lr, n_draws=n_draws, k=k_context)


def _shape_index_row(step, sid, role, n_context, prof):
    """Build one _index.csv row including shape metrics for triage/sorting."""
    return {
        "step": step,
        "sample_id": sid,
        "role": role,
        "n_context": n_context,
        "conc": prof.get("conc", np.nan),
        "pH": prof.get("pH", np.nan),
        "rmse_log10": prof.get("rmse_log10", np.nan),
        "shape_rmse_log10": prof.get("shape_rmse_log10", np.nan),
        "slope_sign_agree": prof.get("slope_sign_agree", np.nan),
        "slope_n_sloped": prof.get("slope_n_sloped", 0),
        "plateau_err_log10": prof.get("plateau_err_log10", np.nan),
        "thin_ratio_log_err": prof.get("thin_ratio_log_err", np.nan),
    }


def render_step_profiles(predictor, ibal_df, id_to_idx, context_ids, holdout_ids, step, order_dir):
    """At a single replay step, render predicted-vs-actual profiles for every
    sample, split into context/ and heldout/ subdirectories. The model's
    encoded memory must already reflect `context_ids` (the caller encodes
    before calling this)."""
    step_dir = os.path.join(order_dir, f"step_{step:02d}")
    ctx_dir = os.path.join(step_dir, "context")
    hold_dir = os.path.join(step_dir, "heldout")

    index_rows = []

    if context_ids:
        ctx_idx = [id_to_idx[s] for s in context_ids if s in id_to_idx]
        ctx_df = ibal_df.loc[ctx_idx].copy()
        ctx_profs = predict_profiles(predictor, ctx_df)
        for sid, prof in ctx_profs.items():
            plot_sample_profile(sid, prof, os.path.join(ctx_dir, f"{sid}.png"), is_context=True)
            index_rows.append(_shape_index_row(step, sid, "context", len(context_ids), prof))

    if holdout_ids:
        hold_idx = [id_to_idx[s] for s in holdout_ids if s in id_to_idx]
        hold_df = ibal_df.loc[hold_idx].copy()
        hold_profs = predict_profiles(predictor, hold_df)
        for sid, prof in hold_profs.items():
            plot_sample_profile(sid, prof, os.path.join(hold_dir, f"{sid}.png"), is_context=False)
            index_rows.append(_shape_index_row(step, sid, "heldout", len(context_ids), prof))

    if index_rows:
        os.makedirs(step_dir, exist_ok=True)
        pd.DataFrame(index_rows).to_csv(os.path.join(step_dir, "_index.csv"), index=False)

    n_ctx = len(context_ids) if context_ids else 0
    n_hold = len(holdout_ids) if holdout_ids else 0
    logger.info(f"  [profiles] step {step:>2}: {n_ctx} context + {n_hold} held-out profiles -> {step_dir}")


def run_convergence_replay(
    predictor,
    ibal_df: pd.DataFrame,
    ordered_ids: list,
    n_draws: int = 20,
    k_context: int = 8,
    max_ctx_pool: int = 15,
    n_unc_samples: int = 30,
    learn_steps: int = 50,
    learn_lr: float = 1e-3,
    order_dir: str = None,
    plot_step_profiles: bool = True,
    profile_max_steps: int | None = 8,
) -> pd.DataFrame:
    """Adds ibalizumab samples one-by-one in `ordered_ids` order. At each
    step: encode context, predict holdout, record metrics + uncertainty +
    shape fidelity.

    Returns DataFrame with columns: step, sample_id, n_context, mae, rmse,
    mape, mae_log10, rmse_log10, std_log10, plus shape-fidelity columns.
    """
    id_to_idx = {str(row["ID"]): idx for idx, row in ibal_df.iterrows()}
    ordered_ids = [str(sid) for sid in ordered_ids if str(sid) in id_to_idx]
    if not ordered_ids:
        raise ValueError(
            "No `ordered_ids` matched an ID in `ibal_df` — check that the order CSV's "
            "sample IDs line up with ibal_df's ID column."
        )

    records = []
    null_metrics = {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "mae_log10": np.nan, "rmse_log10": np.nan}

    logger.info(f"Replaying {len(ordered_ids)} samples (plus 0-shot baseline)...")

    # Step 0: zero-shot baseline.
    reset_memory(predictor)
    all_holdout_idx = [id_to_idx[s] for s in ordered_ids]
    holdout_df_0 = ibal_df.loc[all_holdout_idx].copy()

    metrics_0 = null_metrics.copy()
    std_log10_0 = np.nan
    shape_agg_0 = dict(_NULL_SHAPE_AGG)
    if not has_nan_weights(predictor):
        try:
            drop_targets = [c for c in holdout_df_0.columns if c.startswith("Viscosity_")]
            query_0 = holdout_df_0.drop(columns=drop_targets, errors="ignore")
            results_0 = predictor.predict(query_0)
            metrics_0 = compute_metrics(results_0, holdout_df_0)
            _, unc_0 = predictor.predict_with_uncertainty(query_0, n_samples=n_unc_samples, k=k_context)
            std_log10_0 = float(np.mean(unc_0.get("std_log10", [np.nan])))
            profs_0 = predict_profiles(predictor, holdout_df_0)
            shape_agg_0 = _aggregate_shape(profs_0)
        except Exception as e:
            logger.warning(f"  Step 0: predict failed — {e}")

    records.append({"step": 0, "sample_id": "None", "n_context": 0, **metrics_0, "std_log10": std_log10_0, **shape_agg_0})
    logger.info(
        f"  [ 0/{len(ordered_ids)}] 0-Shot Baseline | Holdout MAE={metrics_0['mae']:.3f} cP  "
        f"RMSE={metrics_0['rmse']:.3f} cP  RMSE(log10)={metrics_0['rmse_log10']:.4f}  "
        f"MAPE={metrics_0['mape']:.2f}%  std(log10)={std_log10_0:.4f}"
    )

    if plot_step_profiles and order_dir is not None:
        try:
            render_step_profiles(predictor, ibal_df, id_to_idx, context_ids=[], holdout_ids=ordered_ids, step=0, order_dir=order_dir)
        except Exception as e:
            logger.warning(f"  Step 0: profile render failed — {e}")

    for step, sample_id in enumerate(ordered_ids, start=1):
        context_ids = ordered_ids[:step]
        holdout_ids = ordered_ids[step:]

        context_idx = [id_to_idx[s] for s in context_ids]
        context_df = ibal_df.loc[context_idx].copy()

        encode_context(
            predictor, context_df, learn_steps=learn_steps, learn_lr=learn_lr,
            n_draws=n_draws, k_context=k_context, max_ctx_pool=max_ctx_pool,
        )

        metrics = null_metrics.copy()
        std_log10 = np.nan

        if holdout_ids and not has_nan_weights(predictor):
            holdout_idx = [id_to_idx[s] for s in holdout_ids]
            holdout_df = ibal_df.loc[holdout_idx].copy()
            drop_targets_n = [c for c in holdout_df.columns if c.startswith("Viscosity_")]
            query_df = holdout_df.drop(columns=drop_targets_n, errors="ignore")
            try:
                results_df = predictor.predict(query_df)
                metrics = compute_metrics(results_df, holdout_df)
            except Exception as e:
                logger.warning(f"  Step {step}: predict failed — {e}")

            try:
                _, unc_stats = predictor.predict_with_uncertainty(query_df, n_samples=n_unc_samples, k=k_context)
                std_log10 = float(np.mean(unc_stats.get("std_log10", [np.nan])))
            except Exception as e:
                logger.warning(f"  Step {step}: uncertainty failed — {e}")

        shape_agg = dict(_NULL_SHAPE_AGG)
        if holdout_ids and not has_nan_weights(predictor):
            try:
                holdout_idx = [id_to_idx[s] for s in holdout_ids]
                holdout_df = ibal_df.loc[holdout_idx].copy()
                profs = predict_profiles(predictor, holdout_df)
                shape_agg = _aggregate_shape(profs)
            except Exception as e:
                logger.warning(f"  Step {step}: shape agg failed — {e}")

        records.append({"step": step, "sample_id": sample_id, "n_context": step, **metrics, "std_log10": std_log10, **shape_agg})
        logger.info(
            f"  [{step:>2}/{len(ordered_ids)}] Added {sample_id:>6} | Holdout MAE={metrics['mae']:.3f} cP  "
            f"RMSE={metrics['rmse']:.3f} cP  RMSE(log10)={metrics['rmse_log10']:.4f}  MAPE={metrics['mape']:.2f}%  "
            f"shapeRMSE={shape_agg['shape_rmse_log10']:.3f}  slopeMatch={shape_agg['slope_sign_agree']:.0%}"
        )

        if plot_step_profiles and order_dir is not None and (profile_max_steps is None or step <= profile_max_steps):
            try:
                render_step_profiles(predictor, ibal_df, id_to_idx, context_ids=context_ids, holdout_ids=holdout_ids, step=step, order_dir=order_dir)
            except Exception as e:
                logger.warning(f"  Step {step}: profile render failed — {e}")

    return pd.DataFrame(records)
