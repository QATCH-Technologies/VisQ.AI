"""
learning_curve.py
==================
Replays the optimal (and a random-baseline) ibalizumab sample-addition order
through ViscosityPredictorCNP, recording convergence metrics at every step,
then renders the full plot suite (linear, MAPE, log10-with-CI, shape).

Argparse'd replacement for ml/cnp_mk2/learning_curve_ibal.py's main() --
the original had NO argparse, just a "CONFIG ← edit these paths" block of
module-level constants meant to be hand-edited before every run. That's
rewritten here as CLI flags, on the argparse template from cli/parity_eval.py
(ibal_parity_test.py originally had the best CLI ergonomics of the three).
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from visqai.eval.convergence_replay import encode_context, run_convergence_replay
from visqai.eval.data_prep import prepare_df
from visqai.eval.plotting.convergence import (
    plot_convergence,
    plot_log_convergence,
    plot_mape,
    plot_shape_convergence,
)
from visqai.eval.predictor_harness import has_nan_weights
from visqai.inference.predictor import ViscosityPredictorCNP
from visqai.logging_config import configure_logging

warnings.filterwarnings("ignore")

logger = logging.getLogger("LearningCurve")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Replay a sample-addition order through ViscosityPredictorCNP and plot the learning curve."
    )
    ap.add_argument("--model_dir", default=Path("models/experiments/o_net_no_ibal_rung2"))
    ap.add_argument(
        "--ibal_csv",
        default=Path("data/processed/ibal_eval.csv"),
        help="Held-out evaluation CSV (e.g. ibalizumab samples).",
    )
    ap.add_argument(
        "--order_csv",
        default=Path("data/processed/context_selection.csv"),
        help="CSV with a Sample_ID column giving the addition order.",
    )
    ap.add_argument(
        "--pretrain_csv",
        default=Path("data/processed/no_ibal.csv"),
        help="Optional cross-protein context pool to encode as a zero-shot prior.",
    )
    ap.add_argument(
        "--output_dir", default=Path("models/experiments/o_net_no_ibal_rung2/benchmarks")
    )
    ap.add_argument("--n_draws", type=int, default=20)
    ap.add_argument("--k_context", type=int, default=8)
    ap.add_argument("--max_ctx_pool", type=int, default=15)
    ap.add_argument("--n_unc_samples", type=int, default=30)
    ap.add_argument("--plot_step_profiles", action="store_true", default=True)
    ap.add_argument("--no-plot-step-profiles", dest="plot_step_profiles", action="store_false")
    ap.add_argument("--profile_max_steps", type=int, default=8, help="-1 for unlimited.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--also-random-baseline",
        action="store_true",
        default=True,
        help="Also replay a shuffled order as a baseline (default on).",
    )
    ap.add_argument("--no-random-baseline", dest="also_random_baseline", action="store_false")
    return ap.parse_args(argv)


def _load_order_ids(order_csv) -> list:
    """Load the sample-addition order from `order_csv`. Accepts either:

    - a `Sample_ID` column (the original convergence-replay order format,
      e.g. `optimal_order_summary.csv`'s Step/Sample_ID/... columns), used
      in file order; or
    - an `ID` column (the format cli/parity_eval.py's context-selection save
      produces, and what a plain filtered formulation-data CSV has), sorted
      by a `cnp_rank`/`rank` column first if present, otherwise used in
      file order.
    """
    order_df = pd.read_csv(order_csv)
    if "Sample_ID" in order_df.columns:
        return order_df["Sample_ID"].tolist()
    if "ID" in order_df.columns:
        rank_col = (
            "cnp_rank"
            if "cnp_rank" in order_df.columns
            else ("rank" if "rank" in order_df.columns else None)
        )
        if rank_col:
            order_df = order_df.sort_values(rank_col)
        return order_df["ID"].tolist()
    raise KeyError(
        f"{order_csv} has neither a 'Sample_ID' nor an 'ID' column (found: {list(order_df.columns)}) "
        "-- can't determine the sample-addition order."
    )


def _init_clean_predictor(model_dir, pretrain_df, n_draws, k_context, max_ctx_pool):
    logger.info(f"Initializing clean model from: {model_dir}")
    pred = ViscosityPredictorCNP(model_dir)
    if pretrain_df is not None:
        logger.info(
            f"  Encoding {len(pretrain_df)} cross-protein context samples as zero-shot prior (encode-only, no weight updates)..."
        )
        encode_context(
            pred, pretrain_df, n_draws=n_draws, k_context=k_context, max_ctx_pool=max_ctx_pool
        )
        if has_nan_weights(pred):
            logger.error("NaN weights detected — reloading clean model.")
            pred = ViscosityPredictorCNP(model_dir)
        else:
            logger.info("  Cross-protein prior encoded successfully.")
    return pred


def _run_and_plot(args, ibal_df, ids, order_dir, csv_path, prefix):
    profile_max_steps = None if args.profile_max_steps < 0 else args.profile_max_steps
    predictor = _init_clean_predictor(
        args.model_dir, args.pretrain_df, args.n_draws, args.k_context, args.max_ctx_pool
    )
    results_df = run_convergence_replay(
        predictor,
        ibal_df,
        ids,
        n_draws=args.n_draws,
        k_context=args.k_context,
        max_ctx_pool=args.max_ctx_pool,
        n_unc_samples=args.n_unc_samples,
        order_dir=order_dir,
        plot_step_profiles=args.plot_step_profiles,
        profile_max_steps=profile_max_steps,
    )
    results_df.to_csv(csv_path, index=False)
    logger.info(f"  Metrics saved: {csv_path}")

    plot_convergence(results_df, args.output_dir, prefix=prefix)
    plot_mape(results_df, args.output_dir, prefix=prefix)
    plot_log_convergence(results_df, args.output_dir, prefix=prefix)
    plot_shape_convergence(results_df, args.output_dir, prefix=prefix)
    return results_df


def _log_summary(name, df):
    valid = df.dropna(subset=["mae", "rmse", "mape"])
    if valid.empty:
        return
    best_rmse = valid.loc[valid["rmse"].idxmin()]
    best_mae = valid.loc[valid["mae"].idxmin()]
    best_mape = valid.loc[valid["mape"].idxmin()]
    best_rmse_log = valid.loc[valid["rmse_log10"].idxmin()]
    logger.info(
        f"\n{'='*65}\n"
        f"  {name} Best RMSE      : {best_rmse['rmse']:.3f} cP  @ n={best_rmse['n_context']}  ({best_rmse['sample_id']} added)\n"
        f"  {name} Best MAE       : {best_mae['mae']:.3f} cP  @ n={best_mae['n_context']}  ({best_mae['sample_id']} added)\n"
        f"  {name} Best MAPE      : {best_mape['mape']:.2f}%  @ n={best_mape['n_context']}  ({best_mape['sample_id']} added)\n"
        f"  {name} Best RMSE(log) : {best_rmse_log['rmse_log10']:.4f}  @ n={best_rmse_log['n_context']}  ({best_rmse_log['sample_id']} added)\n"
        f"{'='*65}"
    )


def main(argv=None):
    configure_logging()
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    pretrain_df = None
    if args.pretrain_csv and os.path.exists(args.pretrain_csv):
        logger.info(f"Loading pre-training context pool: {args.pretrain_csv}")
        pretrain_df = prepare_df(pd.read_csv(args.pretrain_csv), drop_bad_rows=True)
        logger.info(f"  {len(pretrain_df)} valid pre-training samples.")
    args.pretrain_df = pretrain_df

    logger.info(f"Loading ibalizumab data: {args.ibal_csv}")
    ibal_df = prepare_df(pd.read_csv(args.ibal_csv, index_col=False), drop_bad_rows=True)
    logger.info(f"  {len(ibal_df)} valid ibalizumab samples.")

    logger.info(f"Loading optimal order: {args.order_csv}")
    optimal_ids = _load_order_ids(args.order_csv)

    logger.info("\n" + "=" * 55)
    logger.info("RUNNING EVALUATION: OPTIMAL ORDER")
    logger.info("=" * 55)
    results_opt_df = _run_and_plot(
        args,
        ibal_df,
        optimal_ids,
        order_dir=os.path.join(args.output_dir, "profiles_optimal"),
        csv_path=os.path.join(args.output_dir, "optimal_convergence_metrics.csv"),
        prefix="optimal_",
    )

    results_rand_df = None
    if args.also_random_baseline:
        random_ids = optimal_ids.copy()
        np.random.shuffle(random_ids)
        logger.info("\n" + "=" * 55)
        logger.info("RUNNING EVALUATION: RANDOM ORDER")
        logger.info("=" * 55)
        results_rand_df = _run_and_plot(
            args,
            ibal_df,
            random_ids,
            order_dir=os.path.join(args.output_dir, "profiles_random"),
            csv_path=os.path.join(args.output_dir, "random_convergence_metrics.csv"),
            prefix="random_",
        )

    _log_summary("OPTIMAL", results_opt_df)
    if results_rand_df is not None:
        _log_summary("RANDOM", results_rand_df)

    logger.info("Done.")


if __name__ == "__main__":
    main()
