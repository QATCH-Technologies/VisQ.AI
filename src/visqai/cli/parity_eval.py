"""
parity_eval.py
===============
Combined Ibalizumab CNP context experiment: select a small, strategic
context set (greedy forward selection + optional swap refinement), then
evaluate it on the correctly held-out remainder (per-shear metrics, parity
plots, profile plot).

Argparse'd replacement for ml/cnp_mk2/ibal_parity_test.py's main(). This
file's argparse structure was the best-ergonomics CLI in the original
codebase and is the literal template cli/train.py and cli/learning_curve.py
followed. The dynamic __import__/logging.basicConfig-monkeypatch workaround
for reaching ViscosityPredictorCNP is gone -- it existed only because
inference_o_net.py had import-time side effects; visqai.inference.predictor
has none.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch

from visqai.eval.constants import SHEAR_COLS
from visqai.eval.context_selection import greedy_select
from visqai.eval.data_prep import prepare_df
from visqai.eval.metrics import calc_metrics
from visqai.eval.plotting.parity import build_long, make_parity_plot, make_profile_plot
from visqai.inference.predictor import ViscosityPredictorCNP
from visqai.logging_config import configure_logging

warnings.filterwarnings("ignore")

logger = logging.getLogger("ParityEval")

PROTEIN_KEY = "Ibalizumab"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Combined Ibalizumab CNP context select + evaluate.")
    ap.add_argument("--model_dir", default="models/experiments/o_net_no_ibal_rung2")
    ap.add_argument("--data", default=Path("data/processed/ibal_eval.csv"))
    ap.add_argument("--n_select", type=int, default=8)
    ap.add_argument(
        "--objective",
        choices=["mean", "tail", "max"],
        default="tail",
        help="Selection objective. 'tail'/'max' demonstrate range coverage.",
    )
    ap.add_argument(
        "--refine",
        action="store_true",
        default=True,
        help="Run swap-refinement after greedy (default on).",
    )
    ap.add_argument("--no-refine", dest="refine", action="store_false")
    ap.add_argument("--protein_key", default=PROTEIN_KEY)
    ap.add_argument("--out_dir", default="models/experiments/o_net_no_ibal_rung2/benchmarks")
    ap.add_argument(
        "--context_csv",
        default=None,
        help="Skip selection; load context IDs from this CSV (col ID/cnp_rank).",
    )
    ap.add_argument("--plot", action="store_true", default=True)
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    return ap.parse_args(argv)


def main(argv=None):
    configure_logging()
    args = parse_args(argv)

    for path in (args.model_dir, args.data):
        if not os.path.exists(path):
            sys.exit(f"ERROR: path not found — {path}")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # index_col=False: some exports of this CSV carry a couple of trailing
    # unnamed values per row with no matching header field. Without this,
    # pandas silently treats the leading columns as an implicit index and
    # shifts every other column (incl. Protein_type) out from under its header.
    df = prepare_df(pd.read_csv(args.data, index_col=False))
    iba_df = (
        df[df["Protein_type"].astype(str).str.lower() == args.protein_key.lower()].copy().reset_index(drop=True)
    )
    n_total = len(iba_df)
    logger.info(f"{args.protein_key} samples: {n_total}")
    if n_total < args.n_select:
        sys.exit(f"ERROR: only {n_total} samples, cannot select {args.n_select}.")

    logger.info(f"Loading model: {args.model_dir}")
    predictor = ViscosityPredictorCNP(args.model_dir, verbose=False)
    logger.info(
        f"  static_dim={predictor.static_dim}, hidden_dim={predictor.config['hidden_dim']}, latent_dim={predictor.config['latent_dim']}"
    )

    if args.context_csv and os.path.exists(args.context_csv):
        meta = prepare_df(pd.read_csv(args.context_csv))
        rank_col = (
            "cnp_rank"
            if "cnp_rank" in meta.columns
            else ("rank" if "rank" in meta.columns else None)
        )
        if rank_col:
            meta = meta.sort_values(rank_col)
        context_ids = meta["ID"].astype(str).head(args.n_select).tolist()
        logger.info(f"Loaded {len(context_ids)} context IDs from {args.context_csv}")
    else:
        t0 = time.perf_counter()
        selected_idx, step_log, final_score = greedy_select(
            predictor,
            iba_df,
            n_select=args.n_select,
            objective=args.objective,
            refine=args.refine,
            verbose=True,
        )
        context_ids = [str(iba_df.iloc[i]["ID"]) for i in selected_idx]
        logger.info(
            f"Selection complete in {time.perf_counter()-t0:.1f}s (final {args.objective} score={final_score:.5f})"
        )

        sel_path = os.path.join(args.out_dir, "context_selection.csv")
        full_sel = df[df["ID"].isin(context_ids)].copy()
        rank_map = {sid: k + 1 for k, sid in enumerate(context_ids)}
        full_sel["cnp_rank"] = full_sel["ID"].map(rank_map)
        full_sel.sort_values("cnp_rank").to_csv(sel_path, index=False)
        logger.info(f"Context selection saved: {sel_path}")

    context_id_set = set(context_ids)
    n_ctx = len(context_id_set)
    context_df = iba_df[iba_df["ID"].isin(context_id_set)].copy()
    held_out_df = iba_df[~iba_df["ID"].isin(context_id_set)].copy().reset_index(drop=True)
    n_held = len(held_out_df)
    logger.info(f"Context: {n_ctx} | Held-out: {n_held}")

    predictor.memory_vector = None
    predictor.context_t = None
    predictor.learn(context_df)

    results_df = predictor.predict(held_out_df)
    context_pred_df = predictor.predict(context_df)

    long_df = pd.DataFrame(
        build_long(results_df, is_context=False) + build_long(context_pred_df, is_context=True)
    )
    csv_path = os.path.join(args.out_dir, "ibalizumab_parity_results.csv")
    long_df.to_csv(csv_path, index=False)
    logger.info(f"Parity results saved: {csv_path}")

    held_long = long_df[~long_df["is_context"]]
    logger.info("\n" + "=" * 65)
    logger.info(f"PER-SHEAR-RATE SUMMARY  (held-out {n_held} samples, context excluded)")
    logger.info("=" * 65)
    logger.info(
        f"{'Shear Rate':>18}  {'N':>4}  {'MAE log10':>10}  {'RMSE log10':>10}  {'Bias':>8}  {'<=2x%':>7}"
    )
    logger.info("-" * 65)
    for sc in SHEAR_COLS:
        sub = held_long[(held_long["shear_col"] == sc) & held_long["log10_error"].notna()]
        if sub.empty:
            continue
        m = calc_metrics(sub["actual_cP"], sub["pred_cP"])
        logger.info(
            f"{sub['shear_label'].iloc[0]:>18}  {m['n']:>4}  {m['log_mae']:>10.4f}  "
            f"{m['log_rmse']:>10.4f}  {m['log_bias']:>+8.4f}  {m['within_2x']:>6.0f}%"
        )
    logger.info("-" * 65)
    valid_all = held_long[(held_long["actual_cP"] > 0) & (held_long["pred_cP"] > 0)]
    m_all = calc_metrics(valid_all["actual_cP"], valid_all["pred_cP"])
    logger.info(
        f"{'All shear rates':>18}  {m_all['n']:>4}  {m_all['log_mae']:>10.4f}  "
        f"{m_all['log_rmse']:>10.4f}  {m_all['log_bias']:>+8.4f}  {m_all['within_2x']:>6.0f}%"
    )
    logger.info("=" * 65)

    if not args.plot:
        logger.info("Plots suppressed.")
        return

    logger.info("Generating plots …")
    subtitle = f"{n_ctx} context  |  {n_held} held-out"
    make_parity_plot(
        long_df,
        SHEAR_COLS,
        f"Ibalizumab — All Shear Rates\n{subtitle}",
        os.path.join(args.out_dir, "parity_ibal_all_shears.png"),
        single_shear=False,
        context_ids=context_id_set,
    )
    make_parity_plot(
        long_df,
        ["Viscosity_1000"],
        f"Viscosity @ 1 000 s⁻¹ — Ibalizumab\n{subtitle}",
        os.path.join(args.out_dir, "parity_ibal_1000.png"),
        single_shear=True,
        context_ids=context_id_set,
    )

    v1000 = pd.to_numeric(held_out_df["Viscosity_1000"], errors="coerce").fillna(0)
    if v1000.max() > 0:
        pid = str(held_out_df.loc[v1000.idxmax(), "ID"])
        prof = predictor.predict(held_out_df[held_out_df["ID"] == pid].copy())
        make_profile_plot(prof, pid, os.path.join(args.out_dir, f"profile_ibal_{pid}.png"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
