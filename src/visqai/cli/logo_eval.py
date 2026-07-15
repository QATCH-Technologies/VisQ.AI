"""
logo_eval.py
============
Phase 0 scoreboard: leave-one-GROUP-out evaluation across all three held-out
axes (protein, ingredient, protein_class), reporting the reference baseline
(feature-only HistGBM) alongside CNP zero-shot and few-shot log10 MAE for
every held-out group, plus the leave-one-ingredient-out property-vector
ablation check.

This is the primary scoreboard called for by Phase 0 -- parity_eval.py only
ever measured leave-Ibalizumab-out; this generalizes it to every protein,
every chemically-meaningful ingredient category, and every protein class,
and adds the reference baseline every later change must be measured against:

    success rule: CNP zero-shot must reach <= baseline; CNP few-shot must
    beat it. If a change doesn't move the CNP toward those lines on
    held-out groups, revert it.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from visqai.eval.baseline import run_baseline_logo
from visqai.eval.cnp_logo import run_cnp_logo
from visqai.eval.data_prep import prepare_df
from visqai.eval.logo_splits import build_groups
from visqai.logging_config import configure_logging

logger = logging.getLogger("LogoEval")

AXES = ["protein", "ingredient", "protein_class"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Phase 0 leave-one-GROUP-out scoreboard.")
    ap.add_argument(
        "--data",
        default=Path("data/raw/formulation_data_07062026.csv"),
        help="Master CSV containing every protein/ingredient/class (NOT a pre-filtered training split).",
    )
    ap.add_argument("--axis", choices=AXES + ["all"], default="all")
    ap.add_argument("--out_dir", default=Path("models/experiments/logo_eval"))
    ap.add_argument("--min-rows", type=int, default=2, help="Minimum held-out rows for a group to be evaluated.")
    ap.add_argument(
        "--groups",
        default=None,
        help="Comma-separated subset of group keys to run (e.g. 'ibalizumab,adalimumab'). Default: every group.",
    )
    ap.add_argument("--shots", default="0,1,2,4,8", help="Comma-separated context sizes ('0' = zero-shot).")
    ap.add_argument("--n-repeats", type=int, default=5, help="Random context draws averaged per shot count.")
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-only", action="store_true", help="Skip CNP training (fast; baseline reference only).")
    ap.add_argument("--cnp-only", action="store_true", help="Skip the baseline regressor.")
    ap.add_argument("--keep-fold-dirs", action="store_true", help="Keep each fold's trained checkpoint on disk.")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test preset: max-epochs=30, patience=8, n-repeats=2, shots=0,1,4.",
    )
    return ap.parse_args(argv)


def _apply_quick_preset(args):
    if not args.quick:
        return args
    args.max_epochs = min(args.max_epochs, 30)
    args.patience = min(args.patience, 8)
    args.n_repeats = min(args.n_repeats, 2)
    args.shots = "0,1,4"
    return args


def main(argv=None):
    configure_logging()
    args = parse_args(argv)
    args = _apply_quick_preset(args)

    if not os.path.exists(args.data):
        raise SystemExit(f"ERROR: data not found -- {args.data}")

    shots = tuple(int(s) for s in str(args.shots).split(","))
    axes = AXES if args.axis == "all" else [args.axis]
    group_filter = set(args.groups.split(",")) if args.groups else None

    df = prepare_df(pd.read_csv(args.data, index_col=False))
    logger.info(f"Loaded {len(df)} rows from {args.data}")

    os.makedirs(args.out_dir, exist_ok=True)
    all_scoreboards = []

    for axis in axes:
        fold_groups = build_groups(df, axis, min_rows=args.min_rows)
        if group_filter is not None:
            fold_groups = [g for g in fold_groups if g.key in group_filter]
        if not fold_groups:
            logger.warning(f"No groups found for axis='{axis}' (after filtering); skipping.")
            continue
        logger.info(f"\n{'='*70}\nAXIS: {axis}  ({len(fold_groups)} held-out group(s): {[g.key for g in fold_groups]})\n{'='*70}")

        baseline_df = pd.DataFrame()
        if not args.cnp_only:
            logger.info(f"[{axis}] Running reference baseline (HistGBM, feature-only)...")
            baseline_df = run_baseline_logo(df, axis, groups=fold_groups)
            baseline_df = baseline_df.add_prefix("baseline_").rename(
                columns={"baseline_axis": "axis", "baseline_group": "group"}
            )

        cnp_df = pd.DataFrame()
        if not args.baseline_only:
            logger.info(f"[{axis}] Running CNP LOGO fold(s) (this trains one model per group)...")
            work_dir = os.path.join(args.out_dir, f"work_{axis}")
            cnp_df = run_cnp_logo(
                df,
                axis,
                work_dir,
                groups=fold_groups,
                shots=shots,
                n_repeats=args.n_repeats,
                max_epochs=args.max_epochs,
                patience=args.patience,
                seed=args.seed,
                keep_fold_dirs=args.keep_fold_dirs,
            )
            if not args.keep_fold_dirs:
                shutil.rmtree(work_dir, ignore_errors=True)

        if not baseline_df.empty and not cnp_df.empty:
            merged = pd.merge(baseline_df, cnp_df, on=["axis", "group"], how="outer")
        elif not baseline_df.empty:
            merged = baseline_df
        else:
            merged = cnp_df
        all_scoreboards.append(merged)

    if not all_scoreboards:
        raise SystemExit("No results produced -- check --axis/--groups/--min-rows.")

    scoreboard = pd.concat(all_scoreboards, ignore_index=True)

    if "baseline_log_mae" in scoreboard.columns and "zero_shot_log_mae" in scoreboard.columns:
        scoreboard["zero_shot_meets_baseline"] = (
            scoreboard["zero_shot_log_mae"] <= scoreboard["baseline_log_mae"]
        )
    fewshot_cols = [c for c in scoreboard.columns if c.startswith("fewshot_k") and c.endswith("_log_mae")]
    if "baseline_log_mae" in scoreboard.columns and fewshot_cols:
        best_fewshot = scoreboard[fewshot_cols].min(axis=1)
        scoreboard["best_fewshot_beats_baseline"] = best_fewshot < scoreboard["baseline_log_mae"]

    csv_path = os.path.join(args.out_dir, "logo_scoreboard.csv")
    scoreboard.to_csv(csv_path, index=False)
    logger.info(f"\nScoreboard saved: {csv_path}")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        logger.info("\n" + scoreboard.to_string(index=False))

    if "zero_shot_meets_baseline" in scoreboard.columns:
        n_ok = scoreboard["zero_shot_meets_baseline"].sum()
        n_total = scoreboard["zero_shot_meets_baseline"].notna().sum()
        logger.info(f"\nZero-shot <= baseline on {n_ok}/{n_total} held-out groups.")
    if "best_fewshot_beats_baseline" in scoreboard.columns:
        n_ok = scoreboard["best_fewshot_beats_baseline"].sum()
        n_total = scoreboard["best_fewshot_beats_baseline"].notna().sum()
        logger.info(f"Best few-shot beats baseline on {n_ok}/{n_total} held-out groups.")
    if "ablation_delta" in scoreboard.columns:
        ing = scoreboard[scoreboard["axis"] == "ingredient"]
        if not ing.empty:
            n_helps = (ing["ablation_delta"] > 0).sum()
            n_total = ing["ablation_delta"].notna().sum()
            logger.info(
                f"Property vector beats zeroed-ingredient ablation on {n_helps}/{n_total} "
                "held-out ingredient groups (positive ablation_delta = real properties win)."
            )

    logger.info("Done.")
    return scoreboard


if __name__ == "__main__":
    main()
