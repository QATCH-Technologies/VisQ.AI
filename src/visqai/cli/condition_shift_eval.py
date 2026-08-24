"""
condition_shift_eval.py
========================
Task 0.1 scoreboard (issue1_query_conditioned_correction_plan.md): runs
visqai.eval.condition_shift.run_condition_shift over every real protein and
reports the concentration/ingredient/buffer condition-shift board, plus the
shift-validity sanity check against a random-split protein LOGO run.

This is the Phase 1/2 acceptance yardstick -- see that module's docstring
for why the random-split protein LOGO (cli/logo_eval.py) cannot measure the
capability this plan's Weakness #1 names (query-conditioned correction), and
for why this board is UNDERPOWERED on the good-prior stratum (MDE ~= 0.017
log MAE at n=11 protein clusters) -- every future comparison this CLI prints
reports its MDE alongside the result for exactly that reason.

SHIP DECISION (Task A.2/A.3 addendum): corrector_mode="linear" (with the
Task A.3 context-support clamp, always active in that mode) is the
DEPLOYED default. corrector_mode="kernel" is a non-default research branch
-- see condition_shift.py's module docstring for the three-way ablation
that a bad-prior-stratum comparison (the SIGNIFICANT stratum) shows kernel
gives up ~73% of linear's bad-prior gain for a good-prior "gain" that sits
inside the noise floor. Do not flip the default without re-running that
ablation.

Usage:
    PYTHONPATH=src python -m visqai.cli.condition_shift_eval --quick
    PYTHONPATH=src python -m visqai.cli.condition_shift_eval  # full run
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import pandas as pd

from visqai.eval.cnp_logo import run_cnp_logo
from visqai.eval.condition_shift import (
    axis_rollup,
    leave_one_protein_out_sensitivity,
    minimum_detectable_effect,
    run_condition_shift,
    shift_validity_check,
    stratified_summary,
    validated_directions,
)
from visqai.eval.data_prep import prepare_df
from visqai.logging_config import configure_logging

logger = logging.getLogger("ConditionShiftEval")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Task 0.1 condition-shift scoreboard.")
    ap.add_argument(
        "--data",
        default=Path("data/raw/formulation_data_07062026.csv"),
        help="Master CSV containing every protein/ingredient/class (NOT a pre-filtered training split).",
    )
    ap.add_argument("--out_dir", default=Path("models/experiments/condition_shift_eval"))
    ap.add_argument(
        "--min-rows", type=int, default=2, help="Minimum held-out rows for a protein to be evaluated."
    )
    ap.add_argument(
        "--proteins",
        default=None,
        help="Comma-separated subset of protein keys to run (e.g. 'ibalizumab,adalimumab'). Default: every protein.",
    )
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--keep-fold-dirs", action="store_true", help="Keep each fold's trained checkpoint on disk."
    )
    ap.add_argument(
        "--skip-validity-check",
        action="store_true",
        help=(
            "Skip the shift-validity sanity check entirely. Directions cannot be flagged validated/"
            "invalidated, so the axis_rollup headline table and MDE are NOT computed -- prefer "
            "--random-split-board to reuse a cached board instead of skipping outright."
        ),
    )
    ap.add_argument(
        "--random-split-board",
        default=None,
        help=(
            "Path to a previously-saved random_split_zero_shot_board.csv (from an earlier run's "
            "--out_dir) -- reuses it for the shift-validity check instead of retraining 12 more "
            "models. Takes priority over --skip-validity-check."
        ),
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test preset: max-epochs=30, patience=8.",
    )
    ap.add_argument(
        "--corrector-mode",
        choices=["linear", "kernel", "offset_only"],
        default="linear",
        help=(
            "Which few-shot corrector the predictor uses. 'linear' (Task 1.1 + the Task A.3 "
            "context-support clamp) is the SHIPPED default. 'kernel' (Task 1.2) is a non-default "
            "RESEARCH BRANCH ONLY -- see this module's docstring for why it is not deployed. "
            "'offset_only' is the pre-Task-1.1 ablation arm (Task A.2)."
        ),
    )
    return ap.parse_args(argv)


def _apply_quick_preset(args):
    if not args.quick:
        return args
    args.max_epochs = min(args.max_epochs, 30)
    args.patience = min(args.patience, 8)
    return args


def main(argv=None):
    configure_logging()
    args = parse_args(argv)
    args = _apply_quick_preset(args)

    if not os.path.exists(args.data):
        raise SystemExit(f"ERROR: data not found -- {args.data}")

    if args.corrector_mode == "kernel":
        logger.warning(
            "*** corrector_mode='kernel' is a RESEARCH BRANCH, not the shipped corrector -- see "
            "condition_shift.py's module docstring (Task A.2/A.3 ablation) for why. ***"
        )

    proteins = args.proteins.split(",") if args.proteins else None
    df = prepare_df(pd.read_csv(args.data, index_col=False))
    logger.info(f"Loaded {len(df)} rows from {args.data}")

    os.makedirs(args.out_dir, exist_ok=True)
    work_dir = os.path.join(args.out_dir, "work")

    logger.info("Running condition-shift harness (trains one model per held-out protein)...")
    board = run_condition_shift(
        df,
        work_dir,
        proteins=proteins,
        min_rows=args.min_rows,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        keep_fold_dirs=args.keep_fold_dirs,
        corrector_mode=args.corrector_mode,
    )
    if not args.keep_fold_dirs:
        shutil.rmtree(work_dir, ignore_errors=True)

    csv_path = os.path.join(args.out_dir, "condition_shift_scoreboard.csv")
    board.to_csv(csv_path, index=False)
    logger.info(f"\nScoreboard saved: {csv_path}")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        logger.info("\n" + board.to_string(index=False))

    # Shift-validity check runs BEFORE the summary (Task A.1) so
    # stratified_summary/axis_rollup can flag/exclude directions
    # shift_validity_check does NOT confirm induce real extrapolation (e.g.
    # conc_high_ctx_low_target -- see condition_shift.py's module docstring)
    # instead of silently pooling them into a headline capability number.
    validated = None
    random_board = None
    if args.random_split_board:
        logger.info(f"\nLoading cached random-split board: {args.random_split_board}")
        random_board = pd.read_csv(args.random_split_board)
    elif not args.skip_validity_check:
        logger.info("\nRunning random-split protein LOGO (zero-shot only) for the shift-validity check...")
        random_board = run_cnp_logo(
            df,
            "protein",
            os.path.join(args.out_dir, "random_split_work"),
            min_rows=args.min_rows,
            groups=None,
            shots=(0,),
            n_repeats=1,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
            keep_fold_dirs=False,
            enforce_context_gate=False,
        )
        shutil.rmtree(os.path.join(args.out_dir, "random_split_work"), ignore_errors=True)
        random_board.to_csv(os.path.join(args.out_dir, "random_split_zero_shot_board.csv"), index=False)

    if random_board is not None:
        validity = shift_validity_check(board, random_board)
        logger.info(f"\nShift-validity check (per direction): {validity}")
        if not validity.get("ok", False):
            logger.warning(
                "*** Shift-validity check FAILED -- NO concentration-shift direction shows a "
                "harder-than-random-split-baseline prior_only error. Fix the split before trusting "
                "Task 1.x/2.x results against this board. ***"
            )
        else:
            passing = [d for d, v in validity.get("per_direction", {}).items() if v.get("ok")]
            logger.info(f"Shift-validity check PASSED via direction(s): {passing}")
        validated = validated_directions(board, random_board)
        not_validated = set(board["direction"].unique()) - validated
        if not_validated:
            logger.warning(
                f"*** Directions NOT confirmed as real extrapolation (interpolation risk, "
                f"excluded from the headline axis_rollup below): {sorted(not_validated)} ***"
            )
    else:
        logger.warning(
            "*** --skip-validity-check with no --random-split-board: direction validity is "
            "UNKNOWN. The axis_rollup headline table and MDE cannot be computed -- only the raw "
            "per-direction stratified_summary below is available, and it may include a direction "
            "that fails its own extrapolation check. ***"
        )

    if not board.empty and "delta" in board.columns:
        summary = stratified_summary(board, validated=validated)
        summary_path = os.path.join(args.out_dir, "stratified_summary.csv")
        summary.to_csv(summary_path, index=False)
        logger.info(f"\nPer-direction stratified summary (Task A.1) saved: {summary_path}")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            logger.info("\n" + summary.to_string(index=False))

        if validated is not None:
            rollup = axis_rollup(summary, validated=validated)
            rollup_path = os.path.join(args.out_dir, "axis_rollup.csv")
            rollup.to_csv(rollup_path, index=False)
            logger.info(f"\nHEADLINE axis rollup (validated directions only) saved: {rollup_path}")
            with pd.option_context("display.width", 200, "display.max_columns", None):
                logger.info("\n" + rollup.to_string(index=False))

            logger.info("\nMinimum detectable effect (95% confidence / 80% power), per axis/stratum:")
            valid_board = board[board["direction"].isin(validated)].copy()
            valid_board["prior_band"] = valid_board["prior_only_log_mae"].apply(
                lambda v: "bad_prior" if v >= 0.15 else "good_prior"
            )
            for axis, g_axis in valid_board.groupby("axis"):
                strata = [("all", g_axis)] + list(g_axis.groupby("prior_band"))
                for stratum, g in strata:
                    mde = minimum_detectable_effect(g)
                    logger.info(
                        f"  [{axis}/{stratum}] n_proteins={mde['n_proteins']} "
                        f"observed_mean_delta={mde['observed_mean_delta']:+.4f} MDE={mde['mde']:.4f} "
                        f"-- a difference smaller than MDE is NOT resolvable at this sample size."
                    )

        logger.info("\nLeave-one-protein-out sensitivity, concentration axis (by direction):")
        for direction in sorted(board.loc[board["axis"] == "concentration", "direction"].unique()):
            loo = leave_one_protein_out_sensitivity(board, "concentration", direction=direction)
            with pd.option_context("display.width", 200, "display.max_columns", None):
                logger.info(f"\n[{direction}]\n" + loo.to_string(index=False))

    logger.info("Done.")
    return board


if __name__ == "__main__":
    main()
