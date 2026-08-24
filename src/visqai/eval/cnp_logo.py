"""
cnp_logo.py
===========
CNP side of the Phase 0 leave-one-GROUP-out harness: trains a fresh
CrossSampleCNP on each fold's training rows (fresh ColumnTransformer too --
no leakage of held-out statistics into the scaler), then measures zero-shot
(memory_vector=None) and few-shot (k in `shots`, drawn from the held-out
group itself via .learn()) log10-space error on the remainder of the held-
out group.

This generalizes cli/parity_eval.py's single-protein (ibalizumab-only)
context-select-and-score methodology to all three axes (protein, ingredient,
protein_class) and makes it scriptable/repeatable across every group instead
of one hand-run experiment.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd

from visqai.eval.constants import SHEAR_COLS
from visqai.eval.data_prep import prepare_df
from visqai.eval.logo_splits import (
    LogoGroup,
    build_groups,
    zero_ingredient_properties,
    zero_protein_descriptors,
)
from visqai.eval.metrics import calc_metrics
from visqai.inference.predictor import ViscosityPredictorCNP
from visqai.preprocessing.pipeline import build_feature_frame
from visqai.training.data import load_and_preprocess
from visqai.training.run import DEFAULT_PARAMS, train_final_model

# Fold-level train/test range guard: how many training-fold standard
# deviations a held-out row's feature value may sit from the training fold's
# mean before it's flagged. Chosen to match the "no feature should inject a
# >~5sigma activation" bar from the zero-variance fit-time fix in
# visqai.training.data. This is the real fix behind the charge_screened /
# proline conc-split / salt property-vector incidents: each was an engineered
# feature that was well-behaved in-sample but silently exploded in a held-out
# fold because its training-fold scale was degenerate. Catching that
# automatically here means the next such regression is a log line, not a
# retrain-and-diff-by-hand session.
FOLD_RANGE_N_SIGMA: float = 5.0


def _check_fold_feature_range(work_dir, held_df, n_sigma: float = FOLD_RANGE_N_SIGMA) -> dict:
    """Guard (b)/(a) from the P0 fix: compare every held-out row's engineered
    numeric feature value against the TRAINING fold's fitted StandardScaler
    statistics (mean_/var_) -- the exact statistics later used to whiten the
    network's input. Flags:
      (a) any numeric column that is zero-variance in the training fold
          (sklearn then degenerately sets scale_=1, passing the column
          through nearly raw), and
      (b) any held-out value more than `n_sigma` standard deviations from the
          training fold's mean under that scaler.
    Logs (does not raise) so the LOGO harness keeps running across folds --
    the point is visibility, not a hard abort. Returns the violations dict
    for callers that want to assert on it (e.g. in tests)."""
    import joblib

    preprocessor = joblib.load(os.path.join(work_dir, "preprocessor.pkl"))
    num_cols = list(preprocessor.transformers_[0][2])
    scaler = preprocessor.named_transformers_["num"]

    held_built, _num_cols, _cat_cols = build_feature_frame(held_df)

    violations = {"zero_variance": [], "out_of_range": {}}
    for i, col in enumerate(num_cols):
        if col not in held_built.columns:
            continue
        if scaler.var_[i] <= 1e-12:
            violations["zero_variance"].append(col)

        vals = pd.to_numeric(held_built[col], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        scale = scaler.scale_[i] if scaler.scale_[i] > 0 else 1.0
        z = np.abs((vals - scaler.mean_[i]) / scale)
        n_bad = int((z > n_sigma).sum())
        if n_bad:
            violations["out_of_range"][col] = {
                "n_bad": n_bad,
                "n_total": len(vals),
                "max_abs_z": float(z.max()),
            }

    if violations["zero_variance"] or violations["out_of_range"]:
        print(
            "  [cnp_logo] FOLD RANGE GUARD fired -- "
            f"zero-variance train columns={violations['zero_variance']}; "
            f"held-out values beyond {n_sigma}sigma of train="
            f"{violations['out_of_range']}"
        )

    return violations


def _train_fold_model(train_df, work_dir, max_epochs, patience, params=None, seed=None, held_df=None):
    import torch

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    csv_path = os.path.join(work_dir, "train_fold.csv")
    train_df.to_csv(csv_path, index=False)
    samples, static_dim, physics_scaler, protected_indices = load_and_preprocess(csv_path, save_dir=work_dir)
    n_groups = len(set(s["group"] for s in samples))
    if n_groups < 2:
        raise ValueError(
            f"Fold has only {n_groups} protein group(s) after the split -- "
            "CNP needs >=2 to learn any cross-protein structure."
        )

    if held_df is not None:
        _check_fold_feature_range(work_dir, held_df)

    return train_final_model(
        samples,
        static_dim,
        physics_scaler,
        protected_indices,
        out_dir=work_dir,
        params=params or DEFAULT_PARAMS,
        max_epochs=max_epochs,
        patience=patience,
        verbose=False,
    )


# A held-out group's few-shot MAE at a given k is itself an average over
# `n_repeats` random context/target draws. For a SMALL group, the number of
# distinct k-context choices (n choose k) is small, so a fixed n_repeats
# samples a much sparser fraction of that space than it does for a large
# group -- the estimate is noisier exactly where the group is smallest.
# Concretely: belatacept (n_held=12, the smallest real protein) at k=8 has
# only C(12,8)=495 possible context choices; 5 repeats (~1% coverage)
# produced an all_context_lift of -0.0117 (context-gate violation) in a
# real run, while a 30-repeat rerun of the SAME fold (same seed, same
# trained model) settled at a stable, non-violating +0.0074 with
# monotone_context=True -- confirming this was sampling noise in the
# few-shot estimate, not a real corrector regression (see
# visqai.inference.predictor's TAU2_CONC ablation, which reproduced the
# exact same numbers with the concentration term disabled, ruling out Task
# 1.1 as the cause). Inference repeats are cheap (forward passes, not
# retraining), so scaling them up for small groups costs little. Calibrated
# so belatacept lands at exactly the empirically-verified 30 repeats
# (6x its base n_repeats=5); other small groups scale proportionally,
# tapering to no change (1x) once a group is comfortably large.
REFERENCE_N_HELD_FOR_REPEATS: int = 72  # = 6 * 12, i.e. belatacept's own n_held at the calibrated 6x cap
MAX_N_REPEATS_MULTIPLIER: int = 6


def _effective_n_repeats(n_held: int, n_repeats: int) -> int:
    """Scales `n_repeats` up for held-out groups smaller than
    REFERENCE_N_HELD_FOR_REPEATS, capped at MAX_N_REPEATS_MULTIPLIER x. A
    group AT the reference size gets exactly the caller's own n_repeats
    (no change in behavior for typical/large groups); smaller groups scale
    up proportionally to (reference / n_held)."""
    if n_held <= 0:
        return n_repeats
    scale = min(MAX_N_REPEATS_MULTIPLIER, max(1.0, REFERENCE_N_HELD_FOR_REPEATS / n_held))
    return int(round(n_repeats * scale))


def _shot_metrics(predictor, held_df, k, n_repeats, rng):
    """Average calc_metrics (pooled across all shear columns) over
    n_repeats random k-context / rest-target draws. k=0 is zero-shot
    (memory_vector cleared -> predictor.predict defaults it to zero)."""
    n = len(held_df)
    if k > 0 and k >= n:
        return None

    reps = n_repeats if k > 0 else 1
    all_true, all_pred = [], []
    for _ in range(reps):
        predictor.memory_vector = None
        predictor.context_t = None
        if k == 0:
            target_df = held_df
        else:
            idx = rng.permutation(n)
            ctx_idx, tgt_idx = idx[:k], idx[k:]
            predictor.learn(held_df.iloc[ctx_idx].reset_index(drop=True))
            target_df = held_df.iloc[tgt_idx].reset_index(drop=True)

        pred_df = predictor.predict(target_df)
        for sc in SHEAR_COLS:
            pc = f"Pred_{sc}"
            if sc not in target_df.columns or pc not in pred_df.columns:
                continue
            true = pd.to_numeric(target_df[sc], errors="coerce")
            pred = pd.to_numeric(pred_df[pc], errors="coerce")
            mask = true.notna()
            if mask.any():
                all_true.append(true[mask].values)
                all_pred.append(pred[mask].values)

    if not all_true:
        return None
    return calc_metrics(np.concatenate(all_true), np.concatenate(all_pred))


def run_cnp_fold(
    train_df: pd.DataFrame,
    held_df: pd.DataFrame,
    group: LogoGroup,
    work_dir: str,
    shots=(0, 1, 2, 4, 8),
    n_repeats=5,
    max_epochs=500,
    patience=80,
    params=None,
    seed=0,
    corrector_mode="linear",
) -> dict:
    """Train one fold's model and score zero-shot + few-shot log10 error on
    the held-out group. For ingredient groups, also scores the ablation
    counterfactual (property vector zeroed, i.e. the ingredient treated as
    unknown) at zero-shot, to test whether the property vector buys real
    extrapolation over "drop that ingredient's features to zero".

    `corrector_mode` ("linear" or "kernel") selects which few-shot corrector
    visqai.inference.predictor.ViscosityPredictorCNP uses -- "linear" is
    Task 1.1's default; "kernel" opts into Task 1.2's kernel-weighted local
    residual model. Set on the predictor immediately after construction,
    before any .learn() call, so every context draw in this fold uses it."""
    held_df = prepare_df(held_df, drop_bad_rows=True)
    if len(held_df) < 2:
        return {"axis": group.axis, "group": group.key, "n_held": len(held_df), "error": "too few held-out rows"}

    _train_fold_model(train_df, work_dir, max_epochs, patience, params=params, seed=seed, held_df=held_df)
    predictor = ViscosityPredictorCNP(work_dir, verbose=False)
    predictor.corrector_mode = corrector_mode

    rng = np.random.RandomState(seed)
    effective_n_repeats = _effective_n_repeats(len(held_df), n_repeats)
    row = {
        "axis": group.axis,
        "group": group.key,
        "n_held": len(held_df),
        "n_repeats_used": effective_n_repeats,
    }
    for k in shots:
        m = _shot_metrics(predictor, held_df, k, effective_n_repeats, rng)
        prefix = "zero_shot" if k == 0 else f"fewshot_k{k}"
        if m is None:
            row[f"{prefix}_log_mae"] = np.nan
        else:
            row[f"{prefix}_log_mae"] = m["log_mae"]
            row[f"{prefix}_log_rmse"] = m["log_rmse"]
            row[f"{prefix}_within_2x"] = m["within_2x"]
            row[f"{prefix}_n"] = m["n"]

    if group.axis == "ingredient":
        ablated_held = zero_ingredient_properties(held_df, group)
        m_abl = _shot_metrics(predictor, ablated_held, 0, 1, rng)
        row["ablation_zero_shot_log_mae"] = m_abl["log_mae"] if m_abl else np.nan
        zshot = row.get("zero_shot_log_mae", np.nan)
        if m_abl and not np.isnan(zshot):
            # Positive => real properties beat the zeroed/"unknown" fallback
            # (the property vector is buying extrapolation, as designed).
            row["ablation_delta"] = m_abl["log_mae"] - zshot

    # Context gate metric: how much few-shot context helps (positive) or
    # hurts (negative) relative to zero-shot, on the SAME held-out group.
    # This is the number the prior/correction decoder split (visqai.models.
    # cnp) and the residual training objective (visqai.training.loop) are
    # meant to keep >= 0 -- context should help or abstain, never actively
    # make a good zero-shot prediction worse. run_cnp_logo asserts on this
    # column as a hard guardrail once every fold has reported in.
    zshot = row.get("zero_shot_log_mae", np.nan)
    fewshot_maes = [
        row[f"fewshot_k{k}_log_mae"]
        for k in shots
        if k != 0 and f"fewshot_k{k}_log_mae" in row and not np.isnan(row[f"fewshot_k{k}_log_mae"])
    ]
    if fewshot_maes and not np.isnan(zshot):
        best_fewshot = min(fewshot_maes)
        row["best_fewshot_log_mae"] = best_fewshot
        row["lift"] = zshot - best_fewshot

    # Task 0.2 (issue1_query_conditioned_correction_plan.md): honest per-k
    # reporting. `lift` above is the BEST-CASE (min over k) fewshot MAE,
    # which can hide a real regression at a LARGER k behind a smaller k's
    # good score -- exactly the blind spot the plan calls out. Report every
    # k's own lift, the lift at the LARGEST feasible k (`all_context_lift`
    # -- what a caller handing over everything they have actually gets, not
    # the best of several draws), and whether lift is monotonically
    # non-decreasing in k (`monotone_context` -- more context should never
    # make predictions meaningfully worse; Task 1.2's kernel corrector
    # targets exactly this property).
    k_lifts = {}
    if not np.isnan(zshot):
        for k in shots:
            if k == 0:
                continue
            mae_col = f"fewshot_k{k}_log_mae"
            if mae_col in row and not np.isnan(row[mae_col]):
                lift_k = zshot - row[mae_col]
                row[f"fewshot_k{k}_lift"] = lift_k
                k_lifts[k] = lift_k

    if k_lifts:
        largest_k = max(k_lifts)
        row["all_context_lift"] = k_lifts[largest_k]
        ordered_lifts = [k_lifts[k] for k in sorted(k_lifts)]
        row["monotone_context"] = bool(
            all(b >= a - MONOTONE_CONTEXT_TOLERANCE for a, b in zip(ordered_lifts, ordered_lifts[1:]))
        )

    return row


# How far below zero `lift` (zero_shot_log_mae - best_fewshot_log_mae) may
# drift before the context gate fails the run. Originally 0.01 was tight
# enough to fail on pure sampling noise (e.g. ibalizumab's -0.0107 on an
# early clean run of the neural correction_head), so it was loosened to
# 0.03. T-R3.6/T-R3.7 (visqai.inference.predictor's delta corrector) removed
# the source of that noise directly -- offset_hat now abstains (contributes
# 0) unless context residuals are confidently non-zero (formulation-level
# empirical-Bayes confidence gate) and unless there are >=2 distinct context
# formulations at all -- so the gate is retightened to 0.01, its originally-
# intended value, now that the mechanism it's guarding is noise-resistant by
# construction rather than by a looser tolerance.
CONTEXT_GATE_TOLERANCE: float = 0.01

# Mean lift across ALL held-out groups must be >= this floor. Complements the
# per-group tolerance above: a run could have every single group sitting just
# under the per-group tolerance and still represent a real systemic
# regression that the per-group check alone would wave through.
MEAN_LIFT_FLOOR: float = 0.0

# Task 0.2: how far below 0 a step-to-step lift change (larger k vs. the
# previous k) may drift before a group is flagged non-monotone. Loose enough
# to absorb ordinary few-shot sampling noise between draws (n_repeats random
# k-context draws, not a fixed context set) while still catching a real,
# consistent regression as context grows -- same order of magnitude as
# CONTEXT_GATE_TOLERANCE, not a separate calibration.
MONOTONE_CONTEXT_TOLERANCE: float = 0.01


def _assert_context_gate(
    rows: list[dict], tolerance: float = CONTEXT_GATE_TOLERANCE, mean_floor: float = MEAN_LIFT_FLOOR
) -> None:
    """Hard guardrail: context (few-shot) must never score worse than
    zero-shot by more than `tolerance` on any single group, AND the mean lift
    across all held-out groups must be >= `mean_floor`. Pairs with the
    fold-level feature-range guard (_check_fold_feature_range) -- that one
    catches an OOD feature value before it can corrupt a prediction, this one
    catches the downstream symptom (context making a prediction worse) if it
    ever happens anyway. Runs OUTSIDE run_cnp_logo's per-fold try/except so a
    violation raises all the way up and fails the run/CI instead of being
    swallowed into an 'error' row.

    Task 0.2: also fails on `all_context_lift` (the lift at the LARGEST
    feasible k), not just the best-case `lift` (min over k) -- a run where
    small-k context helps but the largest context a caller could actually
    hand over regresses past `tolerance` is exactly the honesty gap Task 0.2
    exists to close, and the old `lift`-only check let it through silently."""
    lifts = [r["lift"] for r in rows if "lift" in r and not pd.isna(r["lift"])]

    violations = [
        (r.get("axis"), r.get("group"), "lift", r["lift"])
        for r in rows
        if "lift" in r and not pd.isna(r["lift"]) and r["lift"] < -tolerance
    ]
    violations += [
        (r.get("axis"), r.get("group"), "all_context_lift", r["all_context_lift"])
        for r in rows
        if "all_context_lift" in r
        and not pd.isna(r["all_context_lift"])
        and r["all_context_lift"] < -tolerance
    ]
    if violations:
        detail = "; ".join(
            f"{axis}/{group} [{metric}]: lift={lift:+.4f}" for axis, group, metric, lift in violations
        )
        raise AssertionError(
            f"Context gate failed: few-shot scored worse than zero-shot by more than "
            f"{tolerance} log MAE on {len(violations)} group(s)/metric(s) -- {detail}"
        )

    if lifts:
        mean_lift = sum(lifts) / len(lifts)
        if mean_lift < mean_floor:
            raise AssertionError(
                f"Context gate failed: mean lift across {len(lifts)} group(s) is "
                f"{mean_lift:+.4f}, below the {mean_floor:+.4f} floor -- context is "
                f"hurting more often than it helps even though no single group "
                f"breached the per-group {tolerance} tolerance."
            )


def run_cnp_logo(
    df: pd.DataFrame,
    axis: str,
    base_work_dir: str,
    min_rows: int = 2,
    groups=None,
    shots=(0, 1, 2, 4, 8),
    n_repeats=5,
    max_epochs=500,
    patience=80,
    params=None,
    seed=0,
    keep_fold_dirs=False,
    enforce_context_gate: bool = True,
    ablate_protein_descriptors: bool = False,
    corrector_mode: str = "linear",
) -> pd.DataFrame:
    """Run the CNP LOGO harness over every group for `axis` (or a
    caller-supplied subset via `groups`, for smoke tests). Each fold trains
    a fresh model in its own subdirectory of `base_work_dir`, deleted after
    scoring unless `keep_fold_dirs`.

    `enforce_context_gate` runs _assert_context_gate on the finished board
    (default on). Undertrained smoke-test configs (few epochs/low patience)
    can produce noisy few-shot metrics that trip the gate for reasons that
    have nothing to do with a real regression; pass False to collect the
    scoreboard without the hard assertion in that case.

    `ablate_protein_descriptors` applies zero_protein_descriptors to `df`
    ONCE, up front, before any fold split -- the P0 descriptor-vs-context
    test: strips every protein's only transferable identity handle (see
    logo_splits.zero_protein_descriptors) so zero-shot must rely on
    population-level features alone and any few-shot lift must come from
    context. Applied dataset-wide (not per-fold like the ingredient
    ablation) because protein identity isn't something a single fold can
    partially hold out.

    `corrector_mode` ("linear" or "kernel") is forwarded to every fold's
    predictor -- see run_cnp_fold's docstring.
    """
    if ablate_protein_descriptors:
        df = zero_protein_descriptors(df)
    fold_groups = groups if groups is not None else build_groups(df, axis, min_rows=min_rows)
    rows = []
    os.makedirs(base_work_dir, exist_ok=True)
    for g in fold_groups:
        train_df, held_df = g.split(df)
        if held_df.empty or train_df.empty:
            continue
        fold_key = g.key.replace("/", "_").replace("=", "-")
        fold_dir = os.path.join(base_work_dir, f"{axis}__{fold_key}")
        os.makedirs(fold_dir, exist_ok=True)
        try:
            row = run_cnp_fold(
                train_df,
                held_df,
                g,
                fold_dir,
                shots=shots,
                n_repeats=n_repeats,
                max_epochs=max_epochs,
                patience=patience,
                params=params,
                seed=seed,
                corrector_mode=corrector_mode,
            )
        except Exception as e:  # keep the harness going across folds
            row = {"axis": g.axis, "group": g.key, "error": str(e)}
        rows.append(row)
        if not keep_fold_dirs:
            shutil.rmtree(fold_dir, ignore_errors=True)

    if enforce_context_gate:
        _assert_context_gate(rows)
    return pd.DataFrame(rows)
