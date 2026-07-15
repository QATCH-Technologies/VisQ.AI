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
from visqai.eval.logo_splits import LogoGroup, build_groups, zero_ingredient_properties
from visqai.eval.metrics import calc_metrics
from visqai.inference.predictor import ViscosityPredictorCNP
from visqai.training.data import load_and_preprocess
from visqai.training.run import DEFAULT_PARAMS, train_final_model


def _train_fold_model(train_df, work_dir, max_epochs, patience, params=None, seed=None):
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
) -> dict:
    """Train one fold's model and score zero-shot + few-shot log10 error on
    the held-out group. For ingredient groups, also scores the ablation
    counterfactual (property vector zeroed, i.e. the ingredient treated as
    unknown) at zero-shot, to test whether the property vector buys real
    extrapolation over "drop that ingredient's features to zero"."""
    held_df = prepare_df(held_df, drop_bad_rows=True)
    if len(held_df) < 2:
        return {"axis": group.axis, "group": group.key, "n_held": len(held_df), "error": "too few held-out rows"}

    _train_fold_model(train_df, work_dir, max_epochs, patience, params=params, seed=seed)
    predictor = ViscosityPredictorCNP(work_dir, verbose=False)

    rng = np.random.RandomState(seed)
    row = {"axis": group.axis, "group": group.key, "n_held": len(held_df)}
    for k in shots:
        m = _shot_metrics(predictor, held_df, k, n_repeats, rng)
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

    return row


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
) -> pd.DataFrame:
    """Run the CNP LOGO harness over every group for `axis` (or a
    caller-supplied subset via `groups`, for smoke tests). Each fold trains
    a fresh model in its own subdirectory of `base_work_dir`, deleted after
    scoring unless `keep_fold_dirs`."""
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
            )
        except Exception as e:  # keep the harness going across folds
            row = {"axis": g.axis, "group": g.key, "error": str(e)}
        rows.append(row)
        if not keep_fold_dirs:
            shutil.rmtree(fold_dir, ignore_errors=True)
    return pd.DataFrame(rows)
