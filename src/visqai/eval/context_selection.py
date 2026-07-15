"""
context_selection.py
=====================
Greedy forward selection (+ optional swap refinement) of a small, strategic
context set from a sample pool, scored by held-out log10-space error.

Moved from ml/cnp_mk2/ibal_parity_test.py. Novel logic, not duplicated
elsewhere -- relocated as-is (only the module-level dynamic predictor-class
loading is dropped, since callers now just import
visqai.inference.predictor.ViscosityPredictorCNP directly).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from visqai.eval.constants import N_SHEARS

logger = logging.getLogger(__name__)


def preprocess_pool(predictor, iba_df):
    """Preprocess all samples once into the tensors the encoder/decoder need."""
    static_t, shear_t, visc_t = predictor._preprocess(iba_df)
    ctx_t = torch.cat([shear_t, visc_t, static_t], dim=-1)
    return ctx_t, static_t, shear_t, visc_t


def _held_out_errors(model, ctx_t, held_out_idx, all_static_t, all_shear_t, all_visc_t):
    """Per-sample log10-space RMSE for every held-out sample."""
    with torch.no_grad():
        memory = model.encode_memory(ctx_t)
        errs = []
        for j in held_out_idx:
            lo, hi = j * N_SHEARS, (j + 1) * N_SHEARS
            pred = model.decode_from_memory(memory, all_shear_t[:, lo:hi, :], all_static_t[:, lo:hi, :])
            truth = all_visc_t[:, lo:hi, :]
            rmse = torch.sqrt(((pred - truth) ** 2).mean()).item()
            errs.append(rmse)
    return np.asarray(errs, dtype=float)


def _objective(errs: np.ndarray, mode: str) -> float:
    """Collapse per-sample errors to a single score to minimise.

    mean : average error (favours the easy bulk)
    tail : 0.5*mean + 0.5*p90 (rewards covering the high-error tail too)
    max  : worst-case error (most aggressive range coverage)
    """
    if errs.size == 0:
        return 0.0
    if mode == "mean":
        return float(errs.mean())
    if mode == "max":
        return float(errs.max())
    if mode == "tail":
        return float(0.5 * errs.mean() + 0.5 * np.percentile(errs, 90))
    raise ValueError(f"Unknown objective '{mode}'")


def _ctx_indices(sample_indices):
    out = []
    for s in sample_indices:
        out.extend(range(s * N_SHEARS, (s + 1) * N_SHEARS))
    return out


def _score_set(model, all_ctx_t, selected, n, pool, objective, all_static_t, all_shear_t, all_visc_t):
    """Objective value for a candidate selected-set (held-out = pool \\ selected)."""
    held = [i for i in pool if i not in selected]
    if not held:
        return 0.0
    ctx_t = all_ctx_t[:, _ctx_indices(selected), :]
    errs = _held_out_errors(model, ctx_t, held, all_static_t, all_shear_t, all_visc_t)
    return _objective(errs, objective)


def greedy_select(predictor, iba_df, n_select, objective="tail", refine=True, verbose=True):
    model = predictor.model
    model.eval()

    n = len(iba_df)
    if n_select > n:
        raise ValueError(f"n_select={n_select} > n_pool={n}")
    pool = list(range(n))

    if verbose:
        logger.info(f"[Preprocess] {n} samples (objective='{objective}', refine={refine}) …")
    t0 = time.perf_counter()
    all_ctx_t, all_static_t, all_shear_t, all_visc_t = preprocess_pool(predictor, iba_df)
    if verbose:
        logger.info(f"  done in {time.perf_counter()-t0:.1f}s, ctx {tuple(all_ctx_t.shape)}")

    selected = []
    step_log = []
    prev = None
    for step in range(n_select):
        remaining = [i for i in pool if i not in selected]
        best_idx, best_score = None, float("inf")
        for cand in remaining:
            score = _score_set(
                model, all_ctx_t, selected + [cand], n, pool, objective, all_static_t, all_shear_t, all_visc_t
            )
            if score < best_score - 1e-12:
                best_idx, best_score = cand, score
        selected.append(best_idx)
        imp = (prev - best_score) if prev is not None else float("nan")
        prev = best_score
        sid = iba_df.iloc[best_idx]["ID"]
        step_log.append(dict(step=step + 1, sample_idx=best_idx, sample_id=sid, score=best_score, improvement=imp))
        if verbose:
            istr = f"  Δ={imp:+.5f}" if np.isfinite(imp) else ""
            logger.info(f"[{step+1}/{n_select}] +{sid:>6} | score={best_score:.5f}{istr}")

    if refine:
        selected, swaps = _swap_refine(
            model, all_ctx_t, selected, pool, objective, all_static_t, all_shear_t, all_visc_t, iba_df, verbose
        )
        if verbose:
            logger.info(f"[Refine] {swaps} swap(s) improved the set.")

    final_score = _score_set(model, all_ctx_t, selected, n, pool, objective, all_static_t, all_shear_t, all_visc_t)
    return selected, step_log, final_score


def _swap_refine(
    model, all_ctx_t, selected, pool, objective, all_static_t, all_shear_t, all_visc_t, iba_df, verbose, max_passes=3
):
    """Try replacing each selected member with each non-member; keep improvements."""
    selected = list(selected)
    cur = _score_set(model, all_ctx_t, selected, len(pool), pool, objective, all_static_t, all_shear_t, all_visc_t)
    total_swaps = 0
    for _pass in range(max_passes):
        improved = False
        for si in range(len(selected)):
            non_members = [i for i in pool if i not in selected]
            best_repl, best_score = None, cur
            for cand in non_members:
                trial = list(selected)
                trial[si] = cand
                score = _score_set(
                    model, all_ctx_t, trial, len(pool), pool, objective, all_static_t, all_shear_t, all_visc_t
                )
                if score < best_score - 1e-9:
                    best_repl, best_score = cand, score
            if best_repl is not None:
                old = selected[si]
                selected[si] = best_repl
                cur = best_score
                total_swaps += 1
                improved = True
                if verbose:
                    logger.info(f"    swap {iba_df.iloc[old]['ID']} -> {iba_df.iloc[best_repl]['ID']} | score={cur:.5f}")
        if not improved:
            break
    return selected, total_swaps
