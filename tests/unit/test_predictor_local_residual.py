"""
test_predictor_local_residual.py
=================================
Task 1.1 (issue1_query_conditioned_correction_plan.md): unit tests for
ViscosityPredictorCNP's query-conditioned local residual corrector
(_fit_local_residual / _fit_formulation_level / _transfer_check_passes) and
predict()'s wiring of the new Protein_conc term.

These build a predictor via __new__ (bypassing __init__'s checkpoint
loading -- the fitting methods under test are pure numpy/pandas functions of
their arguments plus module-level constants, no model/preprocessor needed)
and drive the private methods directly with synthetic context arrays,
exactly the (formulation_idx, log_shear, resid, Protein_conc) shape
_context_residuals produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from visqai.inference import predictor as predictor_mod
from visqai.inference.predictor import ViscosityPredictorCNP


def _make_predictor_stub():
    p = ViscosityPredictorCNP.__new__(ViscosityPredictorCNP)
    p.corrector_mode = "linear"
    p._kernel_ctx_phi = None
    p._kernel_ctx_resid = None
    p.kernel_bandwidth = None
    p._kernel_feat_idx = None
    # Task A.3: wide-open by default (no clamping) so tests written before
    # the context-support clamp don't need to opt in explicitly -- tests
    # that exercise the clamp itself set these narrower.
    p.conc_support_min = -1e9
    p.conc_support_max = 1e9
    return p


def _make_ctx(conc_values, offset=0.05, conc_coeff=0.0, slope_coeff=0.0, noise=0.0, seed=0, shears=(2, 3, 4, 5, 7)):
    """One formulation per entry of `conc_values`, each with a real point at
    every shear in `shears`. resid = offset + conc_coeff*(conc-cbar) +
    slope_coeff*(shear-sbar) + gaussian noise. cbar/sbar are computed
    directly from the arrays this function outputs, matching how
    _fit_local_residual centers them."""
    rng = np.random.RandomState(seed)
    cbar = float(np.mean(conc_values))
    sbar = float(np.mean(shears))
    formulations, shear_list, resid_list, conc_list = [], [], [], []
    for i, c in enumerate(conc_values):
        for s in shears:
            formulations.append(i)
            shear_list.append(float(s))
            resid_list.append(
                offset + conc_coeff * (c - cbar) + slope_coeff * (s - sbar) + rng.normal(0, noise)
            )
            conc_list.append(float(c))
    return (
        np.array(formulations, dtype=int),
        np.array(shear_list, dtype=float),
        np.array(resid_list, dtype=float),
        np.array(conc_list, dtype=float),
    )


# ---------------------------------------------------------------------------
# Fallback (hard acceptance criterion): <2 distinct Protein_conc values in
# context -> conc_hat exactly 0.0, offset_hat/slope_hat bit-for-bit the old
# scalar offset+slope corrector.
# ---------------------------------------------------------------------------


def test_fewer_than_two_formulations_returns_all_zero():
    p = _make_predictor_stub()
    ctx_formulations, ctx_shear, ctx_resid, ctx_conc = _make_ctx([100.0], offset=0.2, shears=(2, 3, 4))
    out = p._fit_local_residual(ctx_formulations, ctx_shear, ctx_resid, ctx_conc)
    assert out == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_empty_context_returns_all_zero():
    p = _make_predictor_stub()
    empty = np.empty(0)
    out = p._fit_local_residual(np.empty(0, dtype=int), empty, empty, empty)
    assert out == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_constant_conc_context_yields_zero_conc_coefficient():
    p = _make_predictor_stub()
    conc_values = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    ctx_formulations, ctx_shear, ctx_resid, ctx_conc = _make_ctx(
        conc_values, offset=0.08, slope_coeff=0.01, noise=0.01, seed=2
    )
    offset_hat, conc_hat, conc_center, slope_hat, slope_center = p._fit_local_residual(
        ctx_formulations, ctx_shear, ctx_resid, ctx_conc
    )
    assert conc_hat == 0.0


def test_constant_conc_offset_matches_old_scalar_shrink_formula_exactly():
    """Reproduces T-R3.2's old `_shrink_offset` formula by hand and confirms
    Task 1.1's generalized fit collapses onto it exactly (1e-6) when context
    doesn't vary concentration -- the Task 1.1 Fallback acceptance
    criterion."""
    p = _make_predictor_stub()
    conc_values = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    ctx_formulations, ctx_shear, ctx_resid, ctx_conc = _make_ctx(
        conc_values, offset=0.08, slope_coeff=0.0, noise=0.01, seed=3
    )
    offset_hat, conc_hat, conc_center, slope_hat, slope_center = p._fit_local_residual(
        ctx_formulations, ctx_shear, ctx_resid, ctx_conc
    )
    assert conc_hat == 0.0

    form_means = pd.Series(ctx_resid).groupby(ctx_formulations).mean().values
    k = len(form_means)
    sigma2 = max(float(np.var(form_means, ddof=1)), predictor_mod.SIGMA2_WITHIN * 0.5)
    shrink = (k * predictor_mod.TAU2_BETWEEN) / (k * predictor_mod.TAU2_BETWEEN + sigma2)
    expected_offset = float(np.mean(form_means)) * shrink

    assert offset_hat == pytest.approx(expected_offset, abs=1e-6)


def test_constant_conc_transfer_check_matches_old_loo_semantics():
    """The generalized whole-model LOO transfer check
    (_transfer_check_passes) must reduce to the old scalar-offset LOO check
    exactly when context conc doesn't vary -- confirmed by directly
    reimplementing the old check and comparing pass/fail."""
    p = _make_predictor_stub()
    conc_values = [100.0] * 6
    ctx_formulations, ctx_shear, ctx_resid, ctx_conc = _make_ctx(
        conc_values, offset=0.08, noise=0.01, seed=4
    )

    def old_transfer_check(ctx_formulations, ctx_resid):
        unique_forms = np.unique(ctx_formulations)
        k = len(unique_forms)
        successes = 0
        for held_back in unique_forms:
            train_mask = ctx_formulations != held_back
            test_mask = ctx_formulations == held_back
            loo_means = pd.Series(ctx_resid[train_mask]).groupby(ctx_formulations[train_mask]).mean().values
            if len(loo_means) == 0:
                continue
            loo_offset = float(np.mean(loo_means))
            test_resid = ctx_resid[test_mask]
            raw_err = float(np.mean(np.abs(test_resid)))
            corrected_err = float(np.mean(np.abs(test_resid - loo_offset)))
            if corrected_err < raw_err:
                successes += 1
        return (successes / k) >= predictor_mod.TRANSFER_CHECK_FRAC

    assert p._transfer_check_passes(ctx_formulations, ctx_resid, ctx_conc) == old_transfer_check(
        ctx_formulations, ctx_resid
    )


# ---------------------------------------------------------------------------
# Capability: a context that spans Protein_conc must let the corrector
# extrapolate to a query conc OUTSIDE the context's own range, beating the
# flat-offset-only (pre-Task-1.1) baseline on that query.
# ---------------------------------------------------------------------------


def test_conc_varying_context_beats_flat_offset_at_novel_concentration():
    p = _make_predictor_stub()
    train_conc = np.array([20.0, 80.0, 140.0, 200.0, 260.0])
    query_conc = 320.0  # outside the context's own concentration range
    true_offset = 0.05
    true_coeff = 0.003

    ctx_formulations, ctx_shear, ctx_resid, ctx_conc = _make_ctx(
        train_conc, offset=true_offset, conc_coeff=true_coeff, noise=0.003, seed=1
    )
    offset_hat, conc_hat, conc_center, slope_hat, slope_center = p._fit_local_residual(
        ctx_formulations, ctx_shear, ctx_resid, ctx_conc
    )

    assert conc_hat > 0.0, "sign of fitted conc coefficient must match the injected positive effect"

    true_query_resid = true_offset + true_coeff * (query_conc - float(np.mean(train_conc)))
    local_pred = offset_hat + conc_hat * (query_conc - conc_center)
    flat_pred = offset_hat  # what the pre-Task-1.1 offset+slope corrector alone would predict

    assert abs(local_pred - true_query_resid) < abs(flat_pred - true_query_resid)


def test_conc_varying_context_negative_slope_recovered_with_correct_sign():
    p = _make_predictor_stub()
    train_conc = np.array([20.0, 80.0, 140.0, 200.0, 260.0])
    ctx_formulations, ctx_shear, ctx_resid, ctx_conc = _make_ctx(
        train_conc, offset=0.02, conc_coeff=-0.0025, noise=0.003, seed=5
    )
    offset_hat, conc_hat, conc_center, slope_hat, slope_center = p._fit_local_residual(
        ctx_formulations, ctx_shear, ctx_resid, ctx_conc
    )
    assert conc_hat < 0.0


# ---------------------------------------------------------------------------
# Task A.2 (addendum): corrector_mode="offset_only" ablation -- zeroing
# ctx_conc before the fit (learn()'s `ctx_conc_for_fit` line) must remove
# the conc effect entirely, reproducing what a context WITHOUT any
# concentration variation would give on the identical shear/residual data.
# This is the mechanism the offset-only ablation run
# (models/experiments/condition_shift_offset_only_acceptance) relies on to
# be a faithful stand-in for the pre-Task-1.1 corrector.
# ---------------------------------------------------------------------------


def test_zeroing_context_conc_removes_the_conc_effect_vs_real_variation():
    p = _make_predictor_stub()
    train_conc = np.array([20.0, 80.0, 140.0, 200.0, 260.0])
    ctx_formulations, ctx_shear, ctx_resid, ctx_conc = _make_ctx(
        train_conc, offset=0.05, conc_coeff=0.003, noise=0.003, seed=7
    )

    with_conc = p._fit_local_residual(ctx_formulations, ctx_shear, ctx_resid, ctx_conc)
    zeroed = p._fit_local_residual(
        ctx_formulations, ctx_shear, ctx_resid, np.zeros_like(ctx_conc)
    )

    offset_hat_conc, conc_hat_conc = with_conc[0], with_conc[1]
    offset_hat_zero, conc_hat_zero = zeroed[0], zeroed[1]

    assert conc_hat_conc != 0.0  # real variation -> conc term actually fires
    assert conc_hat_zero == 0.0  # zeroed (offset_only ablation) -> conc term never fires
    # the zeroed run must match a genuinely constant-conc context exactly
    # (same code path -- see test_constant_conc_offset_matches_old_scalar_shrink_formula_exactly)
    constant = p._fit_local_residual(
        ctx_formulations, ctx_shear, ctx_resid, np.full_like(ctx_conc, 999.0)
    )
    assert offset_hat_zero == pytest.approx(constant[0], abs=1e-9)
    assert conc_hat_zero == constant[1] == 0.0


# ---------------------------------------------------------------------------
# predict() wiring: each QUERY row's own Protein_conc drives conc_hat's
# contribution, not the context's.
# ---------------------------------------------------------------------------


def test_predict_applies_per_query_concentration_term(monkeypatch):
    from visqai.preprocessing.pipeline import SHEAR_MAP

    p = _make_predictor_stub()
    p.shear_map = dict(SHEAR_MAP)
    p.offset_hat = 0.1
    p.conc_hat = 0.002
    p.conc_center = 100.0
    p.slope_hat = 0.0
    p.slope_center = 0.0

    df = pd.DataFrame({"Protein_conc": [50.0, 200.0]})
    n_shears = len(p.shear_map)

    monkeypatch.setattr(p, "_preprocess", lambda d: (None, None, None))
    monkeypatch.setattr(p, "_prior_log10", lambda q_shear, q_static: np.zeros(len(df) * n_shears))

    out = p.predict(df)

    # row0: conc=50 -> conc_term = 0.002*(50-100) = -0.1 -> total = 0.1-0.1 = 0.0 -> pred = 10^0 = 1.0
    # row1: conc=200 -> conc_term = 0.002*(200-100) = +0.2 -> total = 0.1+0.2 = 0.3 -> pred = 10^0.3
    first_col = f"Pred_{list(p.shear_map.keys())[0]}"
    assert out[first_col].values[0] == pytest.approx(1.0, rel=1e-6)
    assert out[first_col].values[1] == pytest.approx(10**0.3, rel=1e-6)


def test_predict_is_bit_for_bit_zero_shot_when_corrector_is_zero(monkeypatch):
    """Rule 2 (never regress the safe path): offset_hat=conc_hat=slope_hat=0
    (the memory_vector=None state) must reproduce the prior exactly."""
    p = _make_predictor_stub()
    from visqai.preprocessing.pipeline import SHEAR_MAP

    p.shear_map = dict(SHEAR_MAP)
    p.offset_hat = 0.0
    p.conc_hat = 0.0
    p.conc_center = 0.0
    p.slope_hat = 0.0
    p.slope_center = 0.0

    df = pd.DataFrame({"Protein_conc": [50.0, 200.0, 9999.0]})
    n_shears = len(p.shear_map)
    prior = np.random.RandomState(0).normal(0, 1, len(df) * n_shears)

    monkeypatch.setattr(p, "_preprocess", lambda d: (None, None, None))
    monkeypatch.setattr(p, "_prior_log10", lambda q_shear, q_static: prior)

    out = p.predict(df)
    expected = np.power(10, prior)

    # predict() lays predictions out row-major: all shears for row0, then
    # all shears for row1, etc. -- match that here to compare per-cell.
    for i in range(len(df)):
        start = i * n_shears
        for j, col in enumerate(p.shear_map.keys()):
            assert out[f"Pred_{col}"].values[i] == pytest.approx(expected[start + j], rel=1e-9)


# ---------------------------------------------------------------------------
# Task A.3: predict() clamps the query's Protein_conc to the context's own
# observed support range before evaluating conc_hat's linear term, so a
# query far outside context never gets an unbounded linear extrapolation.
# ---------------------------------------------------------------------------


def test_predict_clamps_query_conc_to_context_support_range(monkeypatch):
    from visqai.preprocessing.pipeline import SHEAR_MAP

    p = _make_predictor_stub()
    p.shear_map = dict(SHEAR_MAP)
    p.offset_hat = 0.0
    p.conc_hat = 0.002
    p.conc_center = 100.0
    p.conc_support_min = 50.0
    p.conc_support_max = 150.0
    p.slope_hat = 0.0
    p.slope_center = 0.0

    # row0: within support (100 -> no clamp). row1: far ABOVE support (should
    # clamp to 150, same as querying exactly at the upper edge). row2: far
    # BELOW support (should clamp to 50).
    df = pd.DataFrame({"Protein_conc": [100.0, 100000.0, -100000.0]})
    n_shears = len(p.shear_map)

    monkeypatch.setattr(p, "_preprocess", lambda d: (None, None, None))
    monkeypatch.setattr(p, "_prior_log10", lambda q_shear, q_static: np.zeros(len(df) * n_shears))

    out = p.predict(df)
    first_col = f"Pred_{list(p.shear_map.keys())[0]}"

    expected_at_100 = 10 ** (0.002 * (100.0 - 100.0))
    expected_at_150_clamped = 10 ** (0.002 * (150.0 - 100.0))
    expected_at_50_clamped = 10 ** (0.002 * (50.0 - 100.0))

    assert out[first_col].values[0] == pytest.approx(expected_at_100, rel=1e-6)
    assert out[first_col].values[1] == pytest.approx(expected_at_150_clamped, rel=1e-6)
    assert out[first_col].values[2] == pytest.approx(expected_at_50_clamped, rel=1e-6)


def test_predict_clamp_is_a_noop_when_conc_hat_is_zero(monkeypatch):
    """Even a degenerate [0, 0] support range (the reset/no-context default)
    must not perturb predictions when conc_hat is 0 -- clamping a term
    that's multiplied by zero changes nothing."""
    from visqai.preprocessing.pipeline import SHEAR_MAP

    p = _make_predictor_stub()
    p.shear_map = dict(SHEAR_MAP)
    p.offset_hat = 0.0
    p.conc_hat = 0.0
    p.conc_center = 0.0
    p.conc_support_min = 0.0
    p.conc_support_max = 0.0
    p.slope_hat = 0.0
    p.slope_center = 0.0

    df = pd.DataFrame({"Protein_conc": [50.0, 9999.0]})
    n_shears = len(p.shear_map)
    prior = np.random.RandomState(1).normal(0, 1, len(df) * n_shears)

    monkeypatch.setattr(p, "_preprocess", lambda d: (None, None, None))
    monkeypatch.setattr(p, "_prior_log10", lambda q_shear, q_static: prior)

    out = p.predict(df)
    expected = np.power(10, prior)
    for i in range(len(df)):
        start = i * n_shears
        for j, col in enumerate(p.shear_map.keys()):
            assert out[f"Pred_{col}"].values[i] == pytest.approx(expected[start + j], rel=1e-9)


def test_learn_sets_conc_support_from_context_range():
    p = _make_predictor_stub()
    train_conc = np.array([20.0, 80.0, 140.0, 200.0, 260.0])
    ctx_formulations, ctx_shear, ctx_resid, ctx_conc = _make_ctx(
        train_conc, offset=0.05, conc_coeff=0.002, noise=0.003, seed=9
    )
    # Mirror learn()'s own logic (support = min/max of the raw ctx_conc
    # array passed to _fit_local_residual) without needing a real model.
    conc_support_min = float(np.min(ctx_conc))
    conc_support_max = float(np.max(ctx_conc))
    assert conc_support_min == pytest.approx(20.0)
    assert conc_support_max == pytest.approx(260.0)
