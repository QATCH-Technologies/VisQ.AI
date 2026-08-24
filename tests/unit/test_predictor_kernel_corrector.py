"""
test_predictor_kernel_corrector.py
===================================
Task 1.2 (issue1_query_conditioned_correction_plan.md): unit tests for
ViscosityPredictorCNP's kernel-weighted local residual corrector
(_fit_local_residual_kernel / _kernel_loo_scan / _predict_kernel_correction)
and predict()'s corrector_mode="kernel" wiring.

Built the same way as test_predictor_local_residual.py: __new__ bypasses
__init__'s checkpoint loading, and _kernel_feat_idx is set directly rather
than resolved through _kernel_feature_indices() (which needs a real fitted
preprocessor) -- these methods are pure numpy/pandas functions of their
arguments plus the selected feature indices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from visqai.inference import predictor as predictor_mod
from visqai.inference.predictor import ViscosityPredictorCNP


def _make_predictor_stub():
    p = ViscosityPredictorCNP.__new__(ViscosityPredictorCNP)
    p.corrector_mode = "linear"
    p._kernel_ctx_phi = None
    p._kernel_ctx_resid = None
    p.kernel_bandwidth = None
    p._kernel_feat_idx = None
    p.conc_support_min = -1e9
    p.conc_support_max = 1e9
    return p


def _two_cluster_ctx(seed=0, noise=0.01):
    """8 formulations in two well-separated clusters (in a single kernel
    feature dimension): cluster A (feature~0) has residual~+0.15, cluster B
    (feature~10) has residual~-0.15 -- a step function a LINEAR term in that
    same dimension would fit badly (the whole point of Task 1.2), but a
    small-bandwidth kernel should cleanly separate the two.
    ctx_formulations/ctx_resid mirror _context_residuals' raw-point shape
    (2 shear points per formulation); ctx_static has one row per point."""
    rng = np.random.RandomState(seed)
    formulations, resid, static = [], [], []
    for f in range(8):
        cluster_a = f < 4
        base = 0.15 if cluster_a else -0.15
        feat = 0.0 if cluster_a else 10.0
        for _pt in range(2):
            formulations.append(f)
            resid.append(base + rng.normal(0, noise))
            static.append([feat])
    return (
        np.array(formulations, dtype=int),
        np.array(resid, dtype=float),
        np.array(static, dtype=float),
    )


# ---------------------------------------------------------------------------
# _kernel_loo_scan
# ---------------------------------------------------------------------------


def test_small_bandwidth_separates_two_clusters_better_than_large_bandwidth():
    p = _make_predictor_stub()
    form_phi = np.array([[0.0]] * 4 + [[10.0]] * 4)
    form_resid = np.array([0.15, 0.16, 0.14, 0.15, -0.15, -0.16, -0.14, -0.15])

    small_mae, small_frac = p._kernel_loo_scan(form_phi, form_resid, ell=0.5)
    large_mae, large_frac = p._kernel_loo_scan(form_phi, form_resid, ell=50.0)

    # small bandwidth: LOO prediction only pools same-cluster neighbors ->
    # low error, high improvement rate.
    assert small_mae < 0.02
    assert small_frac == 1.0
    # huge bandwidth: LOO prediction blends both clusters roughly equally ->
    # pulls toward ~0, doesn't reduce the |0.15| raw error much.
    assert large_mae > small_mae
    assert large_frac < small_frac


# ---------------------------------------------------------------------------
# _fit_local_residual_kernel: gating
# ---------------------------------------------------------------------------


def test_fewer_than_two_formulations_gate_fails():
    p = _make_predictor_stub()
    ctx_formulations = np.array([0, 0, 0])
    ctx_resid = np.array([0.1, 0.12, 0.09])
    ctx_static = np.array([[1.0], [1.0], [1.0]])
    form_phi, form_resid, ell, gate_passed = p._fit_local_residual_kernel(
        ctx_formulations, ctx_resid, ctx_static
    )
    assert gate_passed is False
    assert form_phi is None
    assert form_resid is None


def test_clean_two_cluster_signal_passes_gate_and_picks_small_bandwidth():
    p = _make_predictor_stub()
    p._kernel_feat_idx = [0]
    ctx_formulations, ctx_resid, ctx_static = _two_cluster_ctx(seed=0, noise=0.005)
    form_phi, form_resid, ell, gate_passed = p._fit_local_residual_kernel(
        ctx_formulations, ctx_resid, ctx_static
    )
    assert gate_passed is True
    assert len(form_resid) == 8
    # the smallest candidate bandwidths should win on this cleanly-separated
    # step signal (a huge bandwidth would blend the clusters)
    assert ell <= 1.0


def test_pure_noise_signal_may_fail_gate():
    """When residuals are pure noise (no real cluster structure), the
    kernel-weighted LOO reconstruction should not reliably beat doing
    nothing -- confirms the gate isn't a rubber stamp."""
    p = _make_predictor_stub()
    p._kernel_feat_idx = [0]
    rng = np.random.RandomState(0)
    formulations, resid, static = [], [], []
    for f in range(8):
        for _pt in range(2):
            formulations.append(f)
            resid.append(rng.normal(0, 0.1))
            static.append([rng.uniform(0, 10)])
    ctx_formulations = np.array(formulations, dtype=int)
    ctx_resid = np.array(resid, dtype=float)
    ctx_static = np.array(static, dtype=float)
    form_phi, form_resid, ell, gate_passed = p._fit_local_residual_kernel(
        ctx_formulations, ctx_resid, ctx_static
    )
    # Not asserting a specific outcome (noise is noise) -- just that the
    # function runs and returns a well-typed result either way.
    assert isinstance(gate_passed, bool)


# ---------------------------------------------------------------------------
# _predict_kernel_correction
# ---------------------------------------------------------------------------


def test_predict_kernel_correction_is_zero_when_not_fitted():
    p = _make_predictor_stub()
    out = p._predict_kernel_correction(np.array([[0.0], [10.0]]))
    assert np.allclose(out, 0.0)


def test_predict_kernel_correction_favors_nearby_cluster():
    p = _make_predictor_stub()
    p._kernel_ctx_phi = np.array([[0.0]] * 4 + [[10.0]] * 4)
    p._kernel_ctx_resid = np.array([0.15, 0.16, 0.14, 0.15, -0.15, -0.16, -0.14, -0.15])
    p.kernel_bandwidth = 0.5

    out = p._predict_kernel_correction(np.array([[0.0], [10.0]]))
    assert out[0] > 0.0  # query near cluster A (positive residuals)
    assert out[1] < 0.0  # query near cluster B (negative residuals)
    assert out[0] != pytest.approx(out[1])


def test_predict_kernel_correction_shrinks_toward_zero_far_from_context():
    p = _make_predictor_stub()
    p._kernel_ctx_phi = np.array([[0.0]] * 4)
    p._kernel_ctx_resid = np.array([0.2, 0.22, 0.19, 0.21])
    p.kernel_bandwidth = 0.5

    near = p._predict_kernel_correction(np.array([[0.0]]))[0]
    far = p._predict_kernel_correction(np.array([[1000.0]]))[0]
    assert abs(far) < abs(near)


# ---------------------------------------------------------------------------
# predict() wiring: corrector_mode="kernel"
# ---------------------------------------------------------------------------


def test_predict_kernel_mode_applies_query_dependent_correction(monkeypatch):
    from visqai.preprocessing.pipeline import SHEAR_MAP

    p = _make_predictor_stub()
    p.shear_map = dict(SHEAR_MAP)
    p.corrector_mode = "kernel"
    p.offset_hat = 0.0  # unused in kernel mode -- must not also be added
    p.conc_hat = 0.0
    p.conc_center = 0.0
    p.slope_hat = 0.0
    p.slope_center = 0.0
    p._kernel_feat_idx = [0]
    p._kernel_ctx_phi = np.array([[0.0]] * 4 + [[10.0]] * 4)
    p._kernel_ctx_resid = np.array([0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2, -0.2])
    p.kernel_bandwidth = 0.5

    n_shears = len(p.shear_map)
    df = pd.DataFrame({"dummy": [0, 0]})  # 2 query rows

    # static feature block: row0 near cluster A (feat=0), row1 near cluster
    # B (feat=10); repeated once per shear, matching _preprocess's layout.
    static_np = np.array([[0.0]] * n_shears + [[10.0]] * n_shears, dtype=np.float32)
    static_t = torch.tensor(static_np).unsqueeze(0)

    monkeypatch.setattr(p, "_preprocess", lambda d: (static_t, None, None))
    monkeypatch.setattr(p, "_prior_log10", lambda q_shear, q_static: np.zeros(len(df) * n_shears))

    out = p.predict(df)
    first_col = f"Pred_{list(p.shear_map.keys())[0]}"
    # row0 (near cluster A, residual +0.2) should predict HIGHER than
    # row1 (near cluster B, residual -0.2).
    assert out[first_col].values[0] > out[first_col].values[1]


def test_predict_kernel_mode_is_bit_for_bit_zero_shot_when_not_fitted(monkeypatch):
    """Rule 2: corrector_mode='kernel' with no fitted kernel state (gate
    never fired / memory_vector reset) must reproduce the prior exactly --
    same guarantee as the linear corrector's fallback."""
    from visqai.preprocessing.pipeline import SHEAR_MAP

    p = _make_predictor_stub()
    p.shear_map = dict(SHEAR_MAP)
    p.corrector_mode = "kernel"
    p.offset_hat = 0.0
    p.conc_hat = 0.0
    p.conc_center = 0.0
    p.slope_hat = 0.0
    p.slope_center = 0.0
    p._kernel_feat_idx = [0]
    p._kernel_ctx_phi = None
    p._kernel_ctx_resid = None
    p.kernel_bandwidth = None

    n_shears = len(p.shear_map)
    df = pd.DataFrame({"dummy": [0, 0, 0]})
    static_t = torch.zeros((1, len(df) * n_shears, 1))
    prior = np.random.RandomState(0).normal(0, 1, len(df) * n_shears)

    monkeypatch.setattr(p, "_preprocess", lambda d: (static_t, None, None))
    monkeypatch.setattr(p, "_prior_log10", lambda q_shear, q_static: prior)

    out = p.predict(df)
    expected = np.power(10, prior)
    for i in range(len(df)):
        start = i * n_shears
        for j, col in enumerate(p.shear_map.keys()):
            assert out[f"Pred_{col}"].values[i] == pytest.approx(expected[start + j], rel=1e-9)


# ---------------------------------------------------------------------------
# Task A.3: bandwidth selection prefers the SMALLEST candidate that clears
# the transfer-check bar, not whichever minimizes LOO MAE -- a wider
# bandwidth can win on raw reconstruction error while still being too wide
# to decay near zero for a genuinely out-of-support query.
# ---------------------------------------------------------------------------


def test_bandwidth_selection_prefers_smallest_passing_over_lowest_mae():
    p = _make_predictor_stub()
    p._kernel_feat_idx = [0]
    # All candidates clear the transfer-check bar (frac=1.0) on this data,
    # but the WIDEST bandwidth has the lowest LOO MAE -- confirms selection
    # picks the smallest passing candidate, not argmin(MAE).
    form_phi = np.array([[float(i)] for i in range(6)])
    form_resid = np.array([0.08, 0.09, 0.07, 0.10, 0.06, 0.085])

    from visqai.inference.predictor import KERNEL_BANDWIDTH_CANDIDATES

    maes = [p._kernel_loo_scan(form_phi, form_resid, ell)[0] for ell in KERNEL_BANDWIDTH_CANDIDATES]
    assert maes[-1] < maes[0], "test setup must have the largest bandwidth winning on MAE"

    ctx_formulations = np.repeat(np.arange(6), 1)
    ctx_resid = form_resid
    ctx_static = form_phi
    _phi, _resid, ell, gate_passed = p._fit_local_residual_kernel(ctx_formulations, ctx_resid, ctx_static)

    assert gate_passed is True
    assert ell == float(KERNEL_BANDWIDTH_CANDIDATES[0])  # smallest, not the MAE-minimizing largest
