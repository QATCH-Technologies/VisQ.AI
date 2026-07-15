import numpy as np
import pytest

from visqai.eval.shape_metrics import (
    _classify_slopes,
    compute_shape_metrics,
    _aggregate_shape,
    SHAPE_FLAT_EPS,
)


def test_classify_slopes_signs():
    log_visc = np.array([2.0, 1.5, 1.5, 2.5])  # thin, flat, thicken
    out = _classify_slopes(log_visc)
    assert list(out) == [-1.0, 0.0, 1.0]


def test_identical_curves_have_zero_shape_error():
    actual = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
    m = compute_shape_metrics(actual, actual.copy())
    assert m["shape_rmse_log10"] == pytest.approx(0.0, abs=1e-9)
    assert m["slope_sign_agree"] == pytest.approx(1.0)
    assert m["thin_ratio_log_err"] == pytest.approx(0.0, abs=1e-9)


def test_opposite_direction_curve_disagrees_on_slope_sign():
    actual = np.array([20.0, 40.0, 60.0, 80.0, 100.0])  # thickening
    pred = np.array([100.0, 80.0, 60.0, 40.0, 20.0])  # thinning
    m = compute_shape_metrics(actual, pred)
    assert m["slope_sign_agree"] == pytest.approx(0.0)


def test_flat_actual_segments_excluded_from_slope_agreement():
    actual = np.array([50.0, 50.0, 50.0])  # perfectly flat -> no sloped segments
    pred = np.array([50.0, 60.0, 40.0])
    m = compute_shape_metrics(actual, pred)
    assert m["slope_n_sloped"] == 0
    assert np.isnan(m["slope_sign_agree"])


def test_fewer_than_two_valid_points_returns_all_nan():
    m = compute_shape_metrics(np.array([np.nan, np.nan, 1.0]), np.array([1.0, np.nan, np.nan]))
    assert np.isnan(m["shape_rmse_log10"])
    assert m["slope_n_sloped"] == 0


def test_aggregate_shape_is_segment_weighted_for_slope_agreement():
    profs = {
        "a": {
            "shape_rmse_log10": 0.1,
            "slope_sign_agree": 1.0,
            "slope_n_sloped": 4,
            "plateau_err_log10": 0.0,
            "thin_ratio_log_err": 0.0,
        },
        "b": {
            "shape_rmse_log10": 0.3,
            "slope_sign_agree": 0.0,
            "slope_n_sloped": 1,
            "plateau_err_log10": 0.0,
            "thin_ratio_log_err": 0.0,
        },
    }
    agg = _aggregate_shape(profs)
    assert agg["shape_rmse_log10"] == pytest.approx(0.2)
    # segment-weighted: (1.0*4 + 0.0*1) / 5 = 0.8, not the plain mean (0.5)
    assert agg["slope_sign_agree"] == pytest.approx(0.8)


def test_aggregate_shape_empty_returns_nans():
    agg = _aggregate_shape({})
    assert np.isnan(agg["shape_rmse_log10"])
