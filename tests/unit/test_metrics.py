import numpy as np
import pandas as pd
import pytest

from visqai.eval.metrics import (
    calc_metrics,
    compute_metrics,
    _log10_safe,
    check_against_noise_band,
    AGGREGATE_MDE,
    PER_FOLD_RUN_SD,
)
from visqai.eval.constants import PRED_COLS, VISC_COLS
from visqai.eval.data_prep import prepare_df


def test_calc_metrics_perfect_prediction_is_zero_error():
    m = calc_metrics([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["within_2x"] == pytest.approx(100.0)
    assert m["n"] == 3


def test_calc_metrics_masks_non_positive_and_nonfinite():
    true = [10.0, -5.0, np.nan, 20.0]
    pred = [10.0, 5.0, 5.0, 0.0]  # last pred is non-positive -> dropped too
    m = calc_metrics(true, pred)
    assert m["n"] == 1  # only the first pair (10,10) survives masking


def test_calc_metrics_empty_after_masking_returns_nans():
    m = calc_metrics([-1.0, -2.0], [-1.0, -2.0])
    assert m["n"] == 0
    assert np.isnan(m["mae"])


def test_log10_safe_clips_instead_of_dropping():
    out = _log10_safe(np.array([0.0, -1.0, 100.0]))
    assert out[0] == pytest.approx(np.log10(1e-6))
    assert out[1] == pytest.approx(np.log10(1e-6))
    assert out[2] == pytest.approx(2.0)


def test_compute_metrics_scoped_to_visc_cols():
    results_df = pd.DataFrame({PRED_COLS[0]: [12.0, 18.0]})
    truth_df = pd.DataFrame({VISC_COLS[0]: [10.0, 20.0]})
    m = compute_metrics(results_df, truth_df)
    assert m["mae"] == pytest.approx(2.0)
    assert set(m.keys()) == {"mae", "rmse", "mape", "mae_log10", "rmse_log10"}


def test_compute_metrics_and_calc_metrics_disagree_on_nonpositive_values():
    """The documented divergence: compute_metrics clips non-positive values
    into the mean (via _log10_safe), calc_metrics drops them entirely."""
    results_df = pd.DataFrame({PRED_COLS[0]: [10.0, -5.0]})
    truth_df = pd.DataFrame({VISC_COLS[0]: [10.0, 20.0]})

    cm = compute_metrics(results_df, truth_df)
    am = calc_metrics(truth_df[VISC_COLS[0]].values, results_df[PRED_COLS[0]].values)

    assert cm["mae"] == pytest.approx(np.mean([0.0, 25.0]))  # both rows included
    assert am["n"] == 1  # negative pred dropped


def test_compute_metrics_missing_columns_returns_nan_dict():
    m = compute_metrics(pd.DataFrame({"other": [1.0]}), pd.DataFrame({"other": [1.0]}))
    assert np.isnan(m["mae"])


def test_prepare_df_coerces_ints_and_id_to_str():
    df = pd.DataFrame({"ID": [1, 2], "kP": [3, 4]})
    out = prepare_df(df)
    assert out["ID"].dtype == object
    assert list(out["ID"]) == ["1", "2"]
    assert out["kP"].dtype == float


def test_prepare_df_drop_bad_rows_filters_invalid_viscosity_and_numerics():
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "MW": [100.0, np.nan, 100.0],
            "Protein_conc": [50.0, 50.0, 50.0],
            "kP": [1.0, 1.0, 1.0],
            VISC_COLS[0]: [10.0, 10.0, -1.0],
        }
    )
    out = prepare_df(df, drop_bad_rows=True)
    assert list(out["ID"]) == ["1"]


def test_prepare_df_default_matches_no_drop_behavior():
    df = pd.DataFrame({"ID": [1], VISC_COLS[0]: [-1.0]})
    out = prepare_df(df)  # drop_bad_rows defaults False
    assert len(out) == 1


def test_check_against_noise_band_single_measurement_uses_raw_sd():
    """is_difference=False: a single run's own deviation from a fixed
    reference is compared directly against PER_FOLD_RUN_SD, no scaling."""
    assert check_against_noise_band(PER_FOLD_RUN_SD * 0.9, scope="per_fold", is_difference=False) is True
    assert check_against_noise_band(PER_FOLD_RUN_SD * 1.1, scope="per_fold", is_difference=False) is False


def test_check_against_noise_band_difference_scales_as_sqrt_2_over_k():
    """Task G's C3 fix: a difference of two independently-measured
    stochastic quantities gets a WIDER band than a single measurement --
    sigma*sqrt(2) at 1 seed each, narrowing toward sigma*sqrt(2/k) as more
    seeds are averaged per side. This is the relationship whose absence (one
    stored 'band' number applied to both single measurements and
    differences alike) was the Task follow-on's C3 error."""
    band_1_seed = PER_FOLD_RUN_SD * (2.0 / 1) ** 0.5
    band_5_seed = PER_FOLD_RUN_SD * (2.0 / 5) ** 0.5
    assert band_5_seed < band_1_seed  # more seeds -> tighter band
    assert band_5_seed == pytest.approx(PER_FOLD_RUN_SD * 0.6325, rel=1e-3)

    just_inside_5seed = band_5_seed * 0.99
    just_outside_5seed_but_inside_1seed = (band_5_seed + band_1_seed) / 2
    assert check_against_noise_band(just_inside_5seed, scope="per_fold", is_difference=True, n_seeds=5) is True
    assert check_against_noise_band(just_outside_5seed_but_inside_1seed, scope="per_fold", is_difference=True, n_seeds=5) is False
    assert check_against_noise_band(just_outside_5seed_but_inside_1seed, scope="per_fold", is_difference=True, n_seeds=1) is True


def test_check_against_noise_band_aggregate_scope_ignores_difference_and_seeds():
    """The aggregate scope is always a single fixed threshold (AGGREGATE_MDE)
    -- is_difference/n_seeds are per_fold-only concepts."""
    assert check_against_noise_band(0.01, scope="aggregate", is_difference=True, n_seeds=5) is True
    assert check_against_noise_band(0.02, scope="aggregate", is_difference=False) is False


def test_check_against_noise_band_rejects_unknown_scope():
    with pytest.raises(ValueError):
        check_against_noise_band(0.01, scope="per_protein", is_difference=False)


def test_check_against_noise_band_per_fold_requires_is_difference():
    """No default: a caller must state whether its number is a difference,
    per Task G -- omitting it is a TypeError (is_difference is a required
    keyword-only argument), not a silently-assumed False."""
    with pytest.raises(TypeError):
        check_against_noise_band(0.01, scope="per_fold")
