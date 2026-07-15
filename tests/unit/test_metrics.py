import numpy as np
import pandas as pd
import pytest

from visqai.eval.metrics import calc_metrics, compute_metrics, _log10_safe
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
