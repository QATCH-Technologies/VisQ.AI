import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from visqai.training.data import ZERO_VARIANCE_FALLBACK_SCALE, _drop_blank_rows, _fix_zero_variance_scale


def test_zero_variance_column_gets_fallback_scale_not_degenerate_one():
    """P0 fix: sklearn's StandardScaler sets scale_=1 for a zero-variance
    training column (see _handle_zeros_in_scale), which means an OOD held-out
    value passes through the fitted transform almost raw. This is the
    at-fit-time guard that replaces that degenerate 1.0 with a fixed,
    generous a-priori scale instead."""
    num_cols = ["constant_col", "varying_col"]
    df = pd.DataFrame(
        {
            "constant_col": [5.0, 5.0, 5.0, 5.0],
            "varying_col": [1.0, 2.0, 3.0, 4.0],
        }
    )
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), num_cols)])
    X = preprocessor.fit_transform(df)

    scaler = preprocessor.named_transformers_["num"]
    assert scaler.scale_[0] == 1.0  # sklearn's degenerate default, pre-fix
    assert scaler.var_[0] == 0.0
    assert scaler.var_[1] > 0.0  # sanity: the varying column is untouched

    X_fixed = _fix_zero_variance_scale(preprocessor, X, num_cols)

    assert scaler.scale_[0] == ZERO_VARIANCE_FALLBACK_SCALE
    assert scaler.scale_[1] != ZERO_VARIANCE_FALLBACK_SCALE  # varying col left alone

    # An OOD held-out value for the constant column now maps to a bounded
    # z-score under the fixed scale, not the raw (value - mean) it would
    # have gotten under sklearn's scale_=1.
    ood_value = np.array([[500.0, 2.5]])
    z = preprocessor.transform(pd.DataFrame(ood_value, columns=num_cols))
    raw_passthrough_z = (500.0 - scaler.mean_[0]) / 1.0
    assert abs(z[0, 0]) < abs(raw_passthrough_z)
    assert z[0, 0] == pytest.approx((500.0 - scaler.mean_[0]) / ZERO_VARIANCE_FALLBACK_SCALE)


def test_no_zero_variance_columns_leaves_scaler_untouched():
    num_cols = ["a", "b"]
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), num_cols)])
    X = preprocessor.fit_transform(df)
    scaler = preprocessor.named_transformers_["num"]
    original_scale = scaler.scale_.copy()

    X_fixed = _fix_zero_variance_scale(preprocessor, X, num_cols)

    assert np.array_equal(scaler.scale_, original_scale)
    assert np.array_equal(X_fixed, X)


# ---------------------------------------------------------------------------
# Pre-2.4 data-hygiene fix (issue1_query_conditioned_correction_plan.md
# Finding C): fully-blank rows (trailing/malformed CSV rows -- no
# Protein_type AND no Protein_conc) must be dropped before load_and_preprocess
# fits the StandardScaler, or they contaminate it as all-zero(ish) rows.
# ---------------------------------------------------------------------------


def test_drop_blank_rows_removes_fully_blank_rows():
    df = pd.DataFrame(
        {
            "Protein_type": ["mab", None, "mab2"],
            "Protein_conc": [100.0, None, 150.0],
            "kP": [3.0, None, 3.5],
        }
    )
    out = _drop_blank_rows(df)
    assert len(out) == 2
    assert list(out["Protein_type"]) == ["mab", "mab2"]


def test_drop_blank_rows_keeps_row_with_only_protein_conc():
    """A row missing Protein_type but with a real Protein_conc is NOT the
    blank/malformed case this targets -- keep it (matches the 86 legitimate
    Protein_type=='none' buffer-only rows, which have a real Protein_conc of
    0.0, i.e. not NaN, so this filter never touches them)."""
    df = pd.DataFrame(
        {
            "Protein_type": [None, "mab"],
            "Protein_conc": [50.0, 100.0],
        }
    )
    out = _drop_blank_rows(df)
    assert len(out) == 2


def test_drop_blank_rows_leftover_stray_value_still_dropped():
    """The real-data case: a row with Protein_type AND Protein_conc both NaN,
    but a stray leftover Viscosity_100 value -- still fully blank by this
    filter's definition and must be dropped."""
    df = pd.DataFrame(
        {
            "Protein_type": ["mab", None],
            "Protein_conc": [100.0, None],
            "Viscosity_100": [12.0, 19.0],
        }
    )
    out = _drop_blank_rows(df)
    assert len(out) == 1
    assert out["Protein_type"].iloc[0] == "mab"


def test_drop_blank_rows_no_blanks_is_a_noop():
    df = pd.DataFrame({"Protein_type": ["mab", "mab2"], "Protein_conc": [100.0, 150.0]})
    out = _drop_blank_rows(df)
    assert len(out) == 2
