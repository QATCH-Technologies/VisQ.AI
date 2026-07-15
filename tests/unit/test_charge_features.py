import numpy as np
import pandas as pd
import pytest

from visqai.features.charge import (
    normalize_charge_columns,
    featurize_charge,
    charge_coupling_index,
    NEAR_PI_SIGMA,
    IMPUTE_SLOPE,
)


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_charge_columns(df)
    out, _cols = featurize_charge(df)
    return out


def test_normalize_charge_columns_renames_raw_headers():
    df = pd.DataFrame({"Charge": [1.0], "ProtPi PI": [7.0], "Unnamed: 0": [0]})
    out = normalize_charge_columns(df)
    assert "_raw_charge" in out.columns
    assert "_raw_theo_pi" in out.columns
    assert not any(str(c).startswith("Unnamed") for c in out.columns)


def test_known_charge_is_used_directly_not_missing():
    df = pd.DataFrame(
        {
            "Protein_type": ["mab_igg1"],
            "Protein_conc": [50.0],
            "Buffer_pH": [6.0],
            "PI_mean": [8.2],
            "Charge": [5.0],
            "ProtPi PI": [8.0],
            "Salt_conc": [0.0],
            "Buffer_conc": [0.0],
        }
    )
    out = _featurize(df)
    assert out.loc[0, "net_charge"] == 5.0
    assert out.loc[0, "abs_charge"] == 5.0
    assert out.loc[0, "charge_missing"] == 0.0


def test_missing_charge_with_protein_present_is_imputed_not_zero_filled():
    """This is the exact trap charge_features.py's docstring warns about:
    a missing net_charge for a present protein must NOT silently become 0
    (0 == at the isoelectric point, the highest-viscosity-risk state)."""
    df = pd.DataFrame(
        {
            "Protein_type": ["mab_igg1"],
            "Protein_conc": [50.0],
            "Buffer_pH": [6.0],
            "PI_mean": [np.nan],
            "Charge": [np.nan],
            "ProtPi PI": [8.0],
            "Salt_conc": [0.0],
            "Buffer_conc": [0.0],
        }
    )
    out = _featurize(df)
    assert out.loc[0, "charge_missing"] == 1.0
    expected = IMPUTE_SLOPE * (8.0 - 6.0)
    assert out.loc[0, "net_charge"] == pytest.approx(expected)
    assert out.loc[0, "net_charge"] != 0.0


def test_no_protein_present_is_a_genuine_zero_not_missing():
    df = pd.DataFrame(
        {
            "Protein_type": ["none"],
            "Protein_conc": [0.0],
            "Buffer_pH": [7.0],
            "PI_mean": [np.nan],
            "Charge": [np.nan],
            "ProtPi PI": [np.nan],
            "Salt_conc": [0.0],
            "Buffer_conc": [0.0],
        }
    )
    out = _featurize(df)
    assert out.loc[0, "net_charge"] == 0.0
    assert out.loc[0, "charge_missing"] == 0.0  # genuine zero, not imputed-missing


def test_near_pI_peaks_at_zero_net_charge():
    df = pd.DataFrame(
        {
            "Protein_type": ["mab_igg1", "mab_igg1"],
            "Protein_conc": [50.0, 50.0],
            "Buffer_pH": [7.0, 7.0],
            "PI_mean": [7.0, 7.0],
            "Charge": [0.0, 40.0],
            "ProtPi PI": [7.0, 7.0],
            "Salt_conc": [0.0, 0.0],
            "Buffer_conc": [0.0, 0.0],
        }
    )
    out = _featurize(df)
    assert out.loc[0, "near_pI"] == pytest.approx(1.0)
    assert out.loc[1, "near_pI"] < out.loc[0, "near_pI"]


def test_charge_coupling_index_uses_net_charge_when_available():
    df = pd.DataFrame({"C_Class": [2.0], "net_charge": [0.0]})
    cci = charge_coupling_index(df)
    assert cci.iloc[0] == pytest.approx(2.0)  # exp(0) == 1, so cci == C_Class


def test_charge_coupling_index_falls_back_without_net_charge():
    df = pd.DataFrame({"C_Class": [2.0], "Buffer_pH": [7.0], "PI_mean": [7.0]})
    cci = charge_coupling_index(df)
    assert cci.iloc[0] == pytest.approx(2.0)  # |pH-PI|==0 -> exp(0)==1


def test_featurize_charge_degrades_gracefully_without_salt_or_buffer_conc():
    """Was a live bug: df.get('Salt_conc', 0.0) returned a bare float (not a
    Series) when the column was entirely absent, and pd.to_numeric() of a
    scalar has no .fillna(), raising AttributeError instead of the
    'degrades gracefully' behavior the module's own docstring promises.
    Fixed via _numeric_col() in visqai.features.charge."""
    df = pd.DataFrame(
        {
            "Protein_type": ["mab_igg1"],
            "Protein_conc": [50.0],
            "Buffer_pH": [7.0],
            "PI_mean": [7.0],
            "Charge": [1.0],
            "ProtPi PI": [7.0],
        }
    )
    out = _featurize(df)
    assert out.loc[0, "net_charge"] == 1.0
    assert out.loc[0, "charge_screened"] >= 0.0


def test_featurize_charge_degrades_gracefully_without_any_charge_columns():
    """Same bug, different trigger: no 'Charge'/'ProtPi PI' columns at all
    (an older, pre-Rung-2 CSV) -- normalize_charge_columns never creates
    _raw_charge/_raw_theo_pi, so featurize_charge must still produce the
    physically-correct null block instead of raising."""
    df = pd.DataFrame(
        {
            "Protein_type": ["mab_igg1"],
            "Protein_conc": [50.0],
            "Buffer_pH": [7.0],
            "PI_mean": [7.0],
        }
    )
    out = _featurize(df)
    assert out.loc[0, "charge_missing"] == 1.0  # protein present, charge unresolvable
    assert np.isfinite(out.loc[0, "net_charge"])
