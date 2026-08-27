import pandas as pd
import pytest

from visqai.features.priors import (
    calculate_cci,
    calculate_regime,
    calculate_row_priors,
    CONC_THRESHOLDS,
    PRIOR_TABLE,
)


def test_cci_peaks_at_zero_ph_distance():
    cci_at_pI = calculate_cci(c_class=2.0, ph=7.0, pi=7.0)
    assert cci_at_pI == pytest.approx(2.0)  # exp(0) == 1 -> cci == c_class

    cci_far_from_pI = calculate_cci(c_class=2.0, ph=7.0, pi=4.0)
    assert cci_far_from_pI < cci_at_pI


def test_regime_thresholds_per_protein_class():
    assert calculate_regime(0.95, "mab_igg1") == "Near-pI"
    assert calculate_regime(0.60, "mab_igg1") == "Mixed"
    assert calculate_regime(0.10, "mab_igg1") == "Far"
    assert calculate_regime(0.85, "mab_igg4") == "Near-pI"
    assert calculate_regime(0.75, "unrecognized_type") == "Near-pI"


def test_calculate_row_priors_uses_ph_distance_proxy():
    row = pd.Series(
        {
            "C_Class": 1.0,
            "Buffer_pH": 7.0,
            "PI_mean": 7.0,
            "Protein_class_type": "mab_igg1",
            "Salt_type": "nacl",
            "Salt_conc": 50.0,
        }
    )
    out = calculate_row_priors(row)
    # pH == PI_mean -> cci==1.0 -> Near-pI regime for mab_igg1 -> prior_nacl == -1
    assert out["prior_nacl"] == PRIOR_TABLE["mab_igg1"]["Near-pI"]["nacl"]


def test_calculate_row_priors_concentration_split():
    row = pd.Series(
        {
            "C_Class": 1.0,
            "Buffer_pH": 7.0,
            "PI_mean": 7.0,
            "Protein_class_type": "mab_igg1",
            "Salt_type": "nacl",
            "Salt_conc": 200.0,  # above the 150.0 nacl threshold
        }
    )
    out = calculate_row_priors(row)
    threshold = CONC_THRESHOLDS["nacl"]
    # nacl_low/nacl_high are fraction-of-threshold (bounded, not raw
    # concentration) -- see priors.py's CONC_HIGH_FRAC_CAP comment.
    assert out["nacl_low"] == pytest.approx(1.0)
    assert out["nacl_high"] == pytest.approx((200.0 - threshold) / threshold)


def test_calculate_row_priors_ignores_zero_concentration_ingredient():
    row = pd.Series(
        {
            "C_Class": 1.0,
            "Buffer_pH": 7.0,
            "PI_mean": 7.0,
            "Protein_class_type": "mab_igg1",
            "Salt_type": "nacl",
            "Salt_conc": 0.0,
        }
    )
    out = calculate_row_priors(row)
    assert out["prior_nacl"] == 0.0
    assert out["nacl_low"] == 0.0
