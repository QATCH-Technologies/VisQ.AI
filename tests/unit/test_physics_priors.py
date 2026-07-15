import numpy as np
import pandas as pd
import pytest

from visqai.physics.priors import (
    calculate_cci,
    calculate_regime,
    calculate_row_priors,
    CONC_THRESHOLDS,
    PRIOR_TABLE,
)
from visqai.features.charge import NEAR_PI_SIGMA


def test_cci_uses_charge_aware_branch_when_net_charge_present():
    """This is the direct regression test for the charge-features bug fix:
    inference_o_net.py's old _calculate_cci never took this branch at all."""
    cci_at_pI = calculate_cci(c_class=2.0, ph=7.0, pi=7.0, net_charge=0.0)
    assert cci_at_pI == pytest.approx(2.0)  # exp(0) == 1 -> cci == c_class

    cci_far_from_pI = calculate_cci(c_class=2.0, ph=7.0, pi=7.0, net_charge=40.0)
    assert cci_far_from_pI < cci_at_pI


def test_cci_falls_back_to_ph_distance_proxy_without_net_charge():
    cci = calculate_cci(c_class=2.0, ph=7.0, pi=7.0, net_charge=None)
    assert cci == pytest.approx(2.0)  # |pH-PI|==0 -> exp(0)==1

    cci_nan = calculate_cci(c_class=2.0, ph=7.0, pi=7.0, net_charge=np.nan)
    assert cci_nan == pytest.approx(2.0)  # NaN net_charge also falls back


def test_charge_aware_and_proxy_can_disagree_for_the_same_row():
    """The whole point of the bug fix: for a row with a nonzero net_charge but
    pH == PI (so the old proxy says 'at pI'), the charge-aware branch can
    correctly say the protein is NOT actually at its isoelectric point."""
    ph, pi = 7.0, 7.0
    proxy_cci = calculate_cci(1.0, ph, pi, net_charge=None)
    charge_cci = calculate_cci(1.0, ph, pi, net_charge=30.0)
    assert proxy_cci == pytest.approx(1.0)
    assert charge_cci < proxy_cci


def test_regime_thresholds_per_protein_class():
    assert calculate_regime(0.95, "mab_igg1") == "Near-pI"
    assert calculate_regime(0.60, "mab_igg1") == "Mixed"
    assert calculate_regime(0.10, "mab_igg1") == "Far"
    assert calculate_regime(0.85, "mab_igg4") == "Near-pI"
    assert calculate_regime(0.75, "unrecognized_type") == "Near-pI"


def test_calculate_row_priors_picks_up_net_charge_when_present():
    row_with_charge = pd.Series(
        {
            "C_Class": 1.0,
            "Buffer_pH": 7.0,
            "PI_mean": 7.0,
            "net_charge": 0.0,
            "Protein_class_type": "mab_igg1",
            "Salt_type": "nacl",
            "Salt_conc": 50.0,
        }
    )
    out = calculate_row_priors(row_with_charge)
    # net_charge==0 -> cci==1.0 -> Near-pI regime for mab_igg1 -> prior_nacl == -1
    assert out["prior_nacl"] == PRIOR_TABLE["mab_igg1"]["Near-pI"]["nacl"]


def test_calculate_row_priors_without_net_charge_uses_proxy():
    row_no_charge = pd.Series(
        {
            "C_Class": 1.0,
            "Buffer_pH": 7.0,
            "PI_mean": 7.0,
            "Protein_class_type": "mab_igg1",
            "Salt_type": "nacl",
            "Salt_conc": 50.0,
        }
    )
    out = calculate_row_priors(row_no_charge)
    assert out["prior_nacl"] == PRIOR_TABLE["mab_igg1"]["Near-pI"]["nacl"]


def test_calculate_row_priors_concentration_split():
    row = pd.Series(
        {
            "C_Class": 1.0,
            "Buffer_pH": 7.0,
            "PI_mean": 7.0,
            "net_charge": 0.0,
            "Protein_class_type": "mab_igg1",
            "Salt_type": "nacl",
            "Salt_conc": 200.0,  # above the 150.0 nacl threshold
        }
    )
    out = calculate_row_priors(row)
    threshold = CONC_THRESHOLDS["nacl"]
    assert out["nacl_low"] == pytest.approx(threshold)
    assert out["nacl_high"] == pytest.approx(200.0 - threshold)


def test_calculate_row_priors_ignores_zero_concentration_ingredient():
    row = pd.Series(
        {
            "C_Class": 1.0,
            "Buffer_pH": 7.0,
            "PI_mean": 7.0,
            "net_charge": 0.0,
            "Protein_class_type": "mab_igg1",
            "Salt_type": "nacl",
            "Salt_conc": 0.0,
        }
    )
    out = calculate_row_priors(row)
    assert out["prior_nacl"] == 0.0
    assert out["nacl_low"] == 0.0
