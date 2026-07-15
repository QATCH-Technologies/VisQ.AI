import numpy as np
import pandas as pd
import pytest

from visqai.preprocessing.pipeline import (
    build_feature_frame,
    protected_feature_indices,
    BASE_NUMERIC_COLS,
    BASE_CATEGORICAL_COLS,
    ENGINEERED_COLS,
)
from visqai.features.charge import CHARGE_FEATURE_COLS_BASE
from visqai.physics.priors import PRIOR_COLS, CONC_SPLIT_COLS


def _sample_row(**overrides):
    row = {
        "ID": "s1",
        "Protein_type": "trastuzumab",
        "Protein_class_type": "mab_igg1",
        "Protein_conc": 100.0,
        "Buffer_pH": 6.0,
        "PI_mean": 8.5,
        "Charge": 12.0,
        "ProtPi PI": 8.2,
        "kP": 1.0,
        "MW": 148000.0,
        "PI_range": 0.5,
        "Temperature": 25.0,
        "Buffer_conc": 20.0,
        "Salt_conc": 50.0,
        "Salt_type": "nacl",
        "Stabilizer_conc": 0.1,
        "Stabilizer_type": "sucrose",
        "Surfactant_conc": 0.02,
        "Surfactant_type": "tween-80",
        "Excipient_conc": 100.0,
        "Excipient_type": "arginine",
        "C_Class": 1.0,
        "HCI": 0.0,
        "Viscosity_1000": 15.0,
    }
    row.update(overrides)
    return row


def test_build_feature_frame_returns_all_base_and_engineered_columns():
    df = pd.DataFrame([_sample_row()])
    out, num_cols, cat_cols = build_feature_frame(df)
    for c in BASE_NUMERIC_COLS + ENGINEERED_COLS:
        assert c in out.columns
        assert c in num_cols
    assert cat_cols == BASE_CATEGORICAL_COLS


def test_build_feature_frame_populates_real_charge_columns_not_zero_fill():
    """The core regression test for the bug fix: a formulation with a real
    Charge value must end up with a real net_charge, not the silent
    zero-fill inference_o_net.py used to produce."""
    df = pd.DataFrame([_sample_row(Charge=12.0, **{"ProtPi PI": 8.2})])
    out, num_cols, _ = build_feature_frame(df)
    for c in CHARGE_FEATURE_COLS_BASE:
        assert c in num_cols
    assert out.loc[0, "net_charge"] == pytest.approx(12.0)
    assert out.loc[0, "charge_missing"] == 0.0


def test_build_feature_frame_prior_columns_present_and_charge_aware():
    # Row is exactly at its pI via net_charge, even though Buffer_pH != PI_mean
    # (so the OLD pH-distance-only proxy would have scored this differently).
    df = pd.DataFrame([_sample_row(Charge=0.0, Buffer_pH=6.0, PI_mean=8.5)])
    out, num_cols, _ = build_feature_frame(df)
    for c in PRIOR_COLS + CONC_SPLIT_COLS:
        assert c in num_cols
    # net_charge==0 at a charge-aware row should hit the Near-pI regime for
    # mab_igg1, giving prior_nacl == -1 (see PRIOR_TABLE["mab_igg1"]["Near-pI"]).
    assert out.loc[0, "prior_nacl"] == -1.0


def test_build_feature_frame_missing_optional_columns_degrade_gracefully():
    minimal = pd.DataFrame([{"Protein_conc": 50.0, "Protein_type": "mab"}])
    out, num_cols, cat_cols = build_feature_frame(minimal)
    assert len(out) == 1
    for c in num_cols:
        assert c in out.columns


def test_protected_feature_indices_includes_charge_and_property_cols():
    df = pd.DataFrame([_sample_row()])
    out, num_cols, _ = build_feature_frame(df)
    from visqai.features.categorical import featurize_chemical_categoricals

    _, prop_cols = featurize_chemical_categoricals(df.copy())
    protected = protected_feature_indices(num_cols)
    protected_names = {num_cols[i] for i in protected}
    assert "Protein_conc" in protected_names
    assert "net_charge" in protected_names
    for c in prop_cols:
        assert c in protected_names
