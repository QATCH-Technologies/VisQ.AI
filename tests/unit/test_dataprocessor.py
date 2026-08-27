import numpy as np
import pandas as pd
import pytest

from visqai.features.dataprocessor import (
    build_feature_frame,
    protected_feature_indices,
    BASE_NUMERIC_COLS,
    BASE_CATEGORICAL_COLS,
    ENGINEERED_COLS,
)
from visqai.features.charge import CHARGE_FEATURE_COLS
from visqai.features.priors import PRIOR_COLS, CONC_SPLIT_COLS, calculate_cci, calculate_regime


def _sample_row(**overrides):
    row = {
        "ID": "s1",
        "Protein_type": "trastuzumab",
        "Protein_class_type": "mab_igg1",
        "Protein_conc": 100.0,
        "Buffer_pH": 6.0,
        "PI_mean": 8.5,
        "Whole_Antibody_Charge_at_Buffer_pH": 12.0,
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


def test_build_feature_frame_populates_real_whole_charge_not_zero_fill():
    df = pd.DataFrame(
        [_sample_row(Protein_type="synthetic_igg1", **{"Whole_Antibody_Charge_at_Buffer_pH": 12.0})]
    )
    out, num_cols, _ = build_feature_frame(df)
    for c in CHARGE_FEATURE_COLS:
        assert c in num_cols
    assert out.loc[0, "whole_charge"] == pytest.approx(12.0)


def test_build_feature_frame_prior_columns_present():
    df = pd.DataFrame([_sample_row(Buffer_pH=8.5, PI_mean=8.5)])
    out, num_cols, _ = build_feature_frame(df)
    for c in PRIOR_COLS + CONC_SPLIT_COLS:
        assert c in num_cols
    # Buffer_pH == PI_mean -> the pH-distance CCI proxy hits Near-pI.
    cci = calculate_cci(c_class=1.0, ph=8.5, pi=8.5)
    assert calculate_regime(cci, "mab_igg1") == "Near-pI"
    # and the regime-dict lookup actually ran (stabilizer prior is always
    # populated when Stabilizer_type is present, regardless of regime).
    assert out.loc[0, "prior_stabilizer"] == 1.0


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
    assert "whole_charge" in protected_names
    for c in prop_cols:
        assert c in protected_names


def test_salt_mg_ml_is_capped_at_construction_not_left_unbounded():
    """P0 fix: Salt_mg_mL used to be an unbounded Salt_conc * salt_mw product
    that could inject an outsized raw magnitude into Total_Solute_Mass /
    Effective_Protein_Fraction under a leave-one-salt-out fold. It's now
    clipped to SALT_MG_ML_CAP at feature-construction time, independent of
    whatever a fold's StandardScaler happens to learn."""
    from visqai.features.dataprocessor import SALT_MG_ML_CAP

    df = pd.DataFrame([_sample_row(Salt_type="nacl", Salt_conc=1_000_000.0)])
    out, num_cols, _ = build_feature_frame(df)
    assert out.loc[0, "Salt_mg_mL"] == pytest.approx(SALT_MG_ML_CAP)
    # And it still actually feeds Total_Solute_Mass (not silently dropped).
    assert out.loc[0, "Total_Solute_Mass"] >= SALT_MG_ML_CAP


def test_num_cols_never_carries_a_raw_salt_molecular_weight():
    df = pd.DataFrame([_sample_row()])
    _, num_cols, _ = build_feature_frame(df)
    assert not any("salt_mw" in c for c in num_cols)
