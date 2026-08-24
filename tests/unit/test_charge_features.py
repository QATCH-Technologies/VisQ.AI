import numpy as np
import pandas as pd

from visqai.features.charge import normalize_charge_columns, featurize_charge, CHARGE_FEATURE_COLS


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_charge_columns(df)
    out, _cols = featurize_charge(df)
    return out


def test_normalize_charge_columns_renames_whole_antibody_header():
    df = pd.DataFrame({"Whole_Antibody_Charge_at_Buffer_pH": [5.0], "Unnamed: 0": [0]})
    out = normalize_charge_columns(df)
    assert "_raw_whole_charge" in out.columns
    assert not any(str(c).startswith("Unnamed") for c in out.columns)


def test_normalize_charge_columns_recognizes_whole_charge_alias():
    df = pd.DataFrame({"Whole_Charge": [5.0]})
    out = normalize_charge_columns(df)
    assert out.loc[0, "_raw_whole_charge"] == 5.0


def test_known_whole_charge_is_used_directly():
    df = pd.DataFrame({"Whole_Antibody_Charge_at_Buffer_pH": [5.0]})
    out = _featurize(df)
    assert out.loc[0, "whole_charge"] == 5.0


def test_missing_whole_charge_degrades_to_zero():
    df = pd.DataFrame({"Whole_Antibody_Charge_at_Buffer_pH": [np.nan]})
    out = _featurize(df)
    assert out.loc[0, "whole_charge"] == 0.0


def test_featurize_charge_degrades_gracefully_without_any_charge_column():
    df = pd.DataFrame({"Protein_type": ["mab_igg1"], "Protein_conc": [50.0]})
    out = _featurize(df)
    assert out.loc[0, "whole_charge"] == 0.0


def test_featurize_charge_returns_charge_feature_cols():
    df = pd.DataFrame({"Whole_Antibody_Charge_at_Buffer_pH": [5.0]})
    out = normalize_charge_columns(df)
    _out, cols = featurize_charge(out)
    assert cols == CHARGE_FEATURE_COLS
