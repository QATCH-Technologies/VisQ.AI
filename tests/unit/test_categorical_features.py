import pandas as pd
import pytest

from visqai.features.categorical import (
    CHEM_CATEGORICALS,
    featurize_chemical_categoricals,
    describe_property_space,
)


def test_known_category_maps_to_its_property_row():
    df = pd.DataFrame({"Buffer_type": ["histidine"]})
    out, prop_cols = featurize_chemical_categoricals(df)
    assert out.loc[0, "buf_pKa"] == 6.0
    assert out.loc[0, "buf_mw"] == 155.2
    assert "buf_pKa" in prop_cols


def test_unknown_category_falls_back_to_none_zero_row(capsys):
    df = pd.DataFrame({"Buffer_type": ["totally-made-up-buffer"]})
    out, _ = featurize_chemical_categoricals(df)
    assert out.loc[0, "buf_pKa"] == 0.0
    assert out.loc[0, "buf_mw"] == 0.0
    captured = capsys.readouterr()
    assert "unknown category" in captured.out


def test_missing_column_treated_as_all_none():
    df = pd.DataFrame({"Salt_type": ["nacl"]})  # Buffer_type absent entirely
    out, _ = featurize_chemical_categoricals(df)
    assert out.loc[0, "buf_pKa"] == 0.0
    assert out.loc[0, "salt_hofmeister"] == 0.0


def test_substring_match_resolves_naming_variants():
    df = pd.DataFrame({"Excipient_type": ["l-arginine hcl"]})
    out, _ = featurize_chemical_categoricals(df)
    assert out.loc[0, "exc_charge"] == 1.0  # matched "arginine"


def test_nan_like_values_normalize_to_none():
    df = pd.DataFrame({"Stabilizer_type": ["", "nan", "unknown", None]})
    out, _ = featurize_chemical_categoricals(df)
    assert (out["stab_mw"] == 0.0).all()


def test_drop_original_removes_source_column():
    df = pd.DataFrame({"Salt_type": ["kcl"]})
    out, _ = featurize_chemical_categoricals(df, drop_original=True)
    assert "Salt_type" not in out.columns


def test_all_chem_categoricals_covered_in_output():
    df = pd.DataFrame({c: ["none"] for c in CHEM_CATEGORICALS})
    out, prop_cols = featurize_chemical_categoricals(df)
    for col in prop_cols:
        assert col in out.columns


def test_describe_property_space_has_a_row_per_category():
    tidy = describe_property_space()
    assert set(tidy["categorical"].unique()) == set(CHEM_CATEGORICALS)
    assert "none" in tidy["category"].values
