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
    assert out.loc[0, "buf_mw_ratio"] == 1.0  # histidine is BUFFER_MW_REF itself
    assert "buf_pKa" in prop_cols


def test_unknown_category_falls_back_to_none_zero_row(capsys):
    df = pd.DataFrame({"Buffer_type": ["totally-made-up-buffer"]})
    out, _ = featurize_chemical_categoricals(df)
    assert out.loc[0, "buf_pKa"] == 0.0
    assert out.loc[0, "buf_mw_ratio"] == 0.0
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
    assert (out["stab_mw_ratio"] == 0.0).all()


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


def test_salt_props_carries_no_raw_molecular_weight():
    """P0 regression guard: salt_mw used to be a raw Da value (58-258) that
    went near-zero-variance under a leave-one-salt-out fold and blew up the
    fold's StandardScaler. It was dropped entirely -- valence + Hofmeister
    already carry the salt physics -- so no salt_* column should look like a
    raw molecular weight ever again."""
    from visqai.features.categorical import SALT_PROPS

    for row in SALT_PROPS.values():
        assert "salt_mw" not in row
        assert not any(k.endswith("_mw") for k in row)


def test_other_mw_properties_are_dimensionless_ratios_not_raw_daltons():
    """buf_mw / stab_mw / surf_mw / exc_mw used to carry raw Da values
    (60-8400) straight into the numeric pipeline. They're now expressed as a
    ratio against a fixed reference MW, so every non-'none' category should
    land within a small, bounded multiple of 1.0 -- not the old three-digit-
    to-four-digit raw range."""
    df = pd.DataFrame(
        {
            "Buffer_type": ["citrate"],
            "Stabilizer_type": ["sorbitol"],
            "Surfactant_type": ["poloxamer-188"],
            "Excipient_type": ["glycine"],
        }
    )
    out, _ = featurize_chemical_categoricals(df)
    assert out.loc[0, "buf_mw_ratio"] == pytest.approx(192.12 / 155.2)
    assert out.loc[0, "stab_mw_ratio"] == pytest.approx(182.17 / 342.3)
    assert out.loc[0, "surf_mw_ratio"] == pytest.approx(8400.0 / 1228.0)
    assert out.loc[0, "exc_mw_ratio"] == pytest.approx(75.07 / 174.2)
    # Bounded to a small multiple of 1.0, unlike the old raw-Dalton values.
    for col in ("buf_mw_ratio", "stab_mw_ratio", "surf_mw_ratio", "exc_mw_ratio"):
        assert 0.0 <= out.loc[0, col] < 10.0
