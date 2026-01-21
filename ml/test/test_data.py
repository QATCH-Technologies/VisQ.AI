import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from visq_ml.data import DataProcessor

# --- Fixtures ---


@pytest.fixture
def sample_formulation_data():
    """
    Creates a realistic dataframe matching the schema in config.py.
    Includes edge cases for Regime computation and Concentration splitting.
    """
    data = {
        # Identity
        "Protein_type": ["mab1", "mab2", "mab3", "none", "mab1"],
        "Protein_class_type": ["mab_igg1", "mab_igg4", "fc-fusion", "none", "mab_igg1"],
        # Categoricals
        "Buffer_type": ["histidine", "acetate", "phosphate", "water", "histidine"],
        "Salt_type": ["nacl", "nacl", "none", "none", "nacl"],
        "Stabilizer_type": ["sucrose", "trehalose", "none", "none", "sucrose"],
        "Surfactant_type": ["tween 20", "polysorbate 80", "none", "none", "tween 20"],
        "Excipient_type": ["arginine", "none", "lysine", "none", "arginine"],
        # Numerics needed for Regime (CCI Calculation)
        # CCI = C_Class * exp(-|pH - pI| / 1.5)
        "C_Class": [1.0, 1.0, 0.2, 0.0, 0.6],
        "Buffer_pH": [6.0, 5.5, 7.0, 7.0, 5.0],
        "PI_mean": [6.0, 8.5, 7.0, 0.0, 9.0],  # Case 1: delta=0, Case 2: delta=3, etc.
        "Protein_conc": [
            150.0,
            50.0,
            10.0,
            0.0,
            100.0,
        ],  # 0.0 conc triggers 'noprotein'
        # Numerics for Splitting
        "Salt_conc": [100.0, 200.0, 0.0, 0.0, 150.0],  # Threshold for NaCl is 150.0
        "Stabilizer_conc": [0.0, 250.0, 0.0, 0.0, 0.0],
        # Other numerics
        "MW": [145000, 146000, 75000, 0, 145000],
        "Temperature": [25, 25, 25, 25, 25],
        "kP": [-2.5, -1.0, 0.5, 0.0, -2.0],
        "HCI": [0.5, 1.4, 0.2, 0.0, 0.8],
    }
    return pd.DataFrame(data)


@pytest.fixture
def processor():
    return DataProcessor(allow_new_categories=False)


# --- Tests ---


def test_initialization(processor):
    """Test proper initialization of feature lists from config."""
    assert not processor.is_fitted
    assert "Protein_type" in processor.categorical_features
    assert "Protein_conc" in processor.numeric_features
    # Check that Regime is added dynamically later or checked here if expected
    assert isinstance(processor.cat_maps, dict)


def test_regime_computation_logic(processor, sample_formulation_data):
    """
    Validates the physics-informed Regime computation logic.
    Formula: CCI = C_Class * exp(-|pH - pI| / 1.5)
    """
    df = sample_formulation_data.copy()

    # --- Manual Calculations for Verification ---
    # Row 0: IgG1, C=1.0, pH=6, pI=6 -> delta=0 -> CCI = 1.0 * 1 = 1.0
    # IgG1 Rule: CCI >= 0.9 -> 'near'

    # Row 3: Protein_conc = 0.0 -> 'noprotein'

    regimes = processor._compute_regime(df)

    assert (
        regimes.iloc[0] == "near"
    ), f"Expected 'near' for IgG1 with max CCI, got {regimes.iloc[0]}"
    assert (
        regimes.iloc[3] == "noprotein"
    ), f"Expected 'noprotein' for 0 concentration, got {regimes.iloc[3]}"

    # Test Mixed/Far logic
    # Create specific test case: IgG1, CCI = 0.6 (Between 0.5 and 0.9 -> mixed)
    df_test = pd.DataFrame(
        {
            "Protein_class_type": ["mab_igg1"],
            "C_Class": [0.6],
            "Buffer_pH": [7.0],
            "PI_mean": [7.0],  # delta 0 -> CCI 0.6
            "Protein_conc": [10.0],
        }
    )
    r_test = processor._compute_regime(df_test)
    assert r_test.iloc[0] == "mixed"


def test_concentration_splitting(processor, sample_formulation_data):
    """
    Verifies that concentration columns are correctly split into _low and _high
    features based on the thresholds in config.py.
    """
    # Fit the processor to populate generated_feature_names and split_indices
    processor.fit_transform(sample_formulation_data)

    X_raw, gen_names = processor._construct_numeric_matrix(
        sample_formulation_data, is_fitting=False
    )

    # Identify indices for Salt_conc
    # config.py: Salt_conc pairs with Salt_type (nacl threshold = 150.0)
    low_idx = gen_names.index("Salt_conc_low")
    high_idx = gen_names.index("Salt_conc_high")

    salt_concs = sample_formulation_data["Salt_conc"].values
    low_vals = X_raw[:, low_idx]
    high_vals = X_raw[:, high_idx]

    # Row 0: 100.0 (Below 150) -> Low=100, High=0
    assert low_vals[0] == 100.0
    assert high_vals[0] == 0.0

    # Row 1: 200.0 (Above 150) -> Low=150, High=50
    assert low_vals[1] == 150.0
    assert high_vals[1] == 50.0

    # Row 4: 150.0 (Exact) -> Low=150, High=0
    assert low_vals[4] == 150.0
    assert high_vals[4] == 0.0


def test_fit_transform_integrity(processor, sample_formulation_data):
    """Test full pipeline execution: Categorical encoding + Numeric Scaling."""
    X_num, X_cat = processor.fit_transform(sample_formulation_data)

    # Check Dimensions
    # Numeric: Base numerics are split.
    # Check if generated_feature_names matches output shape
    assert X_num.shape[1] == len(processor.generated_feature_names)
    assert X_num.shape[0] == len(sample_formulation_data)

    # Categorical: Should have same count as categorical_features list
    assert X_cat.shape[1] == len(processor.categorical_features)

    # Check "Regime" was added to categoricals automatically
    assert "Regime" in processor.categorical_features
    assert "Regime" in processor.cat_maps


def test_transform_unseen_categories(processor, sample_formulation_data):
    """Test behavior when encountering new categories with allowed_new=False."""
    processor.fit_transform(sample_formulation_data)

    # Create new data with an unseen protein type
    new_data = sample_formulation_data.iloc[[0]].copy()
    new_data["Protein_type"] = "unseen_super_mab"

    X_num, X_cat = processor.transform(new_data)

    # Find index of Protein_type
    p_idx = processor.categorical_features.index("Protein_type")

    # Should map to 0 (which is 'none'/'unknown' in this implementation usually)
    # The code inserts 'none' at index 0 in _fit_categorical
    assert X_cat[0, p_idx] == 0


def test_dynamic_category_expansion():
    """Test expanding vocabulary when allow_new_categories=True."""
    proc = DataProcessor(allow_new_categories=True)
    df = pd.DataFrame({"Category": ["A", "B"], "Val": [1, 2]})
    # Mock configs for this isolated test
    proc.categorical_features = ["Category"]
    proc.numeric_features = ["Val"]

    proc.fit_transform(df)

    # New data
    df_new = pd.DataFrame({"Category": ["C"], "Val": [3]})
    _, X_cat = proc.transform(df_new)

    # Should have added C
    assert "C" in proc.cat_maps["Category"]
    # Index should be 3 (none, A, B, C)
    assert X_cat[0, 0] == 3


def test_save_and_load(processor, sample_formulation_data):
    """Test persistence of the processor state via pickle."""
    processor.fit_transform(sample_formulation_data)

    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, "processor.pkl")
        processor.save(save_path)

        # Load into new instance
        new_processor = DataProcessor()
        new_processor.load(save_path)

        assert new_processor.is_fitted
        assert new_processor.cat_maps.keys() == processor.cat_maps.keys()

        # Verify split indices persisted
        assert new_processor.split_indices == processor.split_indices

        # Verify transformation yields identical results
        X_num_orig, _ = processor.transform(sample_formulation_data)
        X_num_new, _ = new_processor.transform(sample_formulation_data)

        np.testing.assert_array_almost_equal(X_num_orig, X_num_new)


def test_constant_feature_detection(processor):
    """Ensure features with zero variance are detected and stored."""
    df = pd.DataFrame(
        {"Var": [1.0, 2.0, 3.0], "Const": [5.0, 5.0, 5.0], "Cat": ["a", "b", "c"]}
    )

    # Override defaults for this specific test
    processor.numeric_features = ["Var", "Const"]
    processor.categorical_features = ["Cat"]

    processor.fit_transform(df)

    assert "Const" in processor.constant_features
    assert processor.constant_values["Const"] == 5.0

    # Ensure transform respects this
    X_num, _ = processor.transform(df)
    # The processor currently keeps constant features in the matrix, just scales them.
    # Standard scaler on constant value results in 0.0 usually.

    col_idx = processor.generated_feature_names.index("Const")
    assert np.all(X_num[:, col_idx] == 0.0)
