# Ensure local imports work
import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.append(os.getcwd())

from src.utils import (
    HIGH_VIS_MULTIPLIER,
    HIGH_VIS_THRESHOLD,
    calculate_sample_weights,
    clean,
    inverse_log_transform,
    log_transform_targets,
    to_tensors,
    validate_data,
)


class TestUtils(unittest.TestCase):

    def test_clean_string_standardization(self):
        """Test that strings are lowercased and bad values become NaN."""
        df = pd.DataFrame(
            {"Cat": ["  IgG1 ", "NONE", "null", "Valid"], "Num": [1, 2, 3, 4]}
        )

        df_clean = clean(df, numeric_cols=["Num"], categorical_cols=["Cat"])

        # "  IgG1 " -> "igg1"
        self.assertEqual(df_clean["Cat"].iloc[0], "igg1")
        # "NONE" -> NaN -> "none" (imputed later in function)
        self.assertEqual(df_clean["Cat"].iloc[1], "none")
        self.assertEqual(df_clean["Cat"].iloc[2], "none")

    def test_clean_imputation_logic(self):
        """
        Test distinct imputation rules:
        - Columns with 'conc' in name -> fill with 0.0
        - Other numeric columns -> fill with Median
        """
        df = pd.DataFrame(
            {
                "Protein_conc": [10.0, np.nan, 30.0],  # Should fill 0.0
                "Temperature": [20.0, np.nan, 30.0],  # Should fill Median (25.0)
                "Type": ["A", "B", "C"],
            }
        )

        numeric = ["Protein_conc", "Temperature"]
        df_clean = clean(df, numeric_cols=numeric, categorical_cols=["Type"])

        # Concentration Check
        self.assertEqual(df_clean["Protein_conc"].iloc[1], 0.0)

        # Physical Property Check
        # Median of [20, 30] is 25.
        self.assertEqual(df_clean["Temperature"].iloc[1], 25.0)

    def test_clean_target_filtering(self):
        """Test that rows with missing or non-positive targets are dropped."""
        df = pd.DataFrame({"Conc": [1, 2, 3, 4], "Viscosity": [10.0, np.nan, 0.0, 5.0]})

        # Row 0: Valid
        # Row 1: NaN Target -> Drop
        # Row 2: 0.0 Target -> Drop
        # Row 3: Valid

        df_clean = clean(
            df, numeric_cols=["Conc"], categorical_cols=[], target_cols=["Viscosity"]
        )

        self.assertEqual(len(df_clean), 2)
        self.assertEqual(df_clean["Viscosity"].iloc[0], 10.0)
        self.assertEqual(df_clean["Viscosity"].iloc[1], 5.0)

    def test_log_transform_reversibility(self):
        """Test that inverse(log(x)) returns x (approx)."""
        original = np.array([1.0, 10.0, 100.0])

        log_y = log_transform_targets(original)
        reconstructed = inverse_log_transform(log_y)

        # log10(1) = 0, log10(10)=1, etc.
        # But code adds +1e-8.
        np.testing.assert_allclose(reconstructed, original, rtol=1e-5)

    def test_tensor_conversion_dtypes(self):
        """Verify numpy arrays convert to correct torch dtypes."""
        X_num = np.zeros((2, 2))
        X_cat = np.zeros((2, 2))
        y = np.zeros((2, 1))

        tensors = to_tensors(X_num, X_cat, y)

        t_num, t_cat, t_y = tensors

        self.assertEqual(t_num.dtype, torch.float32)
        self.assertEqual(t_cat.dtype, torch.long)  # Categorical must be Long/Int64
        self.assertEqual(t_y.dtype, torch.float32)

    def test_validate_data_checks(self):
        """Test exception raising for bad data."""
        # Case 1: NaN in numeric
        with self.assertRaises(ValueError):
            validate_data(np.array([np.nan]), np.zeros(1), np.zeros(1))

        # Case 2: Inf in numeric
        with self.assertRaises(ValueError):
            validate_data(np.array([np.inf]), np.zeros(1), np.zeros(1))

        # Case 3: Clean data passes
        self.assertTrue(validate_data(np.zeros(1), np.zeros(1), np.zeros(1)))

    def test_sample_weights_logic(self):
        """
        Test Weight Calculation:
        1. Base = 1 + log1p(viscosity)
        2. High Viscosity (> Threshold) = Base * Multiplier
        """
        # Threshold is defined in module (e.g. 20.0)
        # Multiplier is defined in module (e.g. 5.0)

        # Sample 1: Viscosity 9.0 (Low). Weight = 1 + log(10) ~ 1 + 2.3 = 3.3
        # Sample 2: Viscosity 100.0 (High). Weight = (1 + log(101)) * 5
        y = np.array([[9.0], [100.0]])

        weights = calculate_sample_weights(y)

        # Manual Calc
        val_low = 9.0
        expected_low = 1.0 + np.log1p(val_low)

        val_high = 100.0
        expected_high = (1.0 + np.log1p(val_high)) * HIGH_VIS_MULTIPLIER

        self.assertAlmostEqual(weights[0], expected_low, places=4)
        self.assertAlmostEqual(weights[1], expected_high, places=4)

        # Verify Threshold Boundary
        # If threshold is 20.0, then 21.0 should trigger multiplier
        y_boundary = np.array([[HIGH_VIS_THRESHOLD + 1.0]])
        w_boundary = calculate_sample_weights(y_boundary)
        self.assertTrue(w_boundary[0] > 10.0)  # Should be boosted


if __name__ == "__main__":
    unittest.main()
