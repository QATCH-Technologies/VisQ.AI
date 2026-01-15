import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Ensure local modules can be imported if running as a script
sys.path.append(os.getcwd())

from src.config import BASE_CATEGORICAL, CONC_THRESHOLDS
from src.data import DataProcessor


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Set up a fresh processor and a comprehensive dummy dataset."""
        self.processor = DataProcessor(allow_new_categories=False)

        # A rich dataset covering various protein types and excipients
        # to trigger specific logic paths in config.py and data.py
        self.df_train = pd.DataFrame(
            {
                "Protein_type": ["IgG1_mAb", "IgG4_mAb", "Bispecific_A", "Buffer_Only"],
                "Protein_class_type": ["mAb_IgG1", "mAb_IgG4", "Bispecific", "None"],
                "Buffer_component": ["Histidine", "Histidine", "Acetate", "Water"],
                # Numeric features (from BASE_NUMERIC)
                "Protein_conc": [50.0, 50.0, 10.0, 0.0],
                "MW": [150, 150, 150, 0],
                "Temperature": [25, 25, 25, 25],
                "Buffer_pH": [6.0, 6.0, 5.0, 7.0],
                "Buffer_conc": [20, 20, 20, 0],
                "PI_mean": [8.5, 8.5, 5.0, 0.0],  # Used for Delta pH
                "C_Class": [1, 1, -1, 0],  # Used for CCI Score
                # Concentration pairs for splitting tests
                # Note: config.py defines 'nacl' threshold as 150.0
                "Salt_conc": [100.0, 200.0, 0.0, 0.0],
                "Salt_type": ["NaCl", "NaCl", "None", "None"],
                # Note: config.py defines 'sucrose' threshold as 200.0
                "Stabilizer_conc": [0.0, 250.0, 50.0, 0.0],
                "Stabilizer_type": ["None", "Sucrose", "Sucrose", "None"],
            }
        )

    def test_initialization_defaults(self):
        """Verify processor loads defaults from the real config.py."""
        # It should load BASE_CATEGORICAL from config.py
        self.assertEqual(self.processor.categorical_features, BASE_CATEGORICAL)
        # It should initialize scaler
        self.assertIsNotNone(self.processor.scaler)
        # Should start unfitted
        self.assertFalse(self.processor.is_fitted)

    def test_regime_logic_igg1_vs_igg4(self):
        """
        Verify the physics logic branches defined in _compute_regime.
        Logic: CCI = C_Class * exp(-|pH - pI| / 1.5)

        We will manually construct cases to hit specific thresholds in data.py
        IgG1 'near' threshold: >= 0.90
        IgG4 'near' threshold: >= 0.80
        """
        # Setup: Delta pH = 0, so exp(0)=1, CCI = C_Class.
        df_physics = pd.DataFrame(
            {
                "C_Class": [0.85, 0.85],
                "Buffer_pH": [7.0, 7.0],
                "PI_mean": [7.0, 7.0],
                "Protein_class_type": ["mAb_IgG1", "mAb_IgG4"],
                "Protein_conc": [10.0, 10.0],
                "Protein_type": ["Mab1", "Mab2"],
            }
        )

        regimes = self.processor._compute_regime(df_physics)

        # IgG1 requires 0.9 for Near. 0.85 is < 0.9 but >= 0.5 -> Mixed
        self.assertEqual(
            regimes.iloc[0],
            "mixed",
            "IgG1 with CCI 0.85 should be 'mixed' (Threshold is 0.9)",
        )

        # IgG4 requires 0.8 for Near. 0.85 is >= 0.8 -> Near
        self.assertEqual(
            regimes.iloc[1],
            "near",
            "IgG4 with CCI 0.85 should be 'near' (Threshold is 0.8)",
        )

    def test_regime_noprotein_detection(self):
        """Test detection of 'noprotein' regime based on concentration or type."""
        df_none = pd.DataFrame(
            {
                "C_Class": [1, 1],
                "Buffer_pH": [7.0, 7.0],
                "PI_mean": [7.0, 7.0],
                "Protein_class_type": ["IgG1", "IgG1"],
                "Protein_conc": [0.0, 10.0],
                "Protein_type": ["IgG1", "Buffer"],
            }
        )

        regimes = self.processor._compute_regime(df_none)

        # Case 1: Protein_conc is 0.0 -> noprotein
        self.assertEqual(regimes.iloc[0], "noprotein")
        # Case 2: Protein_type is 'Buffer' -> noprotein
        self.assertEqual(regimes.iloc[1], "noprotein")

    def test_concentration_splitting_real_thresholds(self):
        """
        Test splitting logic using actual thresholds from config.py.
        Config: NaCl = 150.0
        Formula:
           Low = min(Conc, Threshold)
           High = max(Conc - Threshold, 0)
        """
        df_salt = pd.DataFrame(
            {"Salt_conc": [100.0, 150.0, 200.0], "Salt_type": ["NaCl", "NaCl", "NaCl"]}
        )

        # We limit numeric features to just Salt_conc for this isolation test
        self.processor.numeric_features = ["Salt_conc"]

        matrix, names = self.processor._construct_numeric_matrix(
            df_salt, is_fitting=True
        )

        self.assertIn("Salt_conc_low", names)
        self.assertIn("Salt_conc_high", names)

        # Row 0: 100.0 (Below 150) -> Low: 100, High: 0
        self.assertEqual(matrix[0][0], 100.0)
        self.assertEqual(matrix[0][1], 0.0)

        # Row 1: 150.0 (Equal 150) -> Low: 150, High: 0
        self.assertEqual(matrix[1][0], 150.0)
        self.assertEqual(matrix[1][1], 0.0)

        # Row 2: 200.0 (Above 150) -> Low: 150, High: 50
        self.assertEqual(matrix[2][0], 150.0)
        self.assertEqual(matrix[2][1], 50.0)

    def test_concentration_splitting_unknown_type(self):
        """
        If Salt_type is unknown or 'None', the threshold defaults to extremely high (999999).
        The whole value should go to 'low'.
        """
        df_unknown = pd.DataFrame({"Salt_conc": [500.0], "Salt_type": ["UnknownSalt"]})

        self.processor.numeric_features = ["Salt_conc"]
        matrix, _ = self.processor._construct_numeric_matrix(
            df_unknown, is_fitting=True
        )

        # Threshold is effectively infinite
        self.assertEqual(matrix[0][0], 500.0)  # Low
        self.assertEqual(matrix[0][1], 0.0)  # High

    def test_fit_transform_structure(self):
        """Verify the full pipeline produces expected shapes and scaled output."""
        # Fit on training data
        X_num, X_cat = self.processor.fit_transform(self.df_train)

        self.assertTrue(self.processor.is_fitted)

        # 1. Check Categorical Output
        # Should include 'Regime' plus BASE_CATEGORICAL
        # df_train has Protein_type, Protein_class_type, etc.
        # Regime is auto-calculated.
        self.assertIn("Regime", self.processor.cat_maps)
        self.assertEqual(X_cat.shape[0], len(self.df_train))

        # 2. Check Numeric Output
        # We expect splitting for Salt_conc and Stabilizer_conc (defined in config.CONC_TYPE_PAIRS)
        # Other numeric cols (MW, Temperature, etc) are just passed through (if present).
        # Check if Salt_conc_low and Salt_conc_high exist in generated names
        self.assertIn("Salt_conc_low", self.processor.generated_feature_names)
        self.assertIn("Salt_conc_high", self.processor.generated_feature_names)

        # 3. Check Scaling
        # StandardScaler creates mean ~0
        self.assertTrue(
            np.abs(X_num.mean()) < 1e-5, "Output should be scaled (mean approx 0)"
        )

    def test_transform_new_categories_ignored(self):
        """Test that new categories are mapped to 'none' (0) when allow_new_categories=False."""
        self.processor.fit_transform(self.df_train)

        df_new = self.df_train.copy()
        df_new.iloc[0, 0] = "SuperNewProtein"  # Modify Protein_type

        _, X_cat = self.processor.transform(df_new)

        # Assuming 'none' is at index 0 (as per _fit_categorical logic)
        self.assertEqual(X_cat[0, 0], 0)

    def test_transform_new_categories_allowed(self):
        """Test that new categories are added when allow_new_categories=True."""
        proc = DataProcessor(allow_new_categories=True)
        proc.fit_transform(self.df_train)

        df_new = self.df_train.copy()
        col_idx = proc.categorical_features.index("Protein_type")
        df_new.iloc[0, col_idx] = "SuperNewProtein"

        _, X_cat = proc.transform(df_new)

        # It should NOT be 0 (none), it should be a new index
        new_idx = X_cat[0, col_idx]
        self.assertNotEqual(new_idx, 0)
        self.assertIn("supernewprotein", proc.cat_maps["Protein_type"])

    def test_constant_feature_handling(self):
        """
        Verify that features with single unique value are detected as constant
        but still processed without crashing the scaler.
        """
        df_const = self.df_train.copy()
        # Force Temperature to be constant
        df_const["Temperature"] = 25.0

        self.processor.fit_transform(df_const)

        self.assertIn("Temperature", self.processor.constant_features)
        self.assertEqual(self.processor.constant_values["Temperature"], 25.0)

        # Transform should still work and return 0 (scaled value of constant)
        X_num, _ = self.processor.transform(df_const)

        # Find index of Temperature
        temp_idx = self.processor.generated_feature_names.index("Temperature")
        self.assertTrue(np.all(X_num[:, temp_idx] == 0.0))

    def test_save_load_cycle(self):
        """Verify object persistence."""
        self.processor.fit_transform(self.df_train)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            save_path = tmp.name

        try:
            self.processor.save(save_path)

            loaded_proc = DataProcessor()
            loaded_proc.load(save_path)

            self.assertTrue(loaded_proc.is_fitted)
            self.assertEqual(
                loaded_proc.generated_feature_names,
                self.processor.generated_feature_names,
            )

            # Verify transformation matches
            X_orig, _ = self.processor.transform(self.df_train)
            X_load, _ = loaded_proc.transform(self.df_train)

            np.testing.assert_array_equal(X_orig, X_load)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


if __name__ == "__main__":
    unittest.main()
