"""Data processing module for VisQ.AI.

This module handles the transformation of raw formulation data into model-ready
numeric and categorical features. It includes specialized logic for:
    1.  **Regime Computation**: Deriving electrostatic interaction regimes from
        pH, pI, and charge class.
    2.  **Concentration Splitting**: Splitting single concentration features into
        low/high components based on saturation thresholds.
    3.  **Encoding**: Standardizing categorical vocabulary and handling unknown values.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-01-15

Version:
    1.0
"""

import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .config import (
        BASE_CATEGORICAL,
        BASE_NUMERIC,
        CONC_THRESHOLDS,
        CONC_TYPE_PAIRS,
    )
except (ImportError, ModuleNotFoundError):
    try:
        # Fallback for Predictor dynamic loading (sys.modules['config'])
        from config import (
            BASE_CATEGORICAL,
            BASE_NUMERIC,
            CONC_THRESHOLDS,
            CONC_TYPE_PAIRS,
        )
    except (ImportError, ModuleNotFoundError):
        # Fallback for installed package
        from visq_ml.config import (
            BASE_CATEGORICAL,
            BASE_NUMERIC,
            CONC_THRESHOLDS,
            CONC_TYPE_PAIRS,
        )


class DataProcessor:
    """Handles data preprocessing for viscosity modeling.

    This class wraps categorical encoding, numeric scaling, and physics-informed
    feature engineering into a single pipeline. It supports saving/loading state
    to ensure consistent processing between training and inference.

    Attributes:
        cat_maps (Dict[str, List[str]]): Dictionaries mapping categorical feature
            names to their allowed vocabulary lists.
        scaler (StandardScaler): The fitted scikit-learn scaler for numeric features.
        is_fitted (bool): True if the processor has been fitted to data.
        categorical_features (List[str]): List of categorical column names to process.
        numeric_features (List[str]): List of numeric column names to process.
        allow_new_categories (bool): If True, unseen categories during transform
            are added to the vocabulary. If False, they are mapped to index 0 ('none').
        constant_features (List[str]): Numeric features detected as having zero variance.
        constant_values (Dict[str, float]): The single value associated with constant features.
        split_indices (Dict[str, Tuple[int, int]]): Metadata tracking how original
            concentration columns map to indices in the generated feature matrix.
        generated_feature_names (List[str]): The final list of column names after
            feature engineering (splitting).
    """

    TAU = 1.5

    def __init__(self, allow_new_categories: bool = False):
        """Initializes the DataProcessor.

        Args:
            allow_new_categories (bool, optional): If True, dynamic vocabulary expansion
                is allowed during transformation. Defaults to False.
        """
        self.cat_maps: Dict[str, List[str]] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.categorical_features = BASE_CATEGORICAL.copy()
        self.numeric_features = BASE_NUMERIC.copy()
        self.allow_new_categories = allow_new_categories
        self.constant_features: List[str] = []
        self.constant_values: Dict[str, float] = {}
        self.scalable_features: List[str] = []
        self.split_indices: Dict[str, Tuple[int, int]] = {}
        self.generated_feature_names: List[str] = []

    def _compute_regime(self, df: pd.DataFrame) -> pd.Series:
        """Computes the electrostatic interaction regime.

        Calculates the Charge-Charge Interaction (CCI) score and assigns a regime label
        ('near', 'mixed', 'far', 'noprotein') based on protein class and score thresholds.

        The CCI score is calculated as:
            $$ CCI = C_{Class} \cdot e^{-|pH - pI| / \tau} $$
        where $\tau = 1.5$.

        Args:
            df (pd.DataFrame): Input dataframe containing 'C_Class', 'Buffer_pH',
                'PI_mean', and 'Protein_class_type'.

        Returns:
            pd.Series: A series of regime strings (e.g., 'near', 'noprotein').
        """
        regime = []
        has_p_conc = "Protein_conc" in df.columns
        has_p_type = "Protein_type" in df.columns
        req_cols = ["C_Class", "Buffer_pH", "PI_mean", "Protein_class_type"]
        missing = [c for c in req_cols if c not in df.columns]

        if "Protein_class" in df.columns and "Protein_class_type" in missing:
            df["Protein_class_type"] = df["Protein_class"]
            missing.remove("Protein_class_type")

        # CCI Score Calculation
        if not missing:
            tau = self.TAU
            delta_ph = (df["Buffer_pH"] - df["PI_mean"]).abs()
            df["CCI_Score"] = df["C_Class"] * np.exp(-delta_ph / tau)
            cci_values = df["CCI_Score"]
        else:
            df["CCI_Score"] = np.nan
            cci_values = pd.Series([np.nan] * len(df), index=df.index)

        for i in range(len(df)):
            row = df.iloc[i]
            is_no_protein = False
            if has_p_type:
                p_type = str(row.get("Protein_type", "")).lower()
                if p_type in ["none", "buffer", "nan", "null"]:
                    is_no_protein = True
            if has_p_conc:
                if row.get("Protein_conc", 0.0) <= 0.0:
                    is_no_protein = True

            if is_no_protein:
                regime.append("noprotein")
                continue

            if missing:
                regime.append("unknown")
                continue

            cci = cci_values.iloc[i]
            p_class = str(df["Protein_class_type"].iloc[i]).lower()

            if pd.isna(cci):
                regime.append("unknown")
                continue

            r = "unknown"
            # Logic from Table B (Standardized Regime Rules)
            if "igg1" in p_class:
                if cci >= 0.90:
                    r = "near"
                elif cci >= 0.50:
                    r = "mixed"
                else:
                    r = "far"
            elif "igg4" in p_class:
                if cci >= 0.80:
                    r = "near"
                elif cci >= 0.40:
                    r = "mixed"
                else:
                    r = "far"
            elif (
                "fc-fusion" in p_class
                or "linker" in p_class
                or "trispecific" in p_class
            ):
                if cci >= 0.70:
                    r = "near"
                elif cci >= 0.40:
                    r = "mixed"
                else:
                    r = "far"
            elif "bispecific" in p_class or "adc" in p_class:
                if cci >= 0.80:
                    r = "near"
                elif cci >= 0.45:
                    r = "mixed"
                else:
                    r = "far"
            elif "bsa" in p_class or "polyclonal" in p_class:
                if cci >= 0.70:
                    r = "near"
                elif cci >= 0.40:
                    r = "mixed"
                else:
                    r = "far"
            else:
                if cci >= 0.90:
                    r = "near"
                elif cci >= 0.50:
                    r = "mixed"
                else:
                    r = "far"

            regime.append(r)

        return pd.Series(regime, index=df.index)

    def _construct_numeric_matrix(
        self, df: pd.DataFrame, is_fitting: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        """Builds the numeric feature matrix with concentration splitting.

        Splits specified concentration columns into 'low' and 'high' features based on
        defined thresholds. This allows the model to learn piecewise linear responses
        to concentrations (e.g., different behaviors below vs above saturation).

        For a concentration $C$ and threshold $T$:
        * $E_{low} = \min(C, T)$
        * $E_{high} = \max(C - T, 0)$

        Args:
            df (pd.DataFrame): The input dataframe.
            is_fitting (bool, optional): Whether this is called during the `fit` phase.
                If True, it records split indices and feature names. Defaults to False.

        Returns:
            Tuple[np.ndarray, List[str]]:
                - A numpy array of shape (n_samples, n_features).
                - A list of generated feature names matching the array columns.
        """
        feature_arrays = []
        gen_names = []
        new_split_indices = {}
        current_idx = 0

        for col in self.numeric_features:
            # Check if this column requires splitting (defined in CONC_TYPE_PAIRS)
            if col in CONC_TYPE_PAIRS:
                type_col = CONC_TYPE_PAIRS[col]
                if col in df.columns:
                    conc_vals = df[col].fillna(0.0).values
                else:
                    conc_vals = np.zeros(len(df))

                # Get type strings for threshold lookup
                if type_col in df.columns:
                    type_vals = (
                        df[type_col].fillna("none").astype(str).str.lower().values
                    )
                else:
                    type_vals = np.full(len(df), "none")

                # Default threshold is arbitrarily high so E_low = conc, E_high = 0 if type unknown
                thresh_vals = np.full(len(df), 999999.0)
                for k, t_val in CONC_THRESHOLDS.items():
                    mask = np.array([k in str(v) for v in type_vals])
                    thresh_vals[mask] = t_val

                # E_low = min(E, T)
                E_low = np.minimum(np.asarray(conc_vals, dtype=np.float64), thresh_vals)
                # E_high = max(E - T, 0)
                E_high = np.maximum(conc_vals - thresh_vals, 0.0)
                feature_arrays.append(E_low)
                feature_arrays.append(E_high)
                gen_names.append(f"{col}_low")
                gen_names.append(f"{col}_high")
                new_split_indices[col] = (current_idx, current_idx + 1)
                current_idx += 2

            else:
                # Standard Processing
                if col in self.constant_features:
                    vals = np.full(len(df), self.constant_values[col])
                elif col in df.columns:
                    vals = df[col].fillna(0.0).values
                else:
                    vals = np.zeros(len(df))
                feature_arrays.append(vals)
                gen_names.append(col)
                current_idx += 1

        # Stack all arrays horizontally
        X_combined = np.column_stack(feature_arrays)

        if is_fitting:
            self.generated_feature_names = gen_names
            self.split_indices = new_split_indices

        return X_combined, gen_names

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fits the processor to the data and returns transformed arrays.

        1. Builds categorical vocabularies (including 'Regime').
        2. Identifies constant numeric features.
        3. Constructs the expanded numeric matrix (splitting concentrations).
        4. Fits the StandardScaler.

        Args:
            df (pd.DataFrame): Training data.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X_num: Scaled numeric features (n_samples, n_numeric).
                - X_cat: Integer-encoded categorical features (n_samples, n_categorical).
        """
        # Fit Categorical
        self._fit_categorical(df)

        # Fit Numeric (with Splitting)
        for col in self.numeric_features:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 1:
                    self.constant_features.append(col)
                    self.constant_values[col] = unique_vals[0]

        X_num_full, _ = self._construct_numeric_matrix(df, is_fitting=True)
        self.scaler.fit(X_num_full)
        self.is_fitted = True
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms data using the previously fitted processor.

        Args:
            df (pd.DataFrame): Input data to transform.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X_num: Scaled numeric features.
                - X_cat: Integer-encoded categorical features.

        Raises:
            ValueError: If the processor has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted first")

        X_cat = self._transform_categorical(df)

        # Construct numeric matrix (using fitted split logic)
        X_num_raw, _ = self._construct_numeric_matrix(df, is_fitting=False)

        # Apply scaling
        X_num = self.scaler.transform(X_num_raw)

        return X_num, X_cat

    def _fit_categorical(self, df: pd.DataFrame) -> None:
        """Builds categorical vocabularies from the dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing categorical columns.
        """
        if "Regime" not in self.categorical_features:
            self.categorical_features.append("Regime")

        for col in self.categorical_features:
            if col == "Regime":
                # Preset Regime vocabulary including new 'noprotein'
                self.cat_maps["Regime"] = [
                    "far",
                    "mixed",
                    "near",
                    "noprotein",
                    "unknown",
                ]
                continue

            if col not in df.columns:
                print(f"Warning: {col} not in dataframe, skipping")
                self.cat_maps[col] = ["none"]
                continue

            unique_vals = df[col].dropna().unique().tolist()
            if "none" not in unique_vals:
                unique_vals.insert(0, "none")

            self.cat_maps[col] = unique_vals

    def _transform_categorical(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms categorical features to integer indices.

        Args:
            df (pd.DataFrame): Dataframe to encode.

        Returns:
            np.ndarray: Transposed array of integer indices (n_samples, n_cat_features).
        """
        cat_data = []

        # Pre-compute regimes to handle 'noprotein' logic
        regime_series = self._compute_regime(df)

        for col in self.categorical_features:
            if col == "Regime":
                indices = [
                    (
                        self.cat_maps["Regime"].index(val)
                        if val in self.cat_maps["Regime"]
                        else self.cat_maps["Regime"].index("unknown")
                    )
                    for val in regime_series
                ]
            else:
                if col not in df.columns:
                    indices = [0] * len(df)
                else:
                    values = df[col].fillna("none").astype(str).str.lower()
                    indices = []
                    for val in values:
                        if val in self.cat_maps[col]:
                            indices.append(self.cat_maps[col].index(val))
                        else:
                            if self.allow_new_categories:
                                self.cat_maps[col].append(val)
                                indices.append(len(self.cat_maps[col]) - 1)
                            else:
                                indices.append(0)  # 'none'

            cat_data.append(indices)

        return np.array(cat_data, dtype=np.int64).T

    def compute_distance(self, X_num: np.ndarray) -> np.ndarray:
        """Computes Euclidean distance of samples from the origin in scaled space.

        Can be used as a heuristic for "trust scores" (distance from training mean).

        Args:
            X_num (np.ndarray): Scaled numeric feature matrix.

        Returns:
            np.ndarray: Array of distances for each sample.
        """
        if not hasattr(self.scaler, "mean_") or self.scaler.mean_ is None:
            return np.zeros(X_num.shape[0])
        distances = np.linalg.norm(X_num, axis=1)
        return distances

    def detect_new_categories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identifies categorical values in `df` that are not in the fitted vocabulary.

        Args:
            df (pd.DataFrame): New data to check.

        Returns:
            Dict[str, List[str]]: A dictionary mapping column names to lists of new values.
        """
        new_cats = {}
        for col in self.categorical_features:
            if col == "Regime" or col not in df.columns:
                continue

            df_vals = set(df[col].dropna().astype(str).str.lower().unique())
            known_vals = set(self.cat_maps[col])
            unseen = df_vals - known_vals

            if unseen:
                new_cats[col] = list(unseen)
        return new_cats

    def add_categories(self, feature_name: str, new_categories: List[str]) -> None:
        """Manually adds new categories to the vocabulary of a feature.

        Args:
            feature_name (str): The name of the categorical feature.
            new_categories (List[str]): List of new category strings to add.

        Raises:
            ValueError: If `feature_name` is not a known categorical feature.
        """
        if feature_name not in self.cat_maps:
            raise ValueError(f"Feature {feature_name} not found")

        for cat in new_categories:
            if cat not in self.cat_maps[feature_name]:
                self.cat_maps[feature_name].append(cat)
        print(f"Added {len(new_categories)} categories to {feature_name}")

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Returns the size of the vocabulary for each categorical feature.

        Returns:
            Dict[str, int]: Dictionary mapping feature names to vocabulary counts.

        Raises:
            ValueError: If the processor has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted first")
        return {col: len(self.cat_maps[col]) for col in self.categorical_features}

    def save(self, path: str) -> None:
        """Saves the processor state to a pickle file.

        Args:
            path (str): The file path to save the pickle object.
        """
        state = {
            "cat_maps": self.cat_maps,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "allow_new_categories": self.allow_new_categories,
            "constant_features": self.constant_features,
            "constant_values": self.constant_values,
            "scalable_features": self.scalable_features,
            "split_indices": self.split_indices,
            "generated_feature_names": self.generated_feature_names,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Loads the processor state from a pickle file.

        Args:
            path (str): The file path to load the pickle object from.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.cat_maps = state["cat_maps"]
        self.scaler = state["scaler"]
        self.is_fitted = state["is_fitted"]
        self.categorical_features = state.get(
            "categorical_features", BASE_CATEGORICAL.copy()
        )
        self.numeric_features = state.get("numeric_features", BASE_NUMERIC.copy())
        self.allow_new_categories = state.get("allow_new_categories", False)
        self.constant_features = state.get("constant_features", [])
        self.constant_values = state.get("constant_values", {})
        self.scalable_features = state.get(
            "scalable_features", self.numeric_features.copy()
        )

        # Load split metadata
        self.split_indices = state.get("split_indices", {})
        self.generated_feature_names = state.get("generated_feature_names", [])


class AnalogSelector:
    """Helper to find the best physicochemical match for a new protein."""

    def __init__(self, reference_df: pd.DataFrame):
        # Create unique library of known proteins
        self.library = reference_df.drop_duplicates(subset=["Protein_type"]).copy()

        # Weights for distance calculation (Adjust based on importance)
        self.weights = {
            "PI_mean": 2.0,  # Charge is dominant
            "kP": 1.5,  # Interaction proxy
            "MW": 1.0,  # Size proxy
            "diff_coeff": 1.0,
        }

    def find_best_analog(self, new_protein_row: pd.Series, known_classes: list) -> str:
        target_class = new_protein_row.get("Protein_class_type", "Unknown")

        # Tier 1: Class Filter
        if target_class in self.library["Protein_class_type"].values:
            candidates = self.library[
                self.library["Protein_class_type"] == target_class
            ].copy()
        else:
            # Fallback for completely new classes
            fallback_map = {
                "Bispecific": ["mAb", "Monoclonal Antibody"],
                "Fusion_Protein": ["Complex_Protein", "Other"],
            }
            fallback_classes = fallback_map.get(target_class, [])
            candidates = self.library[
                self.library["Protein_class_type"].isin(fallback_classes)
            ].copy()
            if candidates.empty:
                candidates = self.library.copy()

        # Tier 2: Weighted Euclidean Distance
        candidates["score"] = 0.0
        valid_cols = 0

        for col, weight in self.weights.items():
            if col in new_protein_row and col in candidates.columns:
                target_val = float(new_protein_row[col])
                col_values = candidates[col].astype(float)
                col_std = col_values.std() + 1e-6

                diff = ((col_values - target_val) / col_std) ** 2
                candidates["score"] += diff * weight
                valid_cols += 1

        if valid_cols == 0:
            return candidates.iloc[0]["Protein_type"]

        best_match = candidates.nsmallest(1, "score").iloc[0]
        return best_match["Protein_type"]
