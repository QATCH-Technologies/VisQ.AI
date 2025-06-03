# transformer_pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class NumericEncoder:
    """
    Maps each categorical column to integer codes (label‐encoding).
    - fit_transform(X) discovers unique values in each categorical column and assigns integers.
    - transform(X) replaces known categories with their integer codes; unknowns become -1.
    """

    def __init__(self):
        self.category_mapping: dict[str, dict[str, int]] = {}
        self.categorical_columns: list[str] = []

    def fit(self, X: pd.DataFrame):
        """
        Identify categorical columns (dtype 'object' or 'category') and build a per‐column mapping.
        """
        # Find columns whose dtype is 'object' or 'category'
        self.categorical_columns = [
            col for col in X.columns
            if X[col].dtype == "object" or X[col].dtype.name == "category"
        ]

        for col in self.categorical_columns:
            # Convert all values to string, then get unique values in order of appearance
            unique_vals = X[col].astype(str).unique().tolist()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            self.category_mapping[col] = mapping

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit on X to learn category→integer mappings, then transform X.
        """
        X_df = X.copy()
        self.fit(X_df)
        return self.transform(X_df)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace each categorical column’s values with their integer codes.
        Unseen categories become -1.
        """
        X_df = X.copy()
        for col in self.categorical_columns:
            if col not in X_df.columns:
                continue
            mapping = self.category_mapping.get(col, {})
            X_df[col] = (
                X_df[col]
                .astype(str)
                .map(lambda v: mapping.get(v, -1))
            )
        return X_df


class ScalerPipeline:
    """
    A thin wrapper around a sklearn Pipeline that implements:
      - fit_transform(X, y) → returns a scaled DataFrame
      - transform(X)        → returns a scaled DataFrame using the fitted scaler
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
        ])

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit on X and return X_scaled as a DataFrame with the same columns/index.
        """
        if not hasattr(X, "shape"):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        X_scaled_arr = self.pipeline.fit_transform(X_df)
        X_scaled = pd.DataFrame(
            X_scaled_arr,
            columns=X_df.columns,
            index=X_df.index
        )
        return X_scaled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X using the already‐fitted scaler.
        """
        if not hasattr(X, "shape"):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        X_scaled_arr = self.pipeline.transform(X_df)
        X_scaled = pd.DataFrame(
            X_scaled_arr,
            columns=X_df.columns,
            index=X_df.index
        )
        return X_scaled


class TransformerPipeline:
    """
    Composite transformer that:
      1. Label‐encodes all categorical columns via NumericEncoder.
      2. Scales all (now‐numeric) columns via StandardScaler.

    - fit_transform(X, y) → label‐encode then scale, returns DataFrame.
    - transform(X)        → label‐encode (with existing mapping) then scale.
    """

    def __init__(self):
        self.encoder = NumericEncoder()
        self.scaler = ScalerPipeline()

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        1) Label‐encode categorical columns
        2) Standard‐scale everything
        Returns a DataFrame with same columns/index.
        """
        # Step 1: encode categories → integers
        X_encoded = self.encoder.fit_transform(X)
        # Step 2: scale numeric columns
        X_scaled = self.scaler.fit_transform(X_encoded)
        return X_scaled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        1) Label‐encode categorical columns (using existing mappings)
        2) Standard‐scale everything (using fitted scaler)
        Returns a DataFrame with same columns/index.
        """
        X_encoded = self.encoder.transform(X)
        X_scaled = self.scaler.transform(X_encoded)
        return X_scaled
