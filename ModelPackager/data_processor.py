# data_processor.py

import pandas as pd
from typing import Any, Tuple, Union


class DataProcessor:
    """
    - load(raw_input) → returns a DataFrame of features (drops ID and all Viscosity_* columns).
    - process(raw_X, raw_y=None) → returns (X_df, y_df) when raw_X is the single CSV with both features and targets.
      If raw_y is provided (not None), it is ignored (for backwards compatibility).
    """

    def __init__(self, config: dict = None):
        """
        config can specify:
          - drop_columns: list[str] of any extra columns to drop (beyond ID and Viscosity_*).
          - feature_columns: if you want to explicitly pick a subset of columns for X.
            Otherwise, we infer by dropping ID + Viscosity_* + drop_columns.
          - target_prefix: prefix string to identify target columns. Default = "Viscosity_".
        """
        cfg = config or {}
        self.drop_columns = cfg.get("drop_columns", [])
        self.feature_columns = cfg.get("feature_columns", None)
        self.target_prefix = cfg.get("target_prefix", "Viscosity_")

    def load(self, raw_input: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Given raw_input (path or DataFrame), return X_df = raw features only.
        Drops:
          - "ID"
          - all columns starting with self.target_prefix
          - any columns in self.drop_columns
        """
        if isinstance(raw_input, str):
            df = pd.read_csv(raw_input)
        elif isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        else:
            raise ValueError(f"Unsupported raw_input type: {type(raw_input)}")

        # Drop ID if present
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])

        # Identify and drop any Viscosity_* columns
        viscosity_cols = [
            c for c in df.columns if c.startswith(self.target_prefix)]
        df = df.drop(columns=viscosity_cols, errors="ignore")

        # Drop any extra columns configured
        df = df.drop(columns=self.drop_columns, errors="ignore")

        # If user explicitly listed feature_columns, select those:
        if self.feature_columns:
            missing = [c for c in self.feature_columns if c not in df.columns]
            if missing:
                raise ValueError(f"feature_columns {missing} not in DataFrame")
            return df[self.feature_columns].copy()
        else:
            return df.copy()

    def process(
        self,
        raw_X: Union[str, pd.DataFrame],
        raw_y: Any = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Given raw_X (path or DataFrame that contains both features+targets),
        return (X_df, y_df), where:
          - X_df drops "ID", all Viscosity_* columns, and any drop_columns.
          - y_df contains only the Viscosity_* columns in the same order.

        raw_y is ignored here (for compatibility with Predictor.update signature).
        """
        # Load full DataFrame
        if isinstance(raw_X, str):
            df_full = pd.read_csv(raw_X)
        elif isinstance(raw_X, pd.DataFrame):
            df_full = raw_X.copy()
        else:
            raise ValueError(f"Unsupported raw_X type: {type(raw_X)}")

        # --- targets: columns that start with target_prefix
        all_cols = df_full.columns.tolist()
        target_cols = [c for c in all_cols if c.startswith(self.target_prefix)]
        if not target_cols:
            raise ValueError(
                f"No target columns found with prefix '{self.target_prefix}'")

        y_df = df_full[target_cols].copy()

        # --- features: drop ID + Viscosity_* + any extra drop_columns
        feature_df = df_full.drop(
            columns=["ID", *target_cols], errors="ignore")
        feature_df = feature_df.drop(
            columns=self.drop_columns, errors="ignore")

        # If user explicitly specified feature_columns, enforce it
        if self.feature_columns:
            missing = [
                c for c in self.feature_columns if c not in feature_df.columns]
            if missing:
                raise ValueError(f"feature_columns {missing} not in DataFrame")
            X_df = feature_df[self.feature_columns].copy()
        else:
            X_df = feature_df.copy()

        # Fill or handle missing values (example: fill with 0)
        X_df = X_df.fillna(0)
        y_df = y_df.fillna(0)

        return X_df, y_df
