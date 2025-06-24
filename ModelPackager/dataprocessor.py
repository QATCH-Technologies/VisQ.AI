
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

    def process_train(self, data: Union[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training data, validate presence of target columns, and return feature matrix X and target matrix y.
        """
        # --- load into a single DataFrame ---
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # --- identify target columns ---
        all_cols = df.columns
        target_cols = [c for c in all_cols if c.startswith(self.target_prefix)]

        # --- validate presence of target cols ---
        if not target_cols:
            raise ValueError(
                f"No target columns found with prefix '{self.target_prefix}'"
            )

        # --- extract y and fill missing values ---
        y_df = df[target_cols].fillna(0)

        # --- build feature matrix ---
        to_drop = ["ID"] + target_cols + list(self.drop_columns)
        feats = df.drop(columns=to_drop, errors="ignore")
        if self.feature_columns:
            missing = set(self.feature_columns) - set(feats.columns)
            if missing:
                raise ValueError(
                    f"feature_columns {sorted(missing)} not in DataFrame"
                )
            feats = feats[self.feature_columns]
        X_df = feats.fillna(0)
        return X_df, y_df

    def process_predict(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load live data, validate absence of target columns, and return feature matrix X.
        """
        # --- load into a single DataFrame ---
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # --- identify target columns ---
        all_cols = df.columns
        target_cols = [c for c in all_cols if c.startswith(self.target_prefix)]

        # --- validate absence of target cols ---
        if target_cols:
            raise ValueError(
                f"Live input should not contain target columns, but found: {target_cols}"
            )

        # --- build feature matrix ---
        to_drop = ["ID"] + list(self.drop_columns)
        feats = df.drop(columns=to_drop, errors="ignore")
        if self.feature_columns:
            missing = set(self.feature_columns) - set(feats.columns)
            if missing:
                raise ValueError(
                    f"feature_columns {sorted(missing)} not in DataFrame"
                )
            feats = feats[self.feature_columns]
        X_df = feats.fillna(0)

        return X_df
