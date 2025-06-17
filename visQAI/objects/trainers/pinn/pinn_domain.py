# data_loader.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline


class DataLoader:
    INDEX_COL = "ID"
    NUMERIC_FEATURES = [
        "MW", "PI_mean", "PI_range", "Protein_concentration",
        "Temperature", "Buffer_pH", "Sugar_concentration",
        "Surfactant_concentration", "Buffer_conc", "NaCl",
    ]
    CATEGORICAL_FEATURES = [
        "Protein_type", "Buffer_type", "Sugar_type", "Surfactant_type",
    ]
    TARGET_COLUMNS = [
        "Viscosity_100", "Viscosity_1000", "Viscosity_10000",
        "Viscosity_100000", "Viscosity_15000000",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None
        self._preprocessor: Optional[ColumnTransformer] = None

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        if self.INDEX_COL in df.columns:
            df = df.drop(columns=[self.INDEX_COL])

        # numeric → fillna(0)
        df[self.NUMERIC_FEATURES] = (
            df[self.NUMERIC_FEATURES]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        # unify 'none' → NaN → 'missing'
        df[self.CATEGORICAL_FEATURES] = (
            df[self.CATEGORICAL_FEATURES]
            .replace({"none": np.nan, "None": np.nan})
            .fillna("missing")
        )

        self._df = df
        return df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Data not loaded. Call `load()` first.")
        return self._df

    def get_raw_features(self) -> pd.DataFrame:
        return self.df[self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES].copy()

    def get_targets(self) -> np.ndarray:
        return self.df[self.TARGET_COLUMNS].values

    def build_preprocessor(self) -> ColumnTransformer:
        if self._df is None:
            self.load()

        # scale all your numeric features as before
        num_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])

        # ordinal‐encode each categorical column to a single integer column
        cat_pipeline = Pipeline([
            ("ordinal", OrdinalEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.NUMERIC_FEATURES),
                ("cat", cat_pipeline, self.CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )

        preprocessor.fit(self.get_raw_features())
        self._preprocessor = preprocessor

        return preprocessor

    @property
    def preprocessor(self) -> ColumnTransformer:
        if self._preprocessor is None:
            self.build_preprocessor()
        return self._preprocessor

    def get_processed_features(self) -> np.ndarray:
        return self.preprocessor.transform(self.get_raw_features())

    def split(
        self,
        preprocess: bool = False
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
        if self._df is None:
            self.load()
        X = self.get_processed_features() if preprocess else self.get_raw_features()
        y = self.get_targets()
        return X, y
