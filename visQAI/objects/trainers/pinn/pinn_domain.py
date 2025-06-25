import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import FunctionTransformer


class DataLoader:
    INDEX_COL = "ID"
    NUMERIC_FEATURES = [
        "MW", "PI_mean", "PI_range", "Protein_conc",
        "Temperature", "Buffer_pH", "Stabilizer_conc",
        "Surfactant_conc", "Buffer_conc", "Salt_conc",
    ]
    CATEGORICAL_FEATURES = [
        "Protein_type", "Buffer_type", "Stabilizer_type", "Surfactant_type", "Salt_type",
    ]
    TARGET_COLUMNS = [
        "Viscosity_100", "Viscosity_1000", "Viscosity_10000",
        "Viscosity_100000", "Viscosity_15000000",
    ]

    def __init__(
        self,
        csv_path: str,
        preprocessor_path: Optional[str] = None,
    ):
        """
        Args:
            csv_path: Path to input CSV file.
            preprocessor_path: Optional path to a saved preprocessor to load.
        """
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None
        self._preprocessor: Optional[ColumnTransformer] = None
        self.preprocessor_path: Optional[str] = None

        # Load a saved preprocessor if provided
        if preprocessor_path:
            self.load_preprocessor(preprocessor_path)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        if self.INDEX_COL in df.columns:
            df = df.drop(columns=[self.INDEX_COL])

        # Clean numeric features
        df[self.NUMERIC_FEATURES] = (
            df[self.NUMERIC_FEATURES]
            .replace(r'(?i)\bnone\b', np.nan, regex=True)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        # Clean categorical features
        df[self.CATEGORICAL_FEATURES] = (
            df[self.CATEGORICAL_FEATURES]
            .replace(r'(?i)\bnone\b', np.nan, regex=True)
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

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1) Drop INDEX_COL if present
        2) Force 'none' (any case) → NaN everywhere
        3) Coerce numeric cols, fill NaN→0
        4) Fill categoricals NaN→"missing"
        """
        df = df.copy()
        if self.INDEX_COL in df.columns:
            df = df.drop(columns=[self.INDEX_COL])

        # unify any literal “none” to real NaN
        df = df.replace(r'(?i)\bnone\b', np.nan, regex=True)

        # numeric impute+scale
        df[self.NUMERIC_FEATURES] = (
            df[self.NUMERIC_FEATURES]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        # categorical impute
        df[self.CATEGORICAL_FEATURES] = (
            df[self.CATEGORICAL_FEATURES]
            .fillna("missing")
        )

        return df

    def build_preprocessor(self) -> Pipeline:
        """
        Build one pipeline that:
          (a) cleans every new DataFrame via _clean_df,
          (b) applies your ColumnTransformer of scaler + encoder.
        """
        # our cleaning step
        cleaner = FunctionTransformer(self._clean_df, validate=False)

        # numeric → MinMaxScaler (already zeros for any missing)
        num_pipe = Pipeline([
            ("scaler", MinMaxScaler())
        ])

        # categorical → OneHot(ignoring unknowns)
        cat_pipe = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        ct = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.NUMERIC_FEATURES),
                ("cat", cat_pipe, self.CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )

        full = Pipeline([
            ("clean", cleaner),
            ("ct", ct)
        ])

        # fit on your already-loaded df (loads & cleans internally)
        if self._df is None:
            self.load()
        full.fit(self._df)

        self._preprocessor = full
        self.preprocessor_path = None
        return full

    @property
    def preprocessor(self) -> ColumnTransformer:
        """
        Return the fitted preprocessor, building or loading it if necessary.
        """
        if self._preprocessor is None:
            self.build_preprocessor()
        return self._preprocessor

    def save_preprocessor(self, path: str) -> None:
        """
        Save the fitted preprocessor to disk.

        Args:
            path: File path to save the preprocessor (e.g., .joblib).
        """
        if self._preprocessor is None:
            raise RuntimeError(
                "No preprocessor available to save. Call `build_preprocessor()` first.")
        joblib.dump(self._preprocessor, path)
        self.preprocessor_path = path

    def load_preprocessor(self, path: str) -> None:
        """
        Load a preprocessor from disk and store it for transformations.

        Args:
            path: File path of the saved preprocessor.
        """
        self._preprocessor = joblib.load(path)
        self.preprocessor_path = path

    def get_processed_features(self) -> np.ndarray:
        return self.preprocessor.transform(self.get_raw_features())

    def split(
        self,
        preprocess: bool = False
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray]:
        """
        Split dataset into features and targets, optionally applying preprocessing.

        Args:
            preprocess: Whether to transform features using the preprocessor.

        Returns:
            X: Raw or processed features.
            y: Target array.
        """
        if self._df is None:
            self.load()
        X = self.get_processed_features() if preprocess else self.get_raw_features()
        y = self.get_targets()
        return X, y
