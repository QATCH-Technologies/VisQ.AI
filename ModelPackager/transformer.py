import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ScalerPipeline:

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
        ])
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "ScalerPipeline":
        self.feature_names_ = X.columns.tolist()
        self.pipeline.fit(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        arr = self.pipeline.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.pipeline.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_, index=X.index)


class TransformerPipeline:
    """
    Transformer that:
      - Converts any "none" (case-insensitive) or NaN in non-`_type` columns to 0
      - Scales all columns NOT ending with '_type'
      - Leaves '_type' columns exactly as-is
      - Returns a single DataFrame with ALL columns, in original order
    """

    def __init__(self, type_suffix: str = "_type"):
        self.scaler = ScalerPipeline()
        self.type_suffix = type_suffix

    def _split(self, X: pd.DataFrame):
        cols = list(X.columns)
        type_cols = [c for c in cols if c.endswith(self.type_suffix)]
        num_cols = [c for c in cols if c not in type_cols]
        return num_cols, type_cols

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_df = X.copy()
        num_cols, type_cols = self._split(X_df)

        num_block = (
            X_df[num_cols]
            # "none" → 0
            .replace(to_replace=r'(?i)^none$', value=0, regex=True)
            .fillna(0)                                               # NaN → 0
        )
        scaled_num = self.scaler.fit_transform(num_block)
        result = scaled_num.join(X_df[type_cols])
        print(result)
        return result[X_df.columns]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        num_cols, type_cols = self._split(X_df)
        num_block = (
            X_df[num_cols]
            .replace(to_replace=r'(?i)^none$', value=0, regex=True)
            .fillna(0)
        )
        scaled_num = self.scaler.transform(num_block)

        result = scaled_num.join(X_df[type_cols])
        return result[X_df.columns]
