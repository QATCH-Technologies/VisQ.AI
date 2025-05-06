import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from itertools import combinations

TARGET_COLS = [
    "Viscosity100",
    "Viscosity1000",
    "Viscosity10000",
    "Viscosity100000",
    "Viscosity15000000"
]
NUMERIC_FEATURES = [
    "MW(kDa)",
    "PI_mean",
    "PI_range",
    "Protein",
    "Temperature",
    "Sugar(M)",
    "Concentration"
]
CATEGORICAL_FEATURES = [
    "Protein type",
    "Buffer",
    "Sugar",
    "Surfactant"
]


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Apply complex feature‐engineering steps to the raw DataFrame.
    Produces a DataFrame with:
     - numeric transforms (log, poly, sqrt, inverse)
     - binary indicators (above‐median)
     - ratio‐to‐sum features
     - quartile bin features
     - generic pairwise numeric interactions
     - group‐level stats (mean, std) per categorical
     - original categorical columns (strings)
    """

    def __init__(self,
                 numeric_features=NUMERIC_FEATURES,
                 categorical_features=CATEGORICAL_FEATURES):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        # to store stats
        self.medians_ = {}
        self.bins_ = {}
        self.group_mean_ = {}
        self.group_std_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        # prepare cleaned numeric array
        df_num = df[self.numeric_features].apply(pd.to_numeric, errors="coerce").replace([
            np.inf, -np.inf], np.nan).fillna(0)
        # medians & bins
        for f in self.numeric_features:
            self.medians_[f] = df_num[f].median()
            try:
                _, bins = pd.qcut(
                    df_num[f], q=4, retbins=True, duplicates='drop')
                self.bins_[f] = bins
            except ValueError:
                self.bins_[f] = np.unique(df_num[f])
        # group stats
        for cat in self.categorical_features:
            if cat not in df:
                continue
            cat_series = df[cat].astype(str).fillna("none")
            for f in self.numeric_features:
                grp = df_num.groupby(cat_series)[f]
                self.group_mean_[(cat, f)] = grp.mean().to_dict()
                self.group_std_[(cat, f)] = grp.std().to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        # clean numerics
        df_num = df[self.numeric_features].apply(pd.to_numeric, errors="coerce").replace([
            np.inf, -np.inf], np.nan).fillna(0)
        new_cols = {}
        # numeric transforms
        for f in self.numeric_features:
            x = df_num[f]
            new_cols[f] = x
            new_cols[f + '_log'] = np.log1p(x.clip(lower=0))
            new_cols[f + '_squared'] = x**2
            new_cols[f + '_cubed'] = x**3
            new_cols[f + '_sqrt'] = np.sqrt(x.clip(lower=0))
            inv = 1 / x.replace(0, np.nan)
            new_cols[f + '_inv'] = inv.fillna(0)
            new_cols[f + '_above_median'] = (x >
                                             self.medians_.get(f, 0)).astype(int)
        # ratio to sum
        total = pd.DataFrame({f: new_cols[f] for f in self.numeric_features}).sum(
            axis=1).replace(0, np.nan)
        for f in self.numeric_features:
            new_cols[f + '_ratio_sum'] = (new_cols[f] / total).fillna(0)
        # quartile bins
        for f in self.numeric_features:
            bins = self.bins_.get(f)
            if bins is not None:
                new_cols[f + '_bin'] = pd.cut(new_cols[f], bins=bins,
                                              labels=False, include_lowest=True).fillna(0).astype(int)
        # pairwise interactions
        for f1, f2 in combinations(self.numeric_features, 2):
            new_cols[f1 + '_' + f2 +
                     '_interaction'] = new_cols[f1] * new_cols[f2]
        # group stats mapping
        for cat in self.categorical_features:
            if cat not in df:
                continue
            cs = df[cat].astype(str).fillna("none")
            for f in self.numeric_features:
                mean_map = self.group_mean_.get((cat, f), {})
                std_map = self.group_std_.get((cat, f), {})
                new_cols[f'{cat}_{f}_mean'] = cs.map(
                    mean_map).fillna(df_num[f].mean())
                new_cols[f'{cat}_{f}_std'] = cs.map(
                    std_map).fillna(df_num[f].std())
        # assemble numeric df
        df_features = pd.DataFrame(new_cols, index=df.index)
        # pull through categoricals
        df_cat = df[self.categorical_features].astype(str).fillna("none")
        return pd.concat([df_features, df_cat], axis=1)


class VisQDataProcessor:
    def __init__(self):
        self.feature_gen = FeatureGenerator()
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    CATEGORICAL_FEATURES
                ),
            ],
            remainder=StandardScaler()
        )

        self.pipeline = Pipeline([
            ("features", self.feature_gen),
            ("prep",     self.preprocessor)
        ])

    @staticmethod
    def load_content(load_directory: str) -> pd.DataFrame:
        if not os.path.exists(load_directory):
            raise IOError(
                f"Content load path does not exist at `{load_directory}`")
        return pd.read_csv(load_directory)

    def fit(self, df: pd.DataFrame):
        present = [c for c in TARGET_COLS if c in df.columns]
        print(present)
        if not present:
            raise KeyError(
                "None of the target columns were found in the data.")
        y = df[present].copy()
        X_raw = df.drop(columns=TARGET_COLS, errors="ignore")
        X_trans = self.pipeline.fit_transform(X_raw)
        feature_names = self.pipeline.named_steps["prep"].get_feature_names_out(
        )
        X_df = pd.DataFrame(X_trans, columns=feature_names, index=X_raw.index)
        return X_df, y

    def transform(self, df: pd.DataFrame):
        X_raw = df.copy()
        X_trans = self.pipeline.transform(X_raw)

        feature_names = self.pipeline.named_steps["prep"].get_feature_names_out(
        )
        return pd.DataFrame(X_trans, columns=feature_names, index=X_raw.index)

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df)


# -------------------------
# Example usage:
# -------------------------
if __name__ == "__main__":
    # Training:
    df = VisQDataProcessor.load_content(
        "content/formulation_data_05062025.csv")
    processor = VisQDataProcessor()
    X_train, y_train = processor.fit(df)

    # Save the fitted pipeline for later inference:
    import joblib
    joblib.dump(processor.pipeline, "visq_pipeline.pkl")

    # --- Later, at prediction time ---
    new_data = pd.DataFrame([{
        "Protein type":  "BSA",
        "MW(kDa)":       66.5,
        "PI_mean":       6.5,
        "PI_range":      0.2,
        "Protein":       1.2,
        "Temperature":   25,
        "Buffer":        "PBS",
        "Sugar":         "Trehalose",
        "Sugar(M)":      0.5,
        "Surfactant":    "tween-20",
        "Concentration": 5.0
    }])
    # load pipeline
    pipeline = joblib.load("visq_pipeline.pkl")
    X_new = pipeline.transform(new_data)
    # X_new is a NumPy array; you can wrap in a DataFrame if you like:
    cols = pipeline.named_steps["prep"].get_feature_names_out()
    X_new_df = pd.DataFrame(X_new, columns=cols)

    print(X_new_df)
