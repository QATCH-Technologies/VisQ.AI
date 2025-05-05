
import pandas as pd
import numpy as np
import os

TARGET_COLS = [
    "Viscosity100",
    "Viscosity1000",
    "Viscosity10000",
    "Viscosity100000",
    "Viscosity15000000"
]
NUMERIC_FEATURES = ["MW(kDa)",
                    "PI_mean",
                    "PI_range",
                    "Protein",
                    "Temperature",
                    "Sugar(M)",
                    "Concentration"]
CATEGORICAL_FEATURES = ["Protein type",
                        "Buffer",
                        "Sugar",
                        "Surfactant"]


class VisQDataProcessor():
    @staticmethod
    def load_content(load_directory: str):
        if not os.path.exists(load_directory):
            raise IOError(
                f"Content load path does not exist at `{load_directory}`")
        df = pd.read_csv(load_directory)
        # Clean dataframe

        return df

    @staticmethod
    def _one_hot_encode(df: pd.DataFrame, features: list = CATEGORICAL_FEATURES) -> pd.DataFrame:
        encoded_df = pd.get_dummies(df, columns=features, prefix=features)
        return encoded_df.astype(int)

    @staticmethod
    def _frequency_encode(df: pd.DataFrame) -> pd.DataFrame:
        df_freq = df.copy()
        for feature in df.columns:
            freq = df[feature].value_counts(normalize=True)
            df_freq[f"{feature}_freq"] = df[feature].map(freq)
        return df_freq

    @staticmethod
    def _categorical_interaction(df: pd.DataFrame) -> pd.DataFrame:
        # 1) Build a dict of all interaction series
        interactions = {}
        features = df.columns.tolist()
        for i, feature_a in enumerate(features):
            for feature_b in features[i+1:]:
                col_name = f"{feature_a}_{feature_b}_interaction"
                interactions[col_name] = (
                    df[feature_a].astype(str)
                    .str.strip()
                    + "_"
                    + df[feature_b].astype(str)
                    .str.strip()
                )
        # 2) Make a DataFrame of just the new columns
        interactions_df = pd.DataFrame(interactions, index=df.index)

        # 3) Concatenate once
        return pd.concat([df, interactions_df], axis=1)

    @staticmethod
    def _log_transform(df: pd.DataFrame, features: list = NUMERIC_FEATURES) -> pd.DataFrame:
        df_log = df.copy()
        for feature in features:
            df_log[feature] = pd.to_numeric(
                df_log[feature], errors='coerce').fillna(0)
            df_log[f"{feature}_log"] = np.log1p(df_log[feature].clip(lower=0))
        return df_log

    @staticmethod
    def _polynomial_features(df: pd.DataFrame, features: list = NUMERIC_FEATURES) -> pd.DataFrame:
        """Generate polynomial features to capture nonlinear relationships."""
        df_poly = df.copy()
        for feature in features:
            df_poly[f"{feature}_squared"] = df_poly[feature] ** 2
            df_poly[f"{feature}_cubed"] = df_poly[feature] ** 3
        return df_poly

    @staticmethod
    def _interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between relevant numeric features."""
        df_interact = df.copy()
        df_interact['Protein_Conc_Interaction'] = df['Protein'] * \
            df['Concentration']
        df_interact['Temp_Sugar_Interaction'] = df['Temperature'] * \
            df['Sugar(M)']
        df_interact['MW_PImean_Interaction'] = df['MW(kDa)'] * df['PI_mean']
        return df_interact

    @staticmethod
    def _inverse_features(df: pd.DataFrame, features: list = NUMERIC_FEATURES) -> pd.DataFrame:
        """Generate inverse features for capturing inverse relationships."""
        df_inv = df.copy()
        for feature in features:
            # compute inverse, turning zeros into NaN so we don’t get division-by-zero
            inv = 1 / df_inv[feature].replace(0, np.nan)
            # fill NaNs with 0 and assign back in one go
            df_inv[f"{feature}_inv"] = inv.fillna(0)
        return df_inv

    @staticmethod
    def _normalize_features(df: pd.DataFrame, features: list = NUMERIC_FEATURES) -> pd.DataFrame:
        """Min-Max normalization for numeric features."""
        df_norm = df.copy()
        for feature in features:
            min_val, max_val = df_norm[feature].min(), df_norm[feature].max()
            df_norm[f"{feature}_norm"] = (
                df_norm[feature] - min_val) / (max_val - min_val)
        return df_norm

    @staticmethod
    def _generate_features(df: pd.DataFrame) -> pd.DataFrame:
        df_fe = df.copy()

        # Numeric transformations
        df_num = df_fe[NUMERIC_FEATURES].copy()
        df_num.replace([np.inf, -np.inf, 'none', None, 'NaN'], 0, inplace=True)
        df_num.fillna(0, inplace=True)
        df_num = VisQDataProcessor._log_transform(df_num)
        df_num = VisQDataProcessor._polynomial_features(df_num)
        df_num = VisQDataProcessor._interaction_features(df_num)
        df_num = VisQDataProcessor._inverse_features(df_num)
        df_num = VisQDataProcessor._normalize_features(df_num)

        # Categorical transformations
        df_cat = df_fe[CATEGORICAL_FEATURES].copy()
        df_cat = VisQDataProcessor._one_hot_encode(df_cat)
        df_cat = VisQDataProcessor._frequency_encode(df_cat)
        # df_cat = VisQDataProcessor._categorical_interaction(df_cat)

        # Merge numeric and categorical features
        df_combined = pd.concat([df_num, df_cat], axis=1)
        return df_combined

    @staticmethod
    def process(load_directory: str):
        df = VisQDataProcessor.load_content(load_directory)
        present_targets = [c for c in TARGET_COLS if c in df.columns]
        if present_targets:
            y = df[present_targets].copy()
        else:
            raise KeyError(
                "None of the target columns were found in the data.")
        feature_df = df.drop(columns=TARGET_COLS, errors='ignore')
        X = VisQDataProcessor._generate_features(feature_df)
        return X, y
