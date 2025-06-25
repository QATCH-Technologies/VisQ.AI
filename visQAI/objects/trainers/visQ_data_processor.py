from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from itertools import combinations
from typing import Dict

TARGET_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000"
]
NUMERIC_FEATURES = [
    "MW", "PI_mean", "PI_range", "Protein_conc",
    "Temperature", "Buffer_pH", "Stabilizer_conc",
    "Surfactant_conc", "Buffer_conc", "Salt_conc",
]
CATEGORICAL_FEATURES = [
    "Protein_type", "Buffer_type", "Stabilizer_type", "Surfactant_type", "Salt_type",
]


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Apply complex feature-engineering steps to the raw DataFrame,
    including physics-based temperature and pH scaling.
    """

    def __init__(self,
                 numeric_features=NUMERIC_FEATURES,
                 categorical_features=CATEGORICAL_FEATURES,
                 # physics params
                 Ea: float = 5000.0,        # activation energy (J/mol)
                 R: float = 8.314,          # gas constant (J/mol·K)
                 use_arrhenius: bool = True,
                 use_pH_charge: bool = True):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.medians_: Dict[str, float] = {}
        # physics
        self.Ea = Ea
        self.R = R
        self.use_arrhenius = use_arrhenius
        self.use_pH_charge = use_pH_charge

    def fit(self, X, y=None):
        # compute medians as before...
        df_num = (
            X[self.numeric_features]
            .replace({"none": 0, "NaN": 0})
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        for f in self.numeric_features:
            self.medians_[f] = df_num[f].median()
        return self

    def transform(self, X):
        df = X.copy()
        df_num = (
            df[self.numeric_features]
            .replace({"none": 0, "NaN": 0})
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        feats: Dict[str, pd.Series] = {}

        # 1) standard raw/log/inv/sq/cu/above-median features
        for f in self.numeric_features:
            x = df_num[f]
            feats[f] = x
            feats[f + "_log"] = np.log1p(x.clip(lower=0))
            inv = 1 / x.replace(0, np.nan)
            feats[f + "_inv"] = inv.fillna(0)
            feats[f + "_squared"] = x ** 2
            feats[f + "_cubed"] = x ** 3
            feats[f + "_above_med"] = (x > self.medians_[f]).astype(int)

        # 2) conc interactions as before...
        p = df_num["Protein_conc"]
        s = df_num["Stabilizer_conc"]
        t = df_num["Surfactant_conc"]
        feats["prot_plus_Stabilizer"] = p + s
        feats["prot_times_Stabilizer"] = p * s
        feats["total_excipient"] = s + t
        feats["Stabilizer_minus_surfactant"] = s - t

        # 3) pI-buffer difference
        pi = df_num["PI_mean"]
        buf_ph = df_num["Buffer_pH"]
        feats["pi_buffer_diff"] = (pi - buf_ph).abs()

        # 4) presence flags
        feats["has_protein"] = (df["Protein_type"] != "none").astype(int)
        feats["has_Stabilizer"] = (df["Stabilizer_type"] != "none").astype(int)
        feats["has_surfactant"] = (df["Surfactant_type"] != "none").astype(int)

        # 5) —— Physics-based features —— #

        # 5a) Temperature in Kelvin + its inverses/logs
        temp_C = df_num["Temperature"]
        T_K = temp_C + 273.15
        feats["Temperature_K"] = T_K
        feats["inv_Temperature_K"] = 1 / T_K
        feats["log_Temperature_K"] = np.log(T_K)

        if self.use_arrhenius:
            feats["Arrhenius_factor"] = np.exp(- self.Ea / (self.R * T_K))

        # 5c) pH-charge fraction, if enabled
        if self.use_pH_charge:
            # fraction of charged species relative to pI
            feats["charge_fraction"] = 1 / (1 + 10 ** (pi - buf_ph))
        K, a = 0.5e-3, 0.7
        MW_g_per_mol = df_num["MW"] * 1e3
        eta_int = K * MW_g_per_mol**a
        feats["intrinsic_viscosity"] = eta_int
        feats["pred_visc_from_intrinsic"] = eta_int * \
            df_num["Protein_conc"]

        # d) conc power‐laws (viscosity ∝ c^b)
        for b in (1.8, 2.0, 2.2):
            feats[f"prot_conc_pow_{b}"] = df_num["Protein_conc"] ** b
            feats[f"Stabilizer_conc_pow_{b}"] = df_num["Stabilizer_conc"] ** b
        p_g_per_L = df_num["Protein_conc"]      # mg/mL == g/L
        p_mol = p_g_per_L / MW_g_per_mol
        total_m = p_mol + df_num["Stabilizer_conc"]
        T_K = df_num["Temperature"] + 273.15
        feats["osmotic_pressure"] = total_m * self.R * T_K

        # 6) assemble final DataFrame
        df_feats = pd.DataFrame(feats, index=df.index)
        return pd.concat([df_feats, df[self.categorical_features]], axis=1)


class RawFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select and clean raw numeric and categorical features without engineering.
    """

    def __init__(self,
                 numeric_features=NUMERIC_FEATURES,
                 categorical_features=CATEGORICAL_FEATURES):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df_num = df[self.numeric_features] \
            .apply(pd.to_numeric, errors="coerce") \
            .replace([np.inf, -np.inf], np.nan) \
            .fillna(0)
        df_cat = df[self.categorical_features].astype(str).fillna("none")
        return pd.concat([df_num, df_cat], axis=1)


class VisQDataProcessor:
    """
    Data processor with optional feature engineering toggle.
    If `use_feature_engineering` is False, only raw numeric features (cleaned)
    and one-hot encoded categoricals are used.
    """

    def __init__(self, use_feature_engineering: bool = True):
        self.use_feature_engineering = use_feature_engineering
        self.feature_gen = FeatureGenerator()
        self.raw_selector = RawFeatureSelector()

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False),
                    CATEGORICAL_FEATURES
                ),
            ],
            remainder=MinMaxScaler()
        )

        steps = []
        if self.use_feature_engineering:
            steps.append(("features", self.feature_gen))
        else:
            steps.append(("raw", self.raw_selector))
        steps.append(("prep", self.preprocessor))
        self.pipeline = Pipeline(steps)

    @staticmethod
    def load_content(load_directory: str) -> pd.DataFrame:
        if not os.path.exists(load_directory):
            raise IOError(
                f"Content load path does not exist at `{load_directory}`")
        return pd.read_csv(load_directory)

    def fit(self, df: pd.DataFrame):
        present = [c for c in TARGET_COLS if c in df.columns]
        if not present:
            raise KeyError(
                "None of the target columns were found in the data.")
        y = df[present].copy()
        X_raw = df.drop(columns=TARGET_COLS, errors="ignore")
        X_trans = self.pipeline.fit_transform(X_raw)
        feature_names = self.pipeline.named_steps["prep"].get_feature_names_out(
        )
        X_df = pd.DataFrame(X_trans, columns=feature_names, index=X_raw.index)
        X_df = X_df.fillna(0)
        y = y.fillna(0)
        return X_df, y

    def transform(self, df: pd.DataFrame):
        X_trans = self.pipeline.transform(df)
        feature_names = self.pipeline.named_steps["prep"].get_feature_names_out(
        )
        return pd.DataFrame(X_trans, columns=feature_names, index=df.index)

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df)


# -------------------------
# Example usage:
# -------------------------
if __name__ == "__main__":
    # ─── Load & transform ─────────────────────────────────────────────────────────
    # (Skip this if you've already done it in your session)
    df = VisQDataProcessor.load_content(
        "content/formulation_data_05072025.csv")
    proc_eng = VisQDataProcessor(use_feature_engineering=True)
    X_e, y_e = proc_eng.fit(df)

    # If y_e is a DataFrame with multiple targets, we'll keep them all
    # ─── 1) Pearson correlation heatmap ────────────────────────────────────────────
    # build corr matrix between each feature and each target
    corr_matrix = pd.concat(
        [X_e, y_e], axis=1).corr().loc[X_e.columns, y_e.columns]

    plt.figure()
    plt.imshow(corr_matrix.values, aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)),
               corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.index)),   corr_matrix.index)
    plt.title("Pearson correlation: engineered features vs. targets")
    plt.tight_layout()
    plt.show()

    # ─── 2) Mutual information heatmap ─────────────────────────────────────────────
    # compute MI(feature, target) for each target

    mi = pd.DataFrame(
        {t: mutual_info_regression(X_e, y_e[t]) for t in y_e.columns},
        index=X_e.columns
    )

    plt.figure()
    plt.imshow(mi.values, aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(mi.columns)), mi.columns, rotation=90)
    plt.yticks(range(len(mi.index)),   mi.index)
    plt.title("Mutual information: engineered features vs. targets")
    plt.tight_layout()
    plt.show()

    # ─── 3) Average MI per feature ─────────────────────────────────────────────────
    mi_mean = mi.mean(axis=1).sort_values(ascending=False)

    plt.figure()
    plt.bar(mi_mean.index, mi_mean)
    plt.xticks(rotation=90)
    plt.ylabel("Avg. mutual information")
    plt.title("Average non‐linear association per feature")
    plt.tight_layout()
    plt.show()
