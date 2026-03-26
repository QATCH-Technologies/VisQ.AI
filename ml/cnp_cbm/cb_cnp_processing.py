import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==========================================
# Domain Constants
# ==========================================
PROTEIN_CLASS_MAP = {
    "adalimumab": "igg1",
    "bevacizumab": "igg1",
    "trastuzumab": "igg1",
    "pembrolizumab": "igg4",
    "ibalizumab": "igg4",
    "nivolumab": "igg4",
    "belatacept": "fc_fusion",
    "etanercept": "fc_fusion",
    "vudalimab": "bispecific",
    "poly-higg": "polyclonal",
    "bgg": "polyclonal",
    "bsa": "other",
}


class FormulationProcessor:
    """
    Handles data cleaning, physics-based feature engineering, and scaling for formulation data.
    Includes interpretability methods to audit the engineered proxy features.
    """

    def __init__(self):
        self.preprocessor = None
        self.cat_cols = [
            "Protein_type",
            "Protein_class_type",
            "Buffer_type",
            "Salt_type",
            "Stabilizer_type",
            "Surfactant_type",
            "Excipient_type",
        ]
        self.num_cols = [
            "kP",
            "MW",
            "PI_mean",
            "PI_range",
            "Protein_conc",
            "Temperature",
            "Buffer_pH",
            "Buffer_conc",
            "Salt_conc",
            "Stabilizer_conc",
            "Surfactant_conc",
            "Excipient_conc",
            "C_Class",
            "HCI",
            # Engineered Features:
            "Total_Solute_Mass",
            "Effective_Protein_Fraction",
            "Phi_Protein",
            "Phi_Stabilizer",
            "Phi_Total",
            "Exp_Crowding",
            "Ionic_Strength_Proxy",
            "Crowding_Index",
            "Stabilizer_Squared",
            "KD_Asymptote",
            "log_conc",
            "conc_sq",
            "conc_x_HCI",
            "conc_x_kP",
            "charge_x_ionic",
        ]
        self.feature_names_out = None

    def load_and_clean(self, filepath: str) -> pd.DataFrame:
        """Loads CSV, normalizes strings, and fixes specific unit errors."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)

        # Normalize target strings
        df["Protein_type"] = df["Protein_type"].str.lower().str.strip()
        df["Protein_class_type"] = df["Protein_type"].map(PROTEIN_CLASS_MAP).fillna("unknown")

        return df

    def engineer_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies physical chemistry equations to generate proxy features."""
        df_proc = df.copy()

        # 1. Mass and Volume Fractions (Phi)
        df_proc["Total_Solute_Mass"] = (
            df_proc["Protein_conc"]
            + df_proc["Buffer_conc"]
            + df_proc["Salt_conc"]
            + df_proc["Stabilizer_conc"]
            + df_proc["Excipient_conc"]
            + df_proc["Surfactant_conc"]
        )
        df_proc["Effective_Protein_Fraction"] = df_proc["Protein_conc"] / (
            df_proc["Total_Solute_Mass"] + 1e-6
        )

        # Assuming specific volumes (approximate)
        df_proc["Phi_Protein"] = df_proc["Protein_conc"] * 0.73 / 1000.0
        df_proc["Phi_Stabilizer"] = df_proc["Stabilizer_conc"] * 0.62 / 1000.0
        df_proc["Phi_Total"] = df_proc["Phi_Protein"] + df_proc["Phi_Stabilizer"]

        # 2. Rheological & Electrostatic Proxies
        safe_phi = np.clip(df_proc["Phi_Protein"], 0, 0.7)
        df_proc["Exp_Crowding"] = np.exp(safe_phi * 2.5)
        df_proc["Ionic_Strength_Proxy"] = np.sqrt(df_proc["Salt_conc"] / 1000.0)
        df_proc["Crowding_Index"] = df_proc["Phi_Protein"] * (1 + df_proc["Phi_Stabilizer"])
        df_proc["Stabilizer_Squared"] = df_proc["Stabilizer_conc"] ** 2
        df_proc["KD_Asymptote"] = df_proc["kP"] / (1 + df_proc["Phi_Protein"])
        df_proc["log_conc"] = np.log1p(df_proc["Protein_conc"])

        # 3. CBM-Specific Interaction Terms
        df_proc["conc_sq"] = df_proc["Protein_conc"] ** 2
        df_proc["conc_x_HCI"] = df_proc["Protein_conc"] * df_proc["HCI"]
        df_proc["conc_x_kP"] = df_proc["Protein_conc"] * df_proc["kP"]
        df_proc["charge_x_ionic"] = df_proc["C_Class"] * df_proc["Ionic_Strength_Proxy"]

        # Fill missing values
        for col in self.num_cols:
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].fillna(0.0)
        for col in self.cat_cols:
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].fillna("none")

        return df_proc

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fits the scaler/encoder and transforms the data."""
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols),
            ]
        )

        X_processed = self.preprocessor.fit_transform(df)

        # Extract feature names for downstream importance tracking
        num_names = [f"num__{c}" for c in self.num_cols]
        cat_names = self.preprocessor.named_transformers_["cat"].get_feature_names_out(
            self.cat_cols
        )
        self.feature_names_out = num_names + list(cat_names)

        return X_processed

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms new data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Processor must be fit before calling transform().")
        return self.preprocessor.transform(df)

    # ==========================================
    # Interpretability Methods
    # ==========================================

    def generate_proxy_correlation_matrix(self, df: pd.DataFrame, save_path: str = None):
        """
        Plots a heatmap of correlations specifically for the CBM physical proxies.
        This ensures your CBM targets aren't perfectly collinear (e.g., measuring the same thing).
        """
        proxy_cols = [
            "conc_x_kP",
            "conc_x_HCI",
            "charge_x_ionic",
            "Ionic_Strength_Proxy",
            "Protein_conc",
            "conc_sq",
            "Stabilizer_conc",
            "Crowding_Index",
        ]

        # Ensure columns exist
        available_cols = [c for c in proxy_cols if c in df.columns]
        corr = df[available_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, fmt=".2f", square=True
        )
        plt.title("CBM Physical Proxy Correlations")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved proxy correlation matrix to {save_path}")
        plt.show()

    def plot_physics_distributions(self, df: pd.DataFrame, save_path: str = None):
        """
        Generates paired plots to visualize how the engineered physics
        features behave relative to raw protein concentration.
        """
        features_to_plot = ["Exp_Crowding", "conc_x_HCI", "charge_x_ionic", "conc_sq"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, feature in enumerate(features_to_plot):
            if feature in df.columns:
                sns.scatterplot(
                    data=df,
                    x="Protein_conc",
                    y=feature,
                    hue="Protein_type",
                    alpha=0.6,
                    ax=axes[i],
                    legend=(i == 0),
                )
                axes[i].set_title(f"{feature} vs Concentration")
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved physics distribution plots to {save_path}")
        plt.show()


if __name__ == "__main__":
    # 1. Initialize Processor
    processor = FormulationProcessor()

    # 2. Load and Engineer
    raw_df = processor.load_and_clean("data/raw/formulation_data_03042026.csv")
    df_engineered = processor.engineer_physics_features(raw_df)

    # --- [INTERPRETABILITY] Audit the features before training ---
    print("Generating feature interpretability reports...")
    processor.generate_proxy_correlation_matrix(df_engineered, save_path="proxy_correlations.png")
    processor.plot_physics_distributions(df_engineered, save_path="physics_distributions.png")

    # 3. Fit, Scale, and Encode
    static_feats_np = processor.fit_transform(df_engineered)
    feature_names = processor.feature_names_out
