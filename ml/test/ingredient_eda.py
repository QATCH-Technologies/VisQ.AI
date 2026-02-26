import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/formulation_data_02162026.csv")

# 1. Calculate Average Viscosity Profile
visc_cols = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]
df["avg_viscosity"] = df[visc_cols].mean(axis=1)

# 2. Create Binary "Presence" Flags for Additional Ingredients
ingredients_list = ["Stabilizer", "Excipient", "Surfactant", "Salt"]
for ing in ingredients_list:
    col_name = f"{ing}_type"
    df[f"has_{ing}"] = df[col_name].apply(
        lambda x: 0 if pd.isna(x) or str(x).lower().strip() == "none" else 1
    )

ingredients = ["has_Stabilizer", "has_Excipient", "has_Surfactant", "has_Salt"]

# 3. Create a Normalized Viscosity Metric (Concentration Agnostic)
# We filter out rows where Protein_conc is 0 to avoid division by zero.
df_prot = df[df["Protein_conc"] > 0].copy()
df_prot["norm_viscosity"] = df_prot["avg_viscosity"] / df_prot["Protein_conc"]

results = []

# ==========================================
# --- OVERALL CORRELATIONS ---
# ==========================================

# Standard Average Viscosity
corr_overall = df[ingredients + ["avg_viscosity"]].corr()["avg_viscosity"][ingredients]
results.append(
    {
        "Scope": "Overall (All Data)",
        "Metric": "Average Viscosity",
        "Sample_Size_n": len(df),
        "Stabilizer_Correlation": corr_overall["has_Stabilizer"],
        "Excipient_Correlation": corr_overall["has_Excipient"],
        "Surfactant_Correlation": corr_overall["has_Surfactant"],
        "Salt_Correlation": corr_overall["has_Salt"],
    }
)

# Normalized Viscosity (Concentration Agnostic)
corr_overall_norm = df_prot[ingredients + ["norm_viscosity"]].corr()["norm_viscosity"][
    ingredients
]
results.append(
    {
        "Scope": "Overall (Protein > 0)",
        "Metric": "Normalized Viscosity (per unit concentration)",
        "Sample_Size_n": len(df_prot),
        "Stabilizer_Correlation": corr_overall_norm["has_Stabilizer"],
        "Excipient_Correlation": corr_overall_norm["has_Excipient"],
        "Surfactant_Correlation": corr_overall_norm["has_Surfactant"],
        "Salt_Correlation": corr_overall_norm["has_Salt"],
    }
)

# ==========================================
# --- CORRELATIONS BY PROTEIN CLASS ---
# ==========================================

# Standard Correlations
for ptype, group in df.groupby("Protein_class_type"):
    if len(group) > 5:
        corr = group[ingredients + ["avg_viscosity"]].corr()["avg_viscosity"][
            ingredients
        ]
        results.append(
            {
                "Scope": f"Protein Class: {ptype}",
                "Metric": "Average Viscosity",
                "Sample_Size_n": len(group),
                "Stabilizer_Correlation": corr["has_Stabilizer"],
                "Excipient_Correlation": corr["has_Excipient"],
                "Surfactant_Correlation": corr["has_Surfactant"],
                "Salt_Correlation": corr["has_Salt"],
            }
        )

# Normalized Correlations
for ptype, group in df_prot.groupby("Protein_class_type"):
    if len(group) > 5:
        corr_norm = group[ingredients + ["norm_viscosity"]].corr()["norm_viscosity"][
            ingredients
        ]
        results.append(
            {
                "Scope": f"Protein Class: {ptype}",
                "Metric": "Normalized Viscosity (per unit concentration)",
                "Sample_Size_n": len(group),
                "Stabilizer_Correlation": corr_norm["has_Stabilizer"],
                "Excipient_Correlation": corr_norm["has_Excipient"],
                "Surfactant_Correlation": corr_norm["has_Surfactant"],
                "Salt_Correlation": corr_norm["has_Salt"],
            }
        )

# ==========================================
# Formatting and Exporting
# ==========================================

results_df = pd.DataFrame(results)

# Sort values to group scopes together so it reads side-by-side
results_df = results_df.sort_values(by=["Scope", "Metric"])

# Replace NaN values (which occur when an ingredient is always or never present in a group)
results_df = results_df.fillna("No Variation (N/A)")

# Export to CSV
output_filename = "viscosity_ingredient_correlations_with_normalized.csv"
results_df.to_csv(output_filename, index=False)
print(f"Exported successfully to {output_filename}")
