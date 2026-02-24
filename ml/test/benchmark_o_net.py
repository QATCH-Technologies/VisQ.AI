import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import your existing inference class
# Ensure inference_o_net.py is in the same directory or PYTHONPATH
try:
    from inference_o_net import ViscosityPredictorCNP
except ImportError:
    print("Error: Could not import ViscosityPredictorCNP from inference_o_net.py.")
    print("Make sure benchmark_o_net.py is in the same folder as inference_o_net.py")
    sys.exit(1)

# ==========================================
# Configuration
# ==========================================
DATA_PATH = "data/raw/formulation_data_02162026.csv"  # Update this path if needed
MODEL_DIR = "models/experiments/o_net_dense"  # Update this path if needed

SHEAR_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")

    df = pd.read_csv(path)

    # Basic cleanup from your inference script
    int_cols = df.select_dtypes(include=["int", "int64", "int32"]).columns
    for col in int_cols:
        if col != "ID":
            df[col] = df[col].astype(float)

    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)

    return df


def calculate_metrics(true_vals, pred_vals):
    if len(true_vals) < 2:
        return np.nan, np.nan, np.nan

    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, pred_vals)
    r2 = r2_score(true_vals, pred_vals)
    return rmse, mae, r2


def main():
    print(f"--- Starting O-Net Benchmark ---")
    print(f"Model Dir: {MODEL_DIR}")
    print(f"Data Path: {DATA_PATH}")

    # 1. Initialize Predictor
    try:
        predictor = ViscosityPredictorCNP(MODEL_DIR)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return

    # 2. Load Data
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        print(str(e))
        return

    # 3. Group by Protein
    # The 'learn' step is designed to adapt to a specific protein context.
    # We loop through each protein, learn from it, and then check the fit.
    groups = df.groupby("Protein_type")

    print(f"\nEvaluating {len(df)} samples across {len(groups)} protein groups...")

    # Storage for results
    all_results = {col: {"true": [], "pred": []} for col in SHEAR_COLS}
    protein_metrics = []

    for protein_name, group_df in groups:
        print(f"Processing {protein_name} ({len(group_df)} samples)...", end=" ")

        # A. Learn (Calibrate)
        # This resets the internal memory_vector to this specific group
        predictor.learn(group_df, steps=50)  # Using 50 steps as in your default

        # B. Predict (Check Fit)
        preds_df = predictor.predict(group_df)

        # C. Store Pairs
        for col in SHEAR_COLS:
            pred_col = f"Pred_{col}"

            if col in group_df.columns and pred_col in preds_df.columns:
                # Align data
                combined = pd.concat(
                    [group_df[col], preds_df[pred_col]], axis=1
                ).dropna()

                if not combined.empty:
                    t = combined[col].values
                    p = combined[pred_col].values

                    all_results[col]["true"].extend(t)
                    all_results[col]["pred"].extend(p)

        print("Done.")

    # 4. Generate Report
    print("\n" + "=" * 60)
    print(
        f"{'Shear Rate':<20} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10} | {'Samples':<8}"
    )
    print("-" * 60)

    overall_true = []
    overall_pred = []

    for col in SHEAR_COLS:
        t_list = all_results[col]["true"]
        p_list = all_results[col]["pred"]

        if not t_list:
            print(f"{col:<20} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {0:<8}")
            continue

        t_arr = np.array(t_list)
        p_arr = np.array(p_list)

        rmse, mae, r2 = calculate_metrics(t_arr, p_arr)
        count = len(t_arr)

        print(f"{col:<20} | {rmse:<10.4f} | {mae:<10.4f} | {r2:<10.4f} | {count:<8}")

        overall_true.extend(t_list)
        overall_pred.extend(p_list)

    print("-" * 60)

    # Global Metrics
    if overall_true:
        g_rmse, g_mae, g_r2 = calculate_metrics(
            np.array(overall_true), np.array(overall_pred)
        )
        print(
            f"{'OVERALL':<20} | {g_rmse:<10.4f} | {g_mae:<10.4f} | {g_r2:<10.4f} | {len(overall_true):<8}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
