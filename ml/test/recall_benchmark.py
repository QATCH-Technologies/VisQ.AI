import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # Added for better side-by-side plots
import sklearn.metrics as metrics

# Add current directory to path
sys.path.append(os.getcwd())

# Import your predictor
from inference_o_net import ViscosityPredictorCNP


# ==========================================
# Metrics Helper
# ==========================================
def calculate_metrics(y_true, y_pred):
    """Computes R2, MAE, RMSE, robust to NaNs."""
    # Convert to numpy arrays to handle Series or lists
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Create a mask to keep only indices where BOTH True and Pred are valid (finite)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    # Check if we have enough data left to calculate metrics
    if len(y_true_clean) < 2:
        return np.nan, np.nan, np.nan

    r2 = metrics.r2_score(y_true_clean, y_pred_clean)
    mae = metrics.mean_absolute_error(y_true_clean, y_pred_clean)
    mse = metrics.mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)

    return r2, mae, rmse


# ==========================================
# Main Benchmark Logic
# ==========================================
def run_comparative_benchmark(
    eval_data_path: str,
    reference_data_path: str,
    model_dir: str,
    output_dir: str = "benchmark_results_comparative",
):
    """
    Runs two passes of inference:
    1. Baseline (Zero-Shot): Predicting without seeing specific protein history.
    2. Fine-Tuned (Few-Shot): Adapting to the protein's history before predicting.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Data ---
    print(f"Loading Eval Data: {eval_data_path}")
    eval_df = pd.read_csv(eval_data_path)

    print(f"Loading Reference (History) Data: {reference_data_path}")
    if os.path.exists(reference_data_path):
        ref_df = pd.read_csv(reference_data_path)
    else:
        raise FileNotFoundError(f"Reference data not found at {reference_data_path}")

    # Initialize Predictor
    predictor = ViscosityPredictorCNP(model_dir)

    shear_rates = [
        "Viscosity_100",
        "Viscosity_1000",
        "Viscosity_10000",
        "Viscosity_100000",
        "Viscosity_15000000",
    ]

    # Container for all results
    all_results = []

    # ==========================================
    # PASS 1: Baseline (No Fine-Tuning)
    # ==========================================
    print("\n--- Running Pass 1: Baseline (Zero-Shot) ---")
    # Reset model to base state (reload or ensuring clear memory)
    predictor = ViscosityPredictorCNP(model_dir)

    # Predict entire eval set at once
    base_preds = predictor.predict(eval_df)

    # Store Baseline Results
    for idx, row in base_preds.iterrows():
        for shear in shear_rates:
            if shear in row and pd.notna(row[shear]):
                all_results.append(
                    {
                        "ID": row.get("ID", idx),
                        "Protein_type": row.get("Protein_type", "Unknown"),
                        "Shear_Rate": shear,
                        "Actual": row[shear],
                        "Predicted": row.get(f"Pred_{shear}", np.nan),
                        "Mode": "Baseline",
                    }
                )

    # ==========================================
    # PASS 2: Fine-Tuned (Adaptive)
    # ==========================================
    print("\n--- Running Pass 2: Fine-Tuned (Adaptive) ---")

    unique_proteins = eval_df["Protein_type"].unique()
    print(f"Found {len(unique_proteins)} unique proteins to adapt to.")

    for protein in unique_proteins:
        # 1. Isolate Eval Data for this protein
        protein_eval_df = eval_df[eval_df["Protein_type"] == protein].copy()

        # 2. Find History for this protein
        protein_history_df = ref_df[ref_df["Protein_type"] == protein].copy()

        # 3. Initialize fresh model for this protein (critical to avoid bleeding context)
        #    (Or just ensure we call learn() which overwrites the context memory)
        predictor = ViscosityPredictorCNP(model_dir)

        if len(protein_history_df) > 0:
            print(
                f"  > Adapting to '{protein}' using {len(protein_history_df)} historical points..."
            )
            predictor.learn(
                protein_history_df,
                fine_tune=True,
                steps=20,
            )
            status = "Fine-Tuned"
        else:
            print(f"  > No history found for '{protein}'. Falling back to baseline.")
            status = "Fine-Tuned (Fallback)"

        # 4. Predict
        ft_preds = predictor.predict(protein_eval_df)

        # 5. Store Results
        for idx, row in ft_preds.iterrows():
            for shear in shear_rates:
                if shear in row and pd.notna(row[shear]):
                    all_results.append(
                        {
                            "ID": row.get("ID", idx),
                            "Protein_type": row.get("Protein_type", "Unknown"),
                            "Shear_Rate": shear,
                            "Actual": row[shear],
                            "Predicted": row.get(f"Pred_{shear}", np.nan),
                            "Mode": status,
                        }
                    )

    # ==========================================
    # Analysis & Reporting
    # ==========================================
    results_df = pd.DataFrame(all_results)

    # 1. Save Raw Data
    raw_path = os.path.join(output_dir, "comparison_raw_data.csv")
    results_df.to_csv(raw_path, index=False)

    # 2. Compute Metrics Grouped by Mode
    print("\n--- Comparative Metrics ---")
    metrics_list = []

    for mode in results_df["Mode"].unique():
        mode_df = results_df[results_df["Mode"] == mode]

        # Overall for this Mode
        r2, mae, rmse = calculate_metrics(mode_df["Actual"], mode_df["Predicted"])
        metrics_list.append(
            {
                "Mode": mode,
                "Shear_Rate": "OVERALL",
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
                "Count": len(mode_df),
            }
        )

        # Per Shear Rate
        for shear in shear_rates:
            shear_df = mode_df[mode_df["Shear_Rate"] == shear]
            if len(shear_df) > 1:
                r2, mae, rmse = calculate_metrics(
                    shear_df["Actual"], shear_df["Predicted"]
                )
                metrics_list.append(
                    {
                        "Mode": mode,
                        "Shear_Rate": shear,
                        "R2": r2,
                        "MAE": mae,
                        "RMSE": rmse,
                        "Count": len(shear_df),
                    }
                )

    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(output_dir, "comparison_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Print Overall Comparison
    print(metrics_df[metrics_df["Shear_Rate"] == "OVERALL"].to_string(index=False))

    # ==========================================
    # Plotting
    # ==========================================
    try:
        plot_comparative_parity(results_df, output_dir)
        plot_improvement_bar(metrics_df, output_dir)
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(f"\nDone. Results saved to {output_dir}")


def plot_comparative_parity(df, output_dir):
    """Generates a color-coded parity plot."""
    plt.figure(figsize=(10, 8))

    # Filter valid
    df = df.dropna(subset=["Actual", "Predicted"])

    # Plot Baseline
    base_df = df[df["Mode"] == "Baseline"]
    plt.scatter(
        base_df["Actual"],
        base_df["Predicted"],
        alpha=0.5,
        label="Baseline",
        color="gray",
        marker="o",
    )

    # Plot Fine-Tuned
    ft_df = df[df["Mode"].str.contains("Fine-Tuned")]
    plt.scatter(
        ft_df["Actual"],
        ft_df["Predicted"],
        alpha=0.6,
        label="Fine-Tuned",
        color="blue",
        marker="^",
    )

    # Diagonal Line
    min_val = min(df["Actual"].min(), df["Predicted"].min())
    max_val = max(df["Actual"].max(), df["Predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Actual Viscosity (cP)")
    plt.ylabel("Predicted Viscosity (cP)")
    plt.title("Impact of Fine-Tuning: Baseline vs. Adapted")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.savefig(os.path.join(output_dir, "parity_comparison.png"), dpi=300)
    plt.close()


def plot_improvement_bar(metrics_df, output_dir):
    """Bar chart showing R2 improvement."""
    # Filter for Overall and specific shear rates
    df = metrics_df.copy()

    # Pivot for plotting
    pivot = df.pivot(index="Shear_Rate", columns="Mode", values="R2")
    if "Fine-Tuned (Fallback)" in pivot.columns:
        pivot["Fine-Tuned"] = pivot["Fine-Tuned"].fillna(0) + pivot[
            "Fine-Tuned (Fallback)"
        ].fillna(0)
        pivot = pivot.drop(columns=["Fine-Tuned (Fallback)"])

    pivot.plot(kind="bar", figsize=(10, 6), color=["gray", "royalblue"])
    plt.ylabel("R² Score")
    plt.title("Model Accuracy Improvement by Fine-Tuning")
    plt.grid(axis="y", alpha=0.3)
    plt.ylim(bottom=0)  # Assuming R2 is positive, adjust if negative
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "r2_improvement.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # 1. The data you want to predict (Eval)
    EVAL_DATA = "data/raw/formulation_data_01092026.csv"

    # 2. The data containing history for those proteins (Training/Reference)
    REF_DATA = "data/processed/formulation_data_augmented.csv"

    MODEL_DIR = "models/experiments/o_net"
    OUTPUT_DIR = "models/experiments/o_net/benchmarks"

    run_comparative_benchmark(EVAL_DATA, REF_DATA, MODEL_DIR, OUTPUT_DIR)
