import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import KFold

# Add current directory to path
sys.path.append(os.getcwd())
from inference_o_net import ViscosityPredictorCNP


# --- METRICS HELPER ---
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Filter for finite and POSITIVE values (since we need log)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    y_true_c, y_pred_c = y_true[mask], y_pred[mask]

    if len(y_true_c) < 2:
        return np.nan, np.nan, np.nan

    # --- CHANGE: Calculate R2 in Log10 Space ---
    # This stabilizes the metric for low-viscosity proteins where variance is small.
    # It effectively measures: "How well did we predict the order of magnitude?"
    r2 = metrics.r2_score(np.log10(y_true_c), np.log10(y_pred_c))

    # --- KEEP: RMSE and MAE in Linear Space ---
    # We still want these in cP (physical units) for interpretability.
    mae = metrics.mean_absolute_error(y_true_c, y_pred_c)
    rmse = np.sqrt(metrics.mean_squared_error(y_true_c, y_pred_c))

    return r2, mae, rmse


# --- MAIN BENCHMARK ---
def run_cv_learning_curve(
    data_path: str,
    model_dir: str,
    target_protein: str = "Trastuzumab",
    output_dir: str = "models/experiments/o_net_no_trast/benchmarks",
    n_splits: int = 4,
    n_repeats: int = 1,
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    full_df = pd.read_csv(data_path)
    target_df = full_df[
        full_df["Protein_type"].str.lower() == target_protein.lower()
    ].copy()
    target_df = target_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total samples for {target_protein}: {len(target_df)}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_log = []
    raw_predictions_log = []  # <--- NEW: Store raw dataframes here

    shear_rates = [
        "Viscosity_100",
        "Viscosity_1000",
        "Viscosity_10000",
        "Viscosity_100000",
        "Viscosity_15000000",
    ]

    # 2. Outer Loop: Cross-Validation Folds
    fold_idx = 0
    for train_indices, test_indices in kf.split(target_df):
        fold_idx += 1
        print(f"--- Fold {fold_idx}/{n_splits} ---")

        train_pool_df = target_df.iloc[train_indices]
        fixed_test_df = target_df.iloc[test_indices]
        max_train_size = len(train_pool_df)

        # 3. Inner Loop: Increasing Training Size
        training_sizes = list(range(0, max_train_size + 1))

        for k in training_sizes:
            # Sampling redundancy for stability
            current_repeats = n_repeats if (0 < k < max_train_size) else 1

            for r in range(current_repeats):
                # Sample k training points
                if k == 0:
                    current_train_df = pd.DataFrame()
                elif k == max_train_size:
                    current_train_df = train_pool_df
                else:
                    current_train_df = train_pool_df.sample(n=k, random_state=r + k)

                # Initialize & Train
                predictor = ViscosityPredictorCNP(model_dir)
                if k > 0:
                    # fine_tune=False is usually preferred for learning curves
                    # to test pure ICL, but can be True if testing adaptation.
                    predictor.learn(current_train_df, fine_tune=True)

                # --- EVALUATION ---
                eval_sets = {"Unseen": fixed_test_df, "All": target_df}

                for set_name, eval_df in eval_sets.items():
                    preds = predictor.predict(eval_df)

                    # --- A. Metrics Calculation (Existing) ---
                    y_true_all, y_pred_all = [], []
                    valid_shears = [s for s in shear_rates if s in preds.columns]

                    for shear in valid_shears:
                        y_true_all.extend(eval_df[shear].values)
                        y_pred_all.extend(preds[f"Pred_{shear}"].values)

                    r2, mae, rmse = calculate_metrics(y_true_all, y_pred_all)

                    metrics_log.append(
                        {
                            "Fold": fold_idx,
                            "n_train_samples": k,
                            "Set": set_name,
                            "R2": r2,
                            "RMSE": rmse,
                            "MAE": mae,
                        }
                    )

                    # --- B. Raw Prediction Logging (NEW) ---
                    if valid_shears:
                        # Select ID, Actuals, and Predictions
                        # Ensure ID exists; if not, use index
                        temp_df = preds.copy()
                        if "ID" not in temp_df.columns:
                            temp_df["ID"] = temp_df.index

                        pred_cols = [f"Pred_{s}" for s in valid_shears]
                        cols_to_keep = ["ID"] + valid_shears + pred_cols
                        subset = temp_df[cols_to_keep]

                        # Melt Actuals
                        melt_act = subset.melt(
                            id_vars=["ID"],
                            value_vars=valid_shears,
                            var_name="Shear_Rate",
                            value_name="Actual",
                        )

                        # Melt Predictions
                        melt_pred = subset.melt(
                            id_vars=["ID"],
                            value_vars=pred_cols,
                            var_name="Pred_Key",
                            value_name="Predicted",
                        )
                        # Clean Pred_Key to match Shear_Rate (remove 'Pred_')
                        melt_pred["Shear_Rate"] = melt_pred["Pred_Key"].str.replace(
                            "Pred_", "", regex=False
                        )

                        # Merge on ID and Shear_Rate
                        raw_merged = pd.merge(
                            melt_act,
                            melt_pred[["ID", "Shear_Rate", "Predicted"]],
                            on=["ID", "Shear_Rate"],
                        )

                        # Add Metadata
                        raw_merged["Fold"] = fold_idx
                        raw_merged["n_train_samples"] = k
                        raw_merged["Set"] = set_name

                        raw_predictions_log.append(raw_merged)

    # 4. Aggregation & Saving

    # Save Metrics
    df_res = pd.DataFrame(metrics_log)
    agg_res = (
        df_res.groupby(["n_train_samples", "Set"])[["R2", "RMSE", "MAE"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    new_cols = []
    for col in agg_res.columns.values:
        if col[0] in ["n_train_samples", "Set"]:
            new_cols.append(col[0])
        else:
            new_cols.append(f"{col[0]}_{col[1]}")
    agg_res.columns = new_cols

    csv_path = os.path.join(output_dir, "cv_learning_curve_metrics.csv")
    agg_res.to_csv(csv_path, index=False)
    print(f"Saved aggregated metrics to {csv_path}")

    # Save Raw Predictions
    if raw_predictions_log:
        df_raw_all = pd.concat(raw_predictions_log, ignore_index=True)
        raw_csv_path = os.path.join(output_dir, "cv_learning_curve_raw_predictions.csv")
        df_raw_all.to_csv(raw_csv_path, index=False)
        print(f"Saved raw predictions to {raw_csv_path}")

    # 5. Plotting
    plot_cv_results_separated(agg_res, target_protein, output_dir)


def plot_cv_results_separated(df, protein, output_dir):
    sns.set_style("whitegrid")

    # Helper to plot one metric for one set
    def create_single_plot(
        subset_df, metric_col, std_col, color, title, filename, ylabel, ylim=None
    ):
        plt.figure(figsize=(10, 6))

        plt.plot(
            subset_df["n_train_samples"],
            subset_df[metric_col],
            "o-",
            color=color,
            label=f"{title} (CV Mean)",
            linewidth=2,
        )

        plt.fill_between(
            subset_df["n_train_samples"],
            subset_df[metric_col] - subset_df[std_col],
            subset_df[metric_col] + subset_df[std_col],
            alpha=0.2,
            color=color,
        )

        if "R2" in metric_col:
            plt.axhline(0, color="black", lw=1, ls="--")
            # Adjust limits if needed, e.g., for Log R2
            # plt.ylim(-1.0, 1.05)

        if ylim:
            plt.ylim(ylim)

        plt.title(f"Data Efficiency: {protein} - {title}")
        plt.xlabel("Number of Training Samples")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved plot: {filename}")

    # --- Generate Plots ---

    # 1. Unseen - R2
    create_single_plot(
        df[df["Set"] == "Unseen"],
        "R2_mean",
        "R2_std",
        "tab:orange",
        "Unseen Data R²",
        "unseen_r2.png",
        "R² Score",
    )

    # 2. All - R2
    create_single_plot(
        df[df["Set"] == "All"],
        "R2_mean",
        "R2_std",
        "tab:blue",
        "All Data R² (Memory)",
        "all_r2.png",
        "R² Score",
    )

    # 3. Unseen - RMSE
    create_single_plot(
        df[df["Set"] == "Unseen"],
        "RMSE_mean",
        "RMSE_std",
        "tab:orange",
        "Unseen Data RMSE",
        "unseen_rmse.png",
        "RMSE (Linear Scale)",
    )

    # 4. All - RMSE
    create_single_plot(
        df[df["Set"] == "All"],
        "RMSE_mean",
        "RMSE_std",
        "tab:blue",
        "All Data RMSE (Memory)",
        "all_rmse.png",
        "RMSE (Linear Scale)",
    )

    # 5. Unseen - MAE (NEW)
    create_single_plot(
        df[df["Set"] == "Unseen"],
        "MAE_mean",
        "MAE_std",
        "tab:orange",
        "Unseen Data MAE",
        "unseen_mae.png",
        "MAE (cP)",
    )

    # 6. All - MAE (NEW)
    create_single_plot(
        df[df["Set"] == "All"],
        "MAE_mean",
        "MAE_std",
        "tab:blue",
        "All Data MAE (Memory)",
        "all_mae.png",
        "MAE (cP)",
    )


if __name__ == "__main__":
    # --- CONFIG ---
    DATA_PATH = "data/raw/formulation_data_01092026.csv"
    MODEL_DIR = "models/experiments/o_net_no_eta"

    run_cv_learning_curve(
        DATA_PATH,
        MODEL_DIR,
        target_protein="Etanercept",
        output_dir="models/experiments/o_net_no_eta/benchmarks",
        n_splits=4,
    )
