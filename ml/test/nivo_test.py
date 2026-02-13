import logging
import os

import numpy as np
import pandas as pd
from inference_o_net import ViscosityPredictorCNP
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# Configuration
# ==========================================
MODEL_DIR = "models/experiments/o_net"
SEEN_DATA_PATH = "data/processed/nivo_seen_data.csv"
UNSEEN_DATA_PATH = "data/processed/nivo_unseen_data.csv"
OUTPUT_FILE = "data/processed/nivo_eval_uncertainty.xlsx"

# Shear columns expected by the model (matches inference_o_net.py)
SHEAR_KEYS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("EvalScript")


def load_data():
    """Safely loads the datasets."""
    if not os.path.exists(SEEN_DATA_PATH) or not os.path.exists(UNSEEN_DATA_PATH):
        raise FileNotFoundError("Could not find seen/unseen data CSVs.")
    return pd.read_csv(SEEN_DATA_PATH), pd.read_csv(UNSEEN_DATA_PATH)


def format_predictions(df, mean_pred, stats):
    """
    Unpacks the flat arrays from predict_with_uncertainty into DataFrame columns.
    The model outputs flat arrays ordered by Sample -> Shear Rate.
    """
    results = df.copy()
    n_shears = len(SHEAR_KEYS)

    # Containers for new columns
    data_map = {f"Pred_{k}": [] for k in SHEAR_KEYS}
    data_map.update({f"LowerCI_{k}": [] for k in SHEAR_KEYS})
    data_map.update({f"UpperCI_{k}": [] for k in SHEAR_KEYS})

    lower_ci = stats["lower_ci"]
    upper_ci = stats["upper_ci"]

    # Loop through samples and distribute the flat data
    for i in range(len(df)):
        start_idx = i * n_shears

        # Slice the segment for this sample
        sample_means = mean_pred[start_idx : start_idx + n_shears]
        sample_lowers = lower_ci[start_idx : start_idx + n_shears]
        sample_uppers = upper_ci[start_idx : start_idx + n_shears]

        for j, key in enumerate(SHEAR_KEYS):
            data_map[f"Pred_{key}"].append(sample_means[j])
            data_map[f"LowerCI_{key}"].append(sample_lowers[j])
            data_map[f"UpperCI_{key}"].append(sample_uppers[j])

    # Assign new columns to the dataframe
    for col_name, values in data_map.items():
        results[col_name] = values

    return results


def calculate_metrics(df):
    """
    Calculates MAE and RMSE for each shear rate where ground truth exists.
    """
    metrics = []

    for key in SHEAR_KEYS:
        # Check if ground truth column exists
        if key not in df.columns:
            logger.warning(f"Ground truth column {key} missing. Skipping metrics.")
            continue

        pred_key = f"Pred_{key}"

        # Filter rows where both Ground Truth and Prediction are valid (not NaN)
        valid_df = df.dropna(subset=[key, pred_key])

        if valid_df.empty:
            metrics.append(
                {"Shear_Rate": key, "MAE": np.nan, "RMSE": np.nan, "Count": 0}
            )
            continue

        y_true = valid_df[key]
        y_pred = valid_df[pred_key]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        metrics.append(
            {"Shear_Rate": key, "MAE": mae, "RMSE": rmse, "Count": len(valid_df)}
        )

    return pd.DataFrame(metrics)


def run_zero_shot_test(seen_df, unseen_df):
    """
    Test 1: Zero-Shot (No Learning) on Combined Data.
    """
    logger.info("--- Starting Test 1: Zero-Shot (No Fine-Tuning) ---")

    predictor = ViscosityPredictorCNP(model_dir=MODEL_DIR)
    combined_df = pd.concat([seen_df, unseen_df], ignore_index=True)

    # Predict with Uncertainty
    mean_pred, stats = predictor.predict_with_uncertainty(combined_df, n_samples=20)

    # Format Results
    results = format_predictions(combined_df, mean_pred, stats)
    results.insert(0, "Evaluation_Mode", "Zero_Shot")

    # Calculate Metrics
    metrics = calculate_metrics(results)

    return results, metrics


def run_fine_tuning_test(seen_df, unseen_df):
    """
    Test 2: Fine-Tuned on Seen, Predict on Unseen.
    """
    logger.info("--- Starting Test 2: Fine-Tuned (With Learning) ---")

    predictor = ViscosityPredictorCNP(model_dir=MODEL_DIR)

    # 1. Learn
    logger.info(f"Fine-tuning on {len(seen_df)} samples...")
    predictor.learn(seen_df, steps=50, lr=1e-3)

    # 2. Predict
    logger.info(f"Predicting on {len(unseen_df)} unseen samples...")
    mean_pred, stats = predictor.predict_with_uncertainty(unseen_df, n_samples=20)

    # Format Results
    results = format_predictions(unseen_df, mean_pred, stats)
    results.insert(0, "Evaluation_Mode", "Fine_Tuned")

    # Calculate Metrics
    metrics = calculate_metrics(results)

    return results, metrics


def main():
    try:
        # 1. Load Data
        seen_df, unseen_df = load_data()

        # 2. Run Tests
        zs_results, zs_metrics = run_zero_shot_test(seen_df, unseen_df)
        ft_results, ft_metrics = run_fine_tuning_test(seen_df, unseen_df)

        # 3. Save to Excel
        logger.info(f"Saving results to {OUTPUT_FILE}...")

        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            # Zero Shot
            zs_results.to_excel(writer, sheet_name="Zero_Shot_Preds", index=False)
            zs_metrics.to_excel(writer, sheet_name="Zero_Shot_Metrics", index=False)

            # Fine Tuned
            ft_results.to_excel(writer, sheet_name="Fine_Tuned_Preds", index=False)
            ft_metrics.to_excel(writer, sheet_name="Fine_Tuned_Metrics", index=False)

        logger.info("Evaluation complete. Success.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
