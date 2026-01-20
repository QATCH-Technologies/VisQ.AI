import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# Import your class - adjust import path as necessary
from src.inference import ViscosityPredictor


def evaluate_model(model_path, data_path):
    # 1. Load Data and Model
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Loading model from {model_path}...")
    vp = ViscosityPredictor(model_path, is_ensemble=True)

    # 2. Identify Target Columns
    # Filter columns that start with 'Viscosity_'
    target_cols = [c for c in df.columns if c.startswith("Viscosity_")]

    # Validation: Ensure we found exactly 5 target columns
    if len(target_cols) != 5:
        print(
            f"Warning: Found {len(target_cols)} target columns, but model outputs 5 values."
        )
        print(f"Columns found: {target_cols}")
        # You might want to sort them to ensure alignment with model output indices
        target_cols.sort()
        print(f"Using sorted columns: {target_cols}")

    # 3. Generate Predictions
    print("Generating predictions...")
    preds = vp.predict(df)  # Assumes preds is (N_samples, 5)

    # 4. Extract Ground Truth
    ground_truth = df[target_cols].values

    # 5. Calculate Metrics per Shear Rate
    results = []

    print("\n--- Evaluation Results ---")
    for i, col_name in enumerate(target_cols):
        # Extract single column for this shear rate
        y_true = ground_truth[:, i]
        y_pred = preds[:, i]

        # Calculate R2
        r2 = r2_score(y_true, y_pred)

        # Calculate MAPE
        # Note: distinct handling if y_true contains zeros to avoid infinity
        mape = mean_absolute_percentage_error(y_true, y_pred)

        results.append({"Shear Rate / Column": col_name, "R2": r2, "MAPE": mape})

        print(f"{col_name}: R2 = {r2:.4f}, MAPE = {mape:.4f}")

    # 6. Overall Aggregates (Optional)
    avg_r2 = np.mean([r["R2"] for r in results])
    avg_mape = np.mean([r["MAPE"] for r in results])

    print("-" * 30)
    print(f"Average R2: {avg_r2:.4f}")
    print(f"Average MAPE: {avg_mape:.4f}")


if __name__ == "__main__":
    # Adjust paths as needed
    MODEL_PATH = r"models\experiments\20260120_152300"
    DATA_PATH = "data/raw/formulation_data_12292025.csv"

    evaluate_model(MODEL_PATH, DATA_PATH)
