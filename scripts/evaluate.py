import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# --- MODULAR IMPORTS ---
# Ensure your environment can see 'src'
from src.config import BASE_CATEGORICAL, BASE_NUMERIC, TARGETS
from src.data import DataProcessor
from src.management import load_model_checkpoint
from src.models import EnsembleModel
from src.utils import clean, inverse_log_transform

# --- CONFIGURATION ---
# Update this to the specific experiment timestamp folder you want to evaluate
EXPERIMENT_DIR = r"models/experiments/20260120_090347"
DATA_PATH = r"data/processed/formulation_data_augmented.csv"


def extract_shear_rate(column_name: str) -> float:
    """
    Extracts the numeric shear rate from the target column name.
    Example: 'Viscosity_0.1' -> 0.1
    """
    match = re.search(r"(\d+\.?\d*)$", column_name)
    if match:
        return float(match.group(1))
    return -1.0


def check_model_health(model: torch.nn.Module) -> bool:
    """
    Checks if a model's weights are valid (no NaNs or Infs).
    Returns True if healthy, False if corrupted.
    """
    for param in model.parameters():
        if not torch.isfinite(param).all():
            return False
    return True


def load_ensemble(
    experiment_dir: str, device: str
) -> Tuple[EnsembleModel, DataProcessor]:
    """
    Loads all valid model checkpoints from the directory using the management module.
    """
    model_files = sorted(glob.glob(os.path.join(experiment_dir, "model_*.pt")))
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt files found in {experiment_dir}")

    models = []
    processor = None
    print(f"Found {len(model_files)} checkpoints in {experiment_dir}...")

    for path in model_files:
        filename = os.path.basename(path)
        try:
            # --- REQUIREMENT 1: Use management module to load ---
            # This handles the reconstruction of the processor, scaler, and architecture
            model, loaded_proc, _ = load_model_checkpoint(path, device=device)

            # Health check to ensure we don't use dead models
            if check_model_health(model):
                model.eval()
                models.append(model)

                # Keep the processor from the first valid model
                # (All models in an ensemble share the same processor logic)
                if processor is None:
                    processor = loaded_proc
            else:
                print(f"  [WARN] Skipping {filename}: Model weights contain NaNs.")

        except Exception as e:
            print(f"  [ERR] Failed to load {filename}: {e}")

    if not models:
        raise RuntimeError("No valid models could be loaded.")

    print(f"Successfully loaded {len(models)} valid models.")

    # We return an EnsembleModel wrapper (assuming it's available in src.models)
    return EnsembleModel(models), processor


if __name__ == "__main__":
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load Data
    print(f"Loading evaluation data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Use clean from utils.py to ensure consistency with training
    df_clean = clean(df, BASE_NUMERIC, BASE_CATEGORICAL, TARGETS)

    # 3. Load Model & Processor
    ensemble, processor = load_ensemble(EXPERIMENT_DIR, device=device)

    # 4. Prepare Features
    # Transform using the loaded processor (ensures scaling matches training)
    X_num, X_cat = processor.transform(df_clean)

    X_num_t = torch.tensor(X_num, dtype=torch.float32).to(device)
    X_cat_t = torch.tensor(X_cat, dtype=torch.long).to(device)

    # 5. Inference
    print("Running inference...")
    ensemble.to(device)

    with torch.no_grad():
        # Model returns predictions in Log10 space
        preds_log = ensemble(X_num_t, X_cat_t).cpu().numpy()

        # --- REQUIREMENT 2: Inverse Transform using utils module ---
        # Converts Log10 values back to real physical units (e.g., cP or Pa·s)
        preds_real = inverse_log_transform(preds_log)

    # 6. Evaluation
    y_true = df_clean[TARGETS].values

    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS BY SHEAR RATE")
    print("=" * 80)

    metrics = []

    for i, target_name in enumerate(TARGETS):
        y_t = y_true[:, i]
        y_p = preds_real[:, i]

        # Safety check for NaNs in output (in case of runtime instability)
        if np.isnan(y_p).any():
            r2 = np.nan
            mape = np.nan
        else:
            r2 = r2_score(y_t, y_p)
            mape = mean_absolute_percentage_error(y_t, y_p)

        shear_val = extract_shear_rate(target_name)

        metrics.append(
            {
                "Target Column": target_name,
                "Shear Rate": shear_val if shear_val != -1.0 else np.inf,
                "R2": r2,
                "MAPE": mape,
            }
        )

    # Sort by Shear Rate for readable output
    df_metrics = pd.DataFrame(metrics).sort_values("Shear Rate")

    # Display
    cols = ["Target Column", "Shear Rate", "R2", "MAPE"]
    # Replace infinite shear rate with "N/A" for display if regex failed
    display_df = df_metrics.copy()
    display_df["Shear Rate"] = display_df["Shear Rate"].replace(np.inf, "N/A")

    print(
        display_df[cols].to_string(
            index=False, float_format=lambda x: "{:.4f}".format(x)
        )
    )

    # Averages
    avg_r2 = df_metrics["R2"].mean()
    avg_mape = df_metrics["MAPE"].mean()
    print("-" * 80)
    print(f"AVERAGE        |            | {avg_r2:.4f} | {avg_mape:.2%}")
    print("-" * 80)

    # Save Results
    save_path = os.path.join(EXPERIMENT_DIR, "evaluation_metrics.csv")
    df_metrics.to_csv(save_path, index=False)
    print(f"\nDetailed metrics saved to: {save_path}")
