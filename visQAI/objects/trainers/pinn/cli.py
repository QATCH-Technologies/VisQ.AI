#!/usr/bin/env python3
"""
run_model.py

Load a pretrained ensemble of PINN models, apply their saved preprocessors via DataLoader,
and run ensemble predictions on input data.
Data paths are specified as constants below.
"""
import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
from pinn_domain import DataLoader
from tensorflow.keras.models import load_model

# === CONSTANT PATHS ===
ENSEMBLE_DIR = "ensemble_pinn"
INPUT_CSV = r"C:\Users\QATCH\dev\VisQ.AI\visQAI\objects\trainers\pinn\input_data.csv"
OUTPUT_CSV = "ensemble_predictions.csv"


def load_ensemble(ensemble_dir: str):
    """
    Discover and load all models and their corresponding preprocessors in the ensemble directory.
    Assumes model files named pinn_<id>.h5 and transformers named transformer_<id>.joblib
    where <id> is an integer.

    Returns:
        models: List of loaded Keras models
        preprocessors: List of loaded sklearn transformers
    """
    pattern = os.path.join(ensemble_dir, "pinn_*.h5")
    model_paths = glob.glob(pattern)
    if not model_paths:
        raise FileNotFoundError(f"No model files found in {ensemble_dir}")

    def _extract_id(path):
        m = re.search(r'pinn_(\d+)\.h5$', path)
        return int(m.group(1)) if m else float('inf')

    model_paths = sorted(model_paths, key=_extract_id)

    models = []
    preprocessors = []
    for model_path in model_paths:
        model_id = _extract_id(model_path)
        print(f"Loading model {model_id} from {model_path}")
        model = load_model(model_path, compile=False)
        models.append(model)

        transformer_path = os.path.join(
            ensemble_dir, f"transformer_{model_id}.joblib")
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(
                f"Transformer file not found: {transformer_path}")
        print(f"Loading preprocessor {model_id} from {transformer_path}")
        preprocessor = joblib.load(transformer_path)
        preprocessors.append(preprocessor)

    return models, preprocessors


def ensemble_predict(models, preprocessors, loader: DataLoader) -> np.ndarray:
    """
    Given lists of models and preprocessors and a DataLoader pre-loaded with raw data,
    assign each preprocessor to the loader, transform the features, predict, and then average.

    Returns:
        ensemble_preds: array shape (n_samples, n_outputs)
    """
    preds_list = []
    for model, pre in zip(models, preprocessors):
        # assign this model's preprocessor
        loader._preprocessor = pre
        # get features cleaned & transformed
        X = loader.get_processed_features()
        preds = model.predict(X)
        preds_list.append(preds)

    # shape (n_models, n_samples, n_outputs)
    stacked = np.stack(preds_list, axis=0)
    ensemble_preds = np.mean(stacked, axis=0)
    return ensemble_preds


if __name__ == "__main__":
    # Verify input exists
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input data file not found: {INPUT_CSV}")

    # Initialize DataLoader and load + clean raw features
    dl = DataLoader(csv_path=INPUT_CSV)
    dl.load()

    # Load ensemble of models + preprocessors
    models, preprocessors = load_ensemble(ENSEMBLE_DIR)

    # Run ensemble prediction
    print("Running ensemble predictions...")
    preds = ensemble_predict(models, preprocessors, dl)

    # Save predictions
    n_out = preds.shape[1] if preds.ndim > 1 else 1
    columns = [f"output_{i}" for i in range(n_out)]
    pd.DataFrame(preds, columns=columns).to_csv(OUTPUT_CSV, index=False)

    print(f"Saved ensemble predictions to {OUTPUT_CSV}")
