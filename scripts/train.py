"""
Primary Training and Tuning Script for VisQAI.
Handles Optuna hyperparameter optimization, Cross-Validation, and Ensemble Training.
Compatible with the modular 'src' package.
"""

import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
from optuna.trial import Trial
from sklearn.model_selection import KFold

# --- MODULAR IMPORTS ---
from src.config import (
    BASE_CATEGORICAL,
    BASE_NUMERIC,
    TARGETS,
)
from src.data import DataProcessor
from src.loss import PhysicsInformedLoss, get_physics_masks
from src.management import save_model_checkpoint
from src.models import EnsembleModel, Model
from src.utils import (
    calculate_sample_weights,
    clean,
    log_transform_targets,
    to_tensors,
    validate_data,
)
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- HYPERPARAMETER OPTIMIZATION ---


def objective_cv(
    trial: Trial,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y_log: np.ndarray,
    weights: np.ndarray,
    processor: DataProcessor,
    n_folds: int = 5,
) -> float:
    """
    Objective function for Optuna hyperparameter optimization using cross-validation.
    """
    # 1. Hyperparameter Search Space
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Physics Weights
    lambda_input = trial.suggest_float("lambda_input", 0.01, 2.0)
    lambda_shear = trial.suggest_float("lambda_shear", 0.01, 2.0)

    hidden_sizes = [hidden_size] * n_layers
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_num)):
        # Split data
        Xn_train, Xn_val = X_num[train_idx], X_num[val_idx]
        Xc_train, Xc_val = X_cat[train_idx], X_cat[val_idx]
        y_train, y_val = y_log[train_idx], y_log[val_idx]
        w_train, w_val = weights[train_idx], weights[val_idx]

        # Convert to tensors
        Xn_tr_t, Xc_tr_t, y_tr_t, w_tr_t = to_tensors(
            Xn_train, Xc_train, y_train, w_train
        )
        Xn_val_t, Xc_val_t, y_val_t, w_val_t = to_tensors(Xn_val, Xc_val, y_val, w_val)

        # Create model
        model = Model(
            cat_maps=processor.cat_maps,
            numeric_dim=X_num.shape[1],
            out_dim=len(TARGETS),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            split_indices=processor.split_indices,
        )

        # Setup training
        criterion = PhysicsInformedLoss(
            lambda_shear=lambda_shear,
            lambda_input=lambda_input,
            numeric_cols=BASE_NUMERIC,
        )
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        best_fold_loss = float("inf")

        # Training loop
        for epoch in range(100):
            model.train()
            indices = torch.randperm(len(Xn_tr_t))

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i : i + batch_size]

                # Prepare tensors with gradient tracking for Physics Loss
                batch_X_num = Xn_tr_t[batch_idx].clone().detach().requires_grad_(True)
                batch_X_cat = Xc_tr_t[batch_idx]
                batch_y = y_tr_t[batch_idx]
                batch_w = w_tr_t[batch_idx]

                optimizer.zero_grad()

                # Forward pass
                pred = model(batch_X_num, batch_X_cat)

                # Generate physics masks
                batch_masks = get_physics_masks(batch_X_cat, batch_X_num, processor)

                # Calculate loss
                loss = criterion(
                    pred=pred,
                    target=batch_y,
                    inputs_num=batch_X_num,
                    masks=batch_masks,
                    weights=batch_w,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            # Validation
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(Xn_val_t, Xc_val_t)
                    val_loss = torch.nn.MSELoss()(val_pred, y_val_t).item()
                    if val_loss < best_fold_loss:
                        best_fold_loss = val_loss

        fold_losses.append(best_fold_loss)

    return np.mean(fold_losses)


def run_tuning(
    data_path: str,
    n_trials: int = 50,
    n_folds: int = 5,
    output_dir: str = "experiments",
) -> Tuple[Dict, DataProcessor]:
    """
    Run hyperparameter tuning using Optuna.
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df_clean = clean(df, BASE_NUMERIC, BASE_CATEGORICAL, TARGETS)

    # Fit processor and prepare data
    processor = DataProcessor()
    X_num, X_cat = processor.fit_transform(df_clean)
    y_raw = df_clean[TARGETS].values
    y_log = log_transform_targets(y_raw)
    weights = calculate_sample_weights(y_raw)

    validate_data(X_num, X_cat, y_log, "training")

    # Run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective_cv(
            trial, X_num, X_cat, y_log, weights, processor, n_folds
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save processor in the dated directory
    save_path = os.path.join(output_dir, "processor.pkl")
    processor.save(save_path)
    print(f"\nSaved processor to: {save_path}")

    return study.best_params, processor


# --- ENSEMBLE TRAINING ---


def train_final_ensemble(
    data_path: str,
    best_params: Dict,
    processor: Optional[DataProcessor] = None,
    n_models: int = 5,
    output_dir: str = "experiments",
) -> EnsembleModel:
    """
    Train an ensemble of models using the best found hyperparameters.
    """
    print(f"\n{'='*70}")
    print(f"Training Ensemble of {n_models} models")
    print(f"{'='*70}\n")

    df = pd.read_csv(data_path)
    df_clean = clean(df, BASE_NUMERIC, BASE_CATEGORICAL, TARGETS)

    if processor is None:
        processor = DataProcessor()
        X_num, X_cat = processor.fit_transform(df_clean)
    else:
        # Use existing processor logic
        X_num, X_cat = processor.transform(df_clean)

    # Save the processor to the output directory if it wasn't saved during tuning
    # or just to ensure the directory is self-contained.
    proc_path = os.path.join(output_dir, "processor.pkl")
    processor.save(proc_path)
    print(f"Ensured processor is saved at: {proc_path}")

    y_raw = df_clean[TARGETS].values
    y_log = log_transform_targets(y_raw)
    weights = calculate_sample_weights(y_raw)

    X_num_t, X_cat_t, y_t, w_t = to_tensors(X_num, X_cat, y_log, weights)

    trained_models = []

    for i in range(n_models):
        print(f"\nTraining Model {i+1}/{n_models}")
        print("-" * 50)

        # Seeding for ensemble diversity
        torch.manual_seed(i + 42)

        model = Model(
            cat_maps=processor.cat_maps,
            numeric_dim=X_num.shape[1],
            out_dim=len(TARGETS),
            hidden_sizes=[best_params["hidden_size"]] * best_params["n_layers"],
            dropout=best_params["dropout"],
            split_indices=processor.split_indices,
        )

        criterion = PhysicsInformedLoss(
            lambda_shear=best_params["lambda_shear"],
            lambda_input=best_params["lambda_input"],
            numeric_cols=BASE_NUMERIC,
        )
        optimizer = optim.Adam(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )

        n_epochs = 250
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        # Stochastic Weight Averaging (SWA)
        swa_start = int(n_epochs * 0.75)
        swa_weights = None
        swa_n = 0

        model.train()
        for epoch in range(n_epochs):
            indices = torch.randperm(len(X_num_t))

            for j in range(0, len(indices), best_params["batch_size"]):
                batch_idx = indices[j : j + best_params["batch_size"]]

                batch_X_num = X_num_t[batch_idx].clone().detach().requires_grad_(True)
                batch_X_cat = X_cat_t[batch_idx]
                batch_y = y_t[batch_idx]
                batch_w = w_t[batch_idx]

                optimizer.zero_grad()
                pred = model(batch_X_num, batch_X_cat)

                # Compute Masks & Loss
                batch_masks = get_physics_masks(batch_X_cat, batch_X_num, processor)
                loss = criterion(
                    pred=pred,
                    target=batch_y,
                    inputs_num=batch_X_num,
                    masks=batch_masks,
                    weights=batch_w,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            # SWA Accumulation
            if epoch >= swa_start:
                current = {k: v.clone().detach() for k, v in model.state_dict().items()}
                if swa_weights is None:
                    swa_weights = current
                    swa_n = 1
                else:
                    for k in swa_weights:
                        swa_weights[k] += current[k]
                    swa_n += 1

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} complete")

        # Apply SWA
        if swa_weights:
            for k in swa_weights:
                swa_weights[k] /= swa_n
            model.load_state_dict(swa_weights)
            print(f"  Applied SWA from {swa_n} epochs")

        trained_models.append(model)

        # Save Individual Model Checkpoint with clean numbering
        model_name = f"model_{i}.pt"
        model_path = os.path.join(output_dir, model_name)
        save_model_checkpoint(model, processor, best_params, model_path)
        print(f"Saved checkpoint: {model_name}")

    print(f"\n{'='*70}")
    print("Ensemble Training Complete")
    print(f"{'='*70}\n")

    return EnsembleModel(trained_models)


# --- MAIN ENTRY POINT ---


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update paths as needed for your environment
    DATA_PATH = r"data/processed/formulation_data_augmented.csv"
    EVAL_PATH = r"data/raw/formulation_data_01052026.csv"

    # Tuning Settings
    N_TRIALS = 15
    N_FOLDS = 5
    N_MODELS = 5  # Size of final ensemble

    # Flags
    DO_TUNING = True
    DO_TRAINING = True

    # --- DIRECTORY SETUP ---
    # Create a dated directory for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPERIMENT_DIR = os.path.join("models", "experiments", timestamp)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print(f"Experiment initialized. Output Directory: {EXPERIMENT_DIR}")

    best_params = None
    processor = None

    # 1. Hyperparameter Tuning
    if DO_TUNING:
        print("\n" + "=" * 70)
        print("PHASE 1: HYPERPARAMETER TUNING")
        print("=" * 70)

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Training data not found at: {DATA_PATH}")

        # Pass the directory explicitly
        best_params, processor = run_tuning(
            DATA_PATH,
            n_trials=N_TRIALS,
            n_folds=N_FOLDS,
            output_dir=EXPERIMENT_DIR,
        )

    # 2. Ensemble Training
    if DO_TRAINING:
        print("\n" + "=" * 70)
        print("PHASE 2: ENSEMBLE TRAINING")
        print("=" * 70)

        if best_params is None:
            print("Warning: Tuning skipped. Using default parameters.")
            best_params = {
                "lr": 1e-4,
                "n_layers": 3,
                "hidden_size": 256,
                "dropout": 0.1,
                "batch_size": 32,
                "weight_decay": 1e-5,
                "lambda_input": 0.1,
                "lambda_shear": 1.0,
            }

        final_model = train_final_ensemble(
            DATA_PATH,
            best_params,
            n_models=N_MODELS,
            processor=processor,
            output_dir=EXPERIMENT_DIR,
        )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print(f"All artifacts saved to: {EXPERIMENT_DIR}")
    print("=" * 70)
