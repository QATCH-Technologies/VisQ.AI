"""
Primary Training and Tuning Script for VisQAI.
Handles Optuna hyperparameter optimization, Cross-Validation, and Ensemble Training.
Compatible with the modular 'src' package.
"""

import copy  # Added for deepcopying model weights
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim

# --- MODULAR IMPORTS ---
from config import (
    BASE_CATEGORICAL,
    BASE_NUMERIC,
    TARGETS,
)
from loss import PhysicsInformedLoss, get_physics_masks
from management import save_model_checkpoint
from optuna.trial import Trial
from sklearn.model_selection import KFold, train_test_split  # Added train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import (
    calculate_sample_weights,
    clean,
    log_transform_targets,
    to_tensors,
    validate_data,
)

from data import DataProcessor
from models import EnsembleModel, Model


def check_model_health(model: torch.nn.Module) -> bool:
    """Returns True if model weights are finite (healthy), False if NaN/Inf."""
    for param in model.parameters():
        if not torch.isfinite(param).all():
            return False
    return True


def objective_cv(
    trial: Trial,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y_log: np.ndarray,
    weights: np.ndarray,
    processor: DataProcessor,
    n_folds: int = 5,
) -> float:
    # ... [Same as original] ...
    # (Kept concise for brevity; no changes needed in objective_cv)
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    lambda_input = trial.suggest_float("lambda_input", 0.0, 0.1)
    lambda_shear = trial.suggest_float("lambda_shear", 0.0, 0.5)

    hidden_sizes = [hidden_size] * n_layers
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_num)):
        Xn_train, Xn_val = X_num[train_idx], X_num[val_idx]
        Xc_train, Xc_val = X_cat[train_idx], X_cat[val_idx]
        y_train, y_val = y_log[train_idx], y_log[val_idx]
        w_train, w_val = weights[train_idx], weights[val_idx]

        Xn_tr_t, Xc_tr_t, y_tr_t, w_tr_t = to_tensors(
            Xn_train, Xc_train, y_train, w_train
        )
        Xn_val_t, Xc_val_t, y_val_t, w_val_t = to_tensors(Xn_val, Xc_val, y_val, w_val)

        model = Model(
            cat_maps=processor.cat_maps,
            numeric_dim=X_num.shape[1],
            out_dim=len(TARGETS),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            split_indices=processor.split_indices,
        )

        criterion = PhysicsInformedLoss(
            lambda_shear=lambda_shear,
            lambda_input=lambda_input,
            numeric_cols=BASE_NUMERIC,
        )
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        best_fold_loss = float("inf")

        for epoch in range(50):
            model.train()
            indices = torch.randperm(len(Xn_tr_t))

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i : i + batch_size]
                batch_X_num = Xn_tr_t[batch_idx].clone().detach().requires_grad_(True)
                batch_X_cat = Xc_tr_t[batch_idx]
                batch_y = y_tr_t[batch_idx]
                batch_w = w_tr_t[batch_idx]

                optimizer.zero_grad()
                pred = model(batch_X_num, batch_X_cat)

                # 1. Check Prediction NaNs
                if torch.isnan(pred).any():
                    print(f"  [Warn] NaN in prediction at Epoch {epoch}. Skipping Batch.")
                    continue  # Skip step, don't abort

                batch_masks = get_physics_masks(batch_X_cat, batch_X_num, processor)
                loss = criterion(pred, batch_y, batch_X_num, batch_masks, batch_w)

                # 2. Check Loss NaNs
                if torch.isnan(loss):
                    print(f"  [Warn] NaN in loss at Epoch {epoch}. Skipping Batch.")
                    continue # Skip step

                loss.backward()

                # 3. Check Gradient Norms
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"  [Warn] Gradient explosion (Norm: {total_norm}) at Epoch {epoch}. Skipping.")
                    optimizer.zero_grad() # Clear the "poisoned" gradients
                    continue

                optimizer.step()

            scheduler.step()

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(Xn_val_t, Xc_val_t)
                    val_loss = torch.nn.MSELoss()(val_pred, y_val_t).item()
                    if val_loss < best_fold_loss:
                        best_fold_loss = val_loss

        if not check_model_health(model):
            raise optuna.exceptions.TrialPruned("Model weights corrupted")

        fold_losses.append(best_fold_loss)

    return np.mean(fold_losses)


def run_tuning(
    data_path: str,
    n_trials: int = 50,
    n_folds: int = 5,
    output_dir: str = "experiments",
) -> Tuple[Dict, DataProcessor]:
    # ... [Same as original] ...
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df_clean = clean(df, BASE_NUMERIC, BASE_CATEGORICAL, TARGETS)

    processor = DataProcessor()
    X_num, X_cat = processor.fit_transform(df_clean)
    y_raw = df_clean[TARGETS].values
    y_log = log_transform_targets(y_raw)
    weights = calculate_sample_weights(y_raw)

    validate_data(X_num, X_cat, y_log, "training")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective_cv(
            trial, X_num, X_cat, y_log, weights, processor, n_folds
        ),
        n_trials=n_trials,
        catch=(RuntimeError,),
        show_progress_bar=True,
    )

    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    params_path = os.path.join(output_dir, "best_params.json")
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)

    save_path = os.path.join(output_dir, "processor.pkl")
    processor.save(save_path)

    return study.best_params, processor


def train_final_ensemble(
    data_path: str,
    best_params: Dict,
    processor: Optional[DataProcessor] = None,
    n_models: int = 5,
    output_dir: str = "experiments",
) -> EnsembleModel:
    """Train ensemble with robust NaN handling and Best Epoch Checkpointing."""
    print(f"\n{'='*70}")
    print(f"Training Ensemble of {n_models} models")
    print(f"{'='*70}\n")

    df = pd.read_csv(data_path)
    df_clean = clean(df, BASE_NUMERIC, BASE_CATEGORICAL, TARGETS)

    if processor is None:
        processor = DataProcessor()
        X_num, X_cat = processor.fit_transform(df_clean)
    else:
        X_num, X_cat = processor.transform(df_clean)

    proc_path = os.path.join(output_dir, "processor.pkl")
    processor.save(proc_path)

    y_raw = df_clean[TARGETS].values
    y_log = log_transform_targets(y_raw)
    weights = calculate_sample_weights(y_raw)

    # --- UPDATE: Create Validation Split for Best Epoch Tracking ---
    # Splitting 90% Train / 10% Validation to monitor generalizability
    Xn_tr, Xn_val, Xc_tr, Xc_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_num, X_cat, y_log, weights, test_size=0.1, random_state=42
    )

    # Convert to tensors
    Xn_tr_t, Xc_tr_t, y_tr_t, w_tr_t = to_tensors(Xn_tr, Xc_tr, y_tr, w_tr)
    Xn_val_t, Xc_val_t, y_val_t, w_val_t = to_tensors(Xn_val, Xc_val, y_val, w_val)

    trained_models = []

    # if best_params["weight_decay"] < 1e-4:
    #     print(f"  [Auto-Correction] Boosting weight_decay to 1e-4 for stability.")
    #     best_params["weight_decay"] = 1e-4

    for i in range(n_models):
        print(f"\nTraining Model {i+1}/{n_models}")
        print("-" * 50)
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
            lambda_shear=0.0,
            lambda_input=0.0,
            numeric_cols=BASE_NUMERIC,
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
        )

        n_epochs = 250
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        model_is_dead = False

        # --- UPDATE: Best Epoch Tracking Variables ---
        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = -1

        for epoch in range(n_epochs):
            if model_is_dead:
                break

            # --- Training Loop ---
            model.train()
            indices = torch.randperm(len(Xn_tr_t))

            for j in range(0, len(indices), best_params["batch_size"]):
                batch_idx = indices[j : j + best_params["batch_size"]]

                batch_X_num = Xn_tr_t[batch_idx].clone().detach().requires_grad_(True)
                batch_X_cat = Xc_tr_t[batch_idx]
                batch_y = y_tr_t[batch_idx]
                batch_w = w_tr_t[batch_idx]

                optimizer.zero_grad()
                pred = model(batch_X_num, batch_X_cat)

                if torch.isnan(pred).any():
                    print(
                        f"  [Error] Model produced NaN outputs at Epoch {epoch}. Aborting."
                    )
                    optimizer.zero_grad()
                    continue

                batch_masks = get_physics_masks(batch_X_cat, batch_X_num, processor)
                loss = criterion(pred, batch_y, batch_X_num, batch_masks, batch_w)

                if torch.isnan(loss):
                    print(f"  [Warning] Loss is NaN at Epoch {epoch}. Skipping step.")
                    optimizer.zero_grad()
                    continue

                loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(
                        f"  [Warning] Gradients exploded at Epoch {epoch}. Skipping step."
                    )
                    optimizer.zero_grad()
                    continue

                optimizer.step()

            scheduler.step()

            # --- Validation Loop (Save Best) ---
            model.eval()
            with torch.no_grad():
                val_pred = model(Xn_val_t, Xc_val_t)
                val_loss = torch.nn.MSELoss()(val_pred, y_val_t).item()

                # Check if this is the best epoch
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    # Deep copy the weights to memory
                    best_model_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch+1}/{n_epochs} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f} (Ep {best_epoch})"
                )

        # --- Final Check before saving ---
        # Load the best weights we found
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(
                f"  Restored best weights from Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})"
            )
        else:
            print(
                "  Warning: No best model state found (did training fail?). Using last state."
            )

        if check_model_health(model):
            trained_models.append(model)
            model_name = f"model_{i}.pt"
            model_path = os.path.join(output_dir, model_name)
            save_model_checkpoint(model, processor, best_params, model_path)
            print(f"Saved checkpoint: {model_name}")
        else:
            print(f"Model {i+1} FAILED (Weights Corrupted). Not saved.")

    return EnsembleModel(trained_models)


if __name__ == "__main__":
    # ... [Same as original] ...
    DATA_PATH = r"data/processed/formulation_data_augmented_no_trast.csv"

    N_TRIALS = 25
    N_FOLDS = 5
    N_MODELS = 5
    DO_TUNING = True
    DO_TRAINING = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPERIMENT_DIR = os.path.join("models", "experiments", timestamp)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    print(f"Experiment initialized. Output Directory: {EXPERIMENT_DIR}")

    best_params = None
    processor = None

    if DO_TUNING:
        print("\n" + "=" * 70)
        print("PHASE 1: HYPERPARAMETER TUNING")
        print("=" * 70)
        best_params, processor = run_tuning(
            DATA_PATH, n_trials=N_TRIALS, n_folds=N_FOLDS, output_dir=EXPERIMENT_DIR
        )

    if DO_TRAINING:
        print("\n" + "=" * 70)
        print("PHASE 2: ENSEMBLE TRAINING")
        print("=" * 70)

        if best_params is None:
            best_params = {
                "lr": 5e-5,
                "n_layers": 3,
                "hidden_size": 256,
                "dropout": 0.1,
                "batch_size": 32,
                "weight_decay": 1e-5,
                "lambda_input": 0.0,
                "lambda_shear": 0.1,
            }

        final_model = train_final_ensemble(
            DATA_PATH,
            best_params,
            n_models=N_MODELS,
            processor=processor,
            output_dir=EXPERIMENT_DIR,
        )

    print(f"\nPipeline Complete. Artifacts in: {EXPERIMENT_DIR}")
