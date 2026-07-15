"""
train.py
========
CLI training entrypoint: Optuna hyperparameter search (group-held-out CV)
followed by final retraining on all data with stratified early stopping.

Argparse'd replacement for the hardcoded-config `__main__` block previously
at the bottom of ml/cnp_mk2/train_o_net_v4_rung1.py (data path, output dir,
and trial count were module-level constants meant to be hand-edited before
each run).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import optuna
import torch

from visqai.logging_config import configure_logging
from visqai.training.data import load_and_preprocess
from visqai.training.run import DEFAULT_PARAMS, train_final_model
from visqai.training.tuning import objective_cv


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train CrossSampleCNP (Optuna tuning + final retrain).")
    p.add_argument(
        "--data",
        default=Path("data/processed/no_ibal.csv"),
        help="Path to the training CSV.",
    )
    p.add_argument(
        "--out",
        default=Path("models/experiments/o_net_no_ibal_rung2"),
        help="Output directory for checkpoint + preprocessor.",
    )
    p.add_argument(
        "--trials", type=int, default=0, help="Optuna trials (0 to skip tuning and use defaults)."
    )
    p.add_argument("--max-epochs", type=int, default=500, help="Max epochs for the final retrain.")
    p.add_argument(
        "--patience",
        type=int,
        default=80,
        help="Early-stopping patience (epochs) for the final retrain.",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    return p.parse_args(argv)


def main(argv=None):
    configure_logging()
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples, static_dim, physics_scaler, protected_indices = load_and_preprocess(
        args.data, save_dir=args.out
    )
    print(
        f"Loaded {len(samples)} samples from {len(set(s['group'] for s in samples))} protein groups."
    )
    print(f"Protecting {len(protected_indices)} load-bearing static features from masking.")

    best_params = dict(DEFAULT_PARAMS)
    if args.trials > 0:
        print("Starting Group-Held-Out Optuna Optimization...")
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        study.optimize(
            lambda t: objective_cv(
                t, samples, static_dim, device, physics_scaler, protected_indices
            ),
            n_trials=args.trials,
        )
        print("\n--- Tuning Complete ---")
        print("Best params:", study.best_params)
        best_params = study.best_params
    else:
        print("Skipping hyperparameter tuning since --trials=0. Using default params.")
        print("Default params:", best_params)

    print("\nRetraining final model on ALL data...")
    result = train_final_model(
        samples,
        static_dim,
        physics_scaler,
        protected_indices,
        out_dir=args.out,
        params=best_params,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
    )
    print(f"Model saved to {os.path.join(args.out, 'best_model.pth')}")
    print(f"Best val loss: {result['best_loss']:.4f} after {result['epochs_run']} epochs")
    print(
        f"Final group difficulty weights: "
        f"{dict(sorted(result['group_weights'].items(), key=lambda x: -x[1]))}"
    )


if __name__ == "__main__":
    main()
