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
import copy
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna
import torch

from visqai.logging_config import configure_logging
from visqai.models.cnp import CrossSampleCNP
from visqai.training.data import load_and_preprocess
from visqai.training.loop import (
    log_flatness,
    log_latent_variance,
    train_epoch,
    validate,
)
from visqai.training.tuning import objective_cv

DEFAULT_PARAMS = {
    "hidden_dim": 128,
    "latent_dim": 64,
    "dropout": 0.15,
    "lr": 5e-4,
    "weight_decay": 1e-4,
}


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
    final_model = CrossSampleCNP(
        static_dim,
        hidden_dim=best_params["hidden_dim"],
        latent_dim=best_params["latent_dim"],
        dropout=best_params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25
    )

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    # Stratified 10% split: hold out a few samples from every group instead
    # of holding out entire groups.
    final_train_set = []
    final_stop_set = []
    groups_dict = defaultdict(list)
    for s in samples:
        groups_dict[s["group"]].append(s)

    for g, g_samples in groups_dict.items():
        np.random.shuffle(g_samples)
        n_val = max(1, int(len(g_samples) * 0.1))
        if len(g_samples) < 2:
            final_train_set.extend(g_samples)
        else:
            final_stop_set.extend(g_samples[:n_val])
            final_train_set.extend(g_samples[n_val:])

    print(
        f"Final Train: {len(final_train_set)} samples | "
        f"Early Stop Watchlist (stratified 10%): {len(final_stop_set)} samples"
    )

    group_weights = {g: 1.0 for g in set(s["group"] for s in final_train_set)}
    ema_alpha = 0.3

    for ep in range(args.max_epochs):
        train_loss, per_group_mse = train_epoch(
            final_model,
            final_train_set,
            optimizer,
            device,
            iterations=100,
            group_weights=group_weights,
            physics_scaler=physics_scaler,
            protected_indices=protected_indices,
        )

        for g, mse in per_group_mse.items():
            group_weights[g] = ema_alpha * mse + (1 - ema_alpha) * group_weights[g]
        total_w = sum(group_weights.values())
        n_g = len(group_weights)
        for g in group_weights:
            group_weights[g] = group_weights[g] / total_w * n_g

        val_loss = validate(final_model, final_stop_set, device, n_repeats=10)
        scheduler.step(val_loss)

        if ep % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            latent_var = log_latent_variance(final_model, final_train_set, device)
            shear_shape, cross_sample = log_flatness(final_model, final_train_set, device)
            top_hard = sorted(group_weights.items(), key=lambda x: -x[1])[:3]
            hard_str = ", ".join(f"{g}:{w:.2f}" for g, w in top_hard)

            print(
                f"Epoch {ep:3d}: Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"LR {current_lr:.2e} | LatentVar {latent_var:.3f} | "
                f"Flatness [Shape: {shear_shape:.3f} | XGroup: {cross_sample:.3f}] | "
                f"Top hard: [{hard_str}]"
            )

            if ep >= 30 and latent_var < 0.2:
                print(
                    f"  *** WARNING: LatentVar={latent_var:.3f} is very low. "
                    "Context collapse may still be occurring. ***"
                )
            if ep >= 30 and (shear_shape < 0.10 or cross_sample < 0.10):
                print(
                    f"  *** FLATNESS WARNING: Shape={shear_shape:.3f}, XGroup={cross_sample:.3f}. "
                    "Predictions are collapsing toward a constant population-mean curve. ***"
                )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(final_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Stopping early at epoch {ep}. Best Val Loss: {best_loss:.4f}")
            break

    if best_state is not None:
        final_model.load_state_dict(best_state)

    save_path = os.path.join(args.out, "best_model.pth")
    torch.save(
        {"state_dict": final_model.state_dict(), "config": best_params, "static_dim": static_dim},
        save_path,
    )
    print(f"Model saved to {save_path}")
    print(
        f"Final group difficulty weights: {dict(sorted(group_weights.items(), key=lambda x: -x[1]))}"
    )


if __name__ == "__main__":
    main()
