"""
tuning.py
=========
Optuna hyperparameter search objective: group-held-out CV, scored on
few-shot held-out error (the metric that matches deployment) rather than
full-context error.

Moved verbatim (logic unchanged) from ml/cnp_mk2/train_o_net_v4_rung1.py.
"""

from __future__ import annotations

import numpy as np
import optuna
import torch

from visqai.models.cnp import CrossSampleCNP
from visqai.training.loop import train_epoch, validate_fewshot


def objective_cv(trial, samples, static_dim, device, physics_scaler=None, protected_indices=None):
    hidden_dim = trial.suggest_int("hidden_dim", 128, 256, step=64)
    latent_dim = trial.suggest_int("latent_dim", 128, 256, step=64)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    hard_groups = ["etanercept", "vudalimab", "pembrolizumab", "ibalizumab"]
    medium_groups = ["adalimumab", "poly-higg", "nivolumab"]
    priority_held_out = [g for g in hard_groups + medium_groups if any(s["group"] == g for s in samples)]
    held_out_groups = priority_held_out[:6]

    fold_scores = []
    for fold_idx, held_out in enumerate(held_out_groups):
        train_fold = [s for s in samples if s["group"] != held_out]
        val_fold = [s for s in samples if s["group"] == held_out]

        if len(val_fold) < 2:
            continue

        model = CrossSampleCNP(static_dim, hidden_dim, latent_dim, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train long enough for latent routing to actually form: 120 epochs,
        # scored on FEW-SHOT held-out error (k=1,2,4), not full-context error,
        # so Optuna optimises the metric we deploy on.
        n_epochs = 120
        for epoch in range(n_epochs):
            train_loss, _ = train_epoch(
                model,
                train_fold,
                optimizer,
                device,
                iterations=50,
                physics_scaler=physics_scaler,
                protected_indices=protected_indices,
            )
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                val_loss = validate_fewshot(model, val_fold, device, n_repeats=3)
                trial.report(val_loss, fold_idx * n_epochs + epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        fold_scores.append(validate_fewshot(model, val_fold, device, n_repeats=5))

    return float(np.mean(fold_scores)) if fold_scores else float("inf")
