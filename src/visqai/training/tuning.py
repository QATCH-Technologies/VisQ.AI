"""
tuning.py
=========

Optuna-based hyperparameter optimization for the CrossSampleCNP model.

The search evaluates candidate architectures and optimizer settings using
group-held-out cross-validation. Each held-out group represents an unseen
protein at evaluation time, and the objective is the model's few-shot
prediction error across 1-, 2-, and 4-shot contexts. This deliberately
matches the deployment regime rather than optimizing full-context validation
error.
"""

from __future__ import annotations

import numpy as np
import optuna
import torch

from visqai.models.cnp import CrossSampleCNP
from visqai.training.loop import train_epoch, validate_fewshot


def objective_cv(trial, samples, static_dim, device, physics_scaler=None, protected_indices=None):
    """Evaluate one Optuna trial using group-held-out few-shot CV.

    Hyperparameters proposed by `trial` define a new CrossSampleCNP model
    and AdamW optimizer. The available samples are partitioned by protein
    group so that each validation fold contains a group excluded entirely
    from training. Candidate models are trained for a fixed number of epochs
    and periodically evaluated using few-shot prediction with 1-, 2-, and
    4-shot contexts.

    Intermediate validation results are reported to Optuna, allowing
    underperforming trials to be pruned before completing all folds. The
    final objective value is the mean few-shot validation error across all
    completed held-out groups.

    Args:
        trial: Optuna trial used to sample hyperparameters, report
            intermediate results, and determine whether the trial should be
            pruned.
        samples: Training samples grouped by their `group` identifier.
            Each held-out group is excluded from the corresponding training
            fold.
        static_dim: Number of static features provided to the
            CrossSampleCNP.
        device: Torch device on which the model and training tensors are
            placed.
        physics_scaler: Optional fitted physics scaler used by
            :func:`train_epoch` for viscosity-aware loss weighting.
        protected_indices: Optional indices of static features that must be
            preserved when :func:`train_epoch` applies feature masking.

    Returns:
        The mean few-shot validation error across the evaluated held-out
        groups. Returns `float("inf")` when no valid validation folds are
        available.

    Raises:
        optuna.exceptions.TrialPruned: If Optuna determines that the trial
            should be stopped early based on an intermediate validation
            result.
    """
    hidden_dim = trial.suggest_int("hidden_dim", 128, 256, step=64)
    latent_dim = trial.suggest_int("latent_dim", 128, 256, step=64)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    hard_groups = ["etanercept", "vudalimab", "pembrolizumab", "ibalizumab"]
    medium_groups = ["adalimumab", "poly-higg", "nivolumab"]
    priority_held_out = [
        g for g in hard_groups + medium_groups if any(s["group"] == g for s in samples)
    ]
    held_out_groups = priority_held_out[:6]

    fold_scores = []
    for fold_idx, held_out in enumerate(held_out_groups):
        train_fold = [s for s in samples if s["group"] != held_out]
        val_fold = [s for s in samples if s["group"] == held_out]

        if len(val_fold) < 2:
            continue

        model = CrossSampleCNP(static_dim, hidden_dim, latent_dim, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
