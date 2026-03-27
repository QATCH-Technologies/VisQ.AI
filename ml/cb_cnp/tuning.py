"""
tuning.py
=========
Optuna hyperparameter search for the CBM-CNP pipeline.

The objective uses group-held-out cross-validation: hard-to-predict proteins
(etanercept, vudalimab, pembrolizumab, ibalizumab) are held out first to
ensure the search is guided by the most challenging formulations.

Functions
---------
objective_cv(trial, samples, static_dim, device)
    Optuna objective. Trains a ConceptBottleneckCNP on each fold and returns
    the mean held-out validation loss.
"""

from __future__ import annotations

import numpy as np
import optuna
import torch

from cb_cnp.constants import N_CONCEPTS_SUPERVISED
from cb_cnp.models import ConceptBottleneckCNP
from cb_cnp.trainer import train_epoch, validate


def objective_cv(
    trial: optuna.Trial,
    samples: list[dict],
    static_dim: int,
    device: torch.device,
) -> float:
    """
    Group-held-out cross-validation objective for Optuna.

    Hyperparameters searched
    ------------------------
    hidden_dim            : {128, 192, 256}
    latent_dim            : {128, 192, 256}
    dropout               : [0.05, 0.30]
    lr                    : [1e-4, 5e-3] (log scale)
    weight_decay          : [1e-5, 1e-2] (log scale)
    n_free_concepts       : {0, 2, 4, 6, 8}
    lambda_concept_sup    : [0.02, 0.30] (log scale)
    lambda_triplet        : [0.03, 0.15] (log scale)
    lambda_decov          : [0.01, 0.10] (log scale)
    lambda_sparsity       : [0.005, 0.05] (log scale)

    Folds
    -----
    Up to 6 folds, prioritising hardest proteins first. Each fold trains
    for 40 epochs (50 iterations each) with cosine-annealed concept supervision.

    Parameters
    ----------
    trial : optuna.Trial
    samples : list[dict]
    static_dim : int
    device : torch.device

    Returns
    -------
    float
        Mean held-out validation MSE across all valid folds.
        Returns ``float("inf")`` if no folds could be evaluated.
    """
    # --- Hyperparameter suggestions ---
    hidden_dim = trial.suggest_int("hidden_dim", 128, 256, step=64)
    latent_dim = trial.suggest_int("latent_dim", 128, 256, step=64)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    n_free = trial.suggest_int("n_free_concepts", 0, 8, step=2)
    n_concepts = N_CONCEPTS_SUPERVISED + n_free
    lambda_concept_sup = trial.suggest_float("lambda_concept_sup", 0.02, 0.30, log=True)
    lambda_triplet = trial.suggest_float("lambda_triplet", 0.03, 0.15, log=True)
    lambda_decov = trial.suggest_float("lambda_decov", 0.01, 0.10, log=True)
    lambda_sparsity = trial.suggest_float("lambda_sparsity", 0.005, 0.05, log=True)

    # --- Fold construction ---
    hard_groups = ["etanercept", "vudalimab", "pembrolizumab", "ibalizumab"]
    medium_groups = ["adalimumab", "poly-higg", "nivolumab"]
    priority_held_out = [
        g for g in hard_groups + medium_groups if any(s["group"] == g for s in samples)
    ]
    held_out_groups = priority_held_out[:6]

    fold_scores: list[float] = []

    for fold_idx, held_out in enumerate(held_out_groups):
        train_fold = [s for s in samples if s["group"] != held_out]
        val_fold = [s for s in samples if s["group"] == held_out]

        if len(val_fold) < 2:
            continue

        model = ConceptBottleneckCNP(
            static_dim,
            hidden_dim,
            latent_dim,
            n_concepts=n_concepts,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        n_epochs = 40
        for epoch in range(n_epochs):
            # Cosine anneal: lambda_concept_sup -> 10% of initial over 40 epochs
            anneal_frac = epoch / n_epochs
            annealed_sup = lambda_concept_sup * (
                0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * anneal_frac))
            )

            train_epoch(
                model,
                train_fold,
                optimizer,
                device,
                iterations=50,
                lambda_concept_sup=annealed_sup,
                lambda_triplet=lambda_triplet,
                lambda_decov=lambda_decov,
                lambda_sparsity=lambda_sparsity,
            )
            val_loss = validate(model, val_fold, device, n_repeats=2)
            trial.report(val_loss, fold_idx * n_epochs + epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        fold_scores.append(validate(model, val_fold, device, n_repeats=3))

    return float(np.mean(fold_scores)) if fold_scores else float("inf")
