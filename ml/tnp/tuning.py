"""
tuning.py
=========
Optuna hyperparameter search objective for TransformerNP.  [TNP-3]

Public API
----------
objective_cv(trial, samples, static_dim, device) -> float
    Group-held-out cross-validation objective.  Returns mean validation MSE
    across up to 6 folds on priority-hard protein groups.

Search space
------------
  hidden_dim    : {128, 192, 256}
  latent_dim    : {64, 128, 192, 256}
  n_heads       : {2, 4, 8}           [TNP-3] — must divide latent_dim evenly
  dropout       : [0.05, 0.30]
  lr            : [1e-4, 5e-3]  (log)
  weight_decay  : [1e-5, 1e-2]  (log)
  lambda_triplet: [0.03, 0.20]  (log)

Trials where latent_dim % n_heads != 0 are pruned immediately.
"""

import numpy as np
import optuna
import torch
from tnp.model import TransformerNP
from tnp.training import train_epoch, validate

# Groups expected to be hardest to generalise; prioritised for held-out folds.
_HARD_GROUPS = ["etanercept", "vudalimab", "pembrolizumab", "ibalizumab"]
_MEDIUM_GROUPS = ["adalimumab", "poly-higg", "nivolumab"]


def objective_cv(
    trial: optuna.Trial,
    samples: list,
    static_dim: int,
    device: torch.device,
) -> float:
    """
    Group-held-out cross-validation objective for Optuna.

    For each fold a single protein group is withheld as the validation set.
    The model is trained from scratch for 40 epochs with 50 iterations each,
    and intermediate val losses are reported for median pruning.

    Args:
        trial:      Optuna trial object.
        samples:    Full preprocessed sample list.
        static_dim: Dimensionality of the static feature vector.
        device:     Computation device.

    Returns:
        Mean validation MSE across all completed folds; ``inf`` if none finished.

    Raises:
        optuna.exceptions.TrialPruned: if latent_dim is not divisible by n_heads,
            or if the MedianPruner triggers.
    """
    hidden_dim = trial.suggest_int("hidden_dim", 128, 256, step=64)
    latent_dim = trial.suggest_int("latent_dim", 64, 256, step=64)
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.05, 0.30)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    lambda_triplet = trial.suggest_float("lambda_triplet", 0.03, 0.20, log=True)

    if latent_dim % n_heads != 0:
        raise optuna.exceptions.TrialPruned()

    # ---- Fold selection: prioritise hard generalisers ----
    priority_groups = [
        g for g in _HARD_GROUPS + _MEDIUM_GROUPS if any(s["group"] == g for s in samples)
    ]
    held_out_groups = priority_groups[:6]

    fold_scores = []
    for fold_idx, held_out in enumerate(held_out_groups):
        train_fold = [s for s in samples if s["group"] != held_out]
        val_fold = [s for s in samples if s["group"] == held_out]
        if len(val_fold) < 2:
            continue

        model = TransformerNP(
            static_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_heads=n_heads,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(40):
            train_epoch(
                model,
                train_fold,
                optimizer,
                device,
                iterations=50,
                lambda_triplet=lambda_triplet,
            )
            val_loss = validate(model, val_fold, device, n_repeats=2)
            trial.report(val_loss, fold_idx * 40 + epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        fold_scores.append(validate(model, val_fold, device, n_repeats=3))

    return float(np.mean(fold_scores)) if fold_scores else float("inf")
