"""Train and checkpoint the final CrossSampleCNP model.

This module contains the reusable final-model training routine used by both
the command-line training workflow and the leave-one-group-out evaluation
harness. The training loop performs stratified per-group early-stopping
validation, combines context-informed and literal zero-shot validation for
checkpoint selection, adapts group sampling weights from observed training
error, and records latent-space diagnostics during training.

The implementation was extracted from the former `cli.train` entry point
so that fold-specific evaluation can invoke exactly the same final-model
training procedure without maintaining a second implementation.
"""

from __future__ import annotations

import copy
import os
from collections import defaultdict

import numpy as np
import torch

from visqai.constants import DEFAULT_PARAMS
from visqai.models.cnp import CrossSampleCNP
from visqai.training.loop import (
    log_flatness,
    log_latent_variance,
    train_epoch,
    validate,
    validate_zero_shot,
)


def train_final_model(
    samples,
    static_dim,
    physics_scaler,
    protected_indices,
    out_dir,
    params=None,
    max_epochs=500,
    patience=80,
    device=None,
    verbose=True,
):
    """Train a :class:`CrossSampleCNP` and save its best checkpoint.

    The supplied samples are partitioned independently within each group,
    reserving approximately 10% of each group for early-stopping validation.
    Each epoch trains the model with :func:`train_epoch`, updates adaptive
    group weights from the observed per-group training error, and evaluates
    both context-informed and literal zero-shot performance. The two
    validation losses are averaged equally for learning-rate scheduling and
    best-checkpoint selection, ensuring that optimization cannot improve
    few-shot performance at the expense of the model's zero-shot path.

    The model state corresponding to the lowest mixed validation loss is
    restored before being written to `best_model.pth`. The checkpoint also
    contains the training configuration and static feature dimensionality
    required to reconstruct the model for inference.

    Args:
        samples: Training samples consumed by the CrossSampleCNP. Each sample
            is expected to contain the fields required by
            :func:`train_epoch`, including `static`, `points`, and
            `group`.
        static_dim: Number of static features supplied to the model for each
            query or context point.
        physics_scaler: Fitted scaler used to recover physical viscosity
            magnitudes when computing viscosity-aware training weights.
        protected_indices: Indices of static features that must not be
            randomly masked during training because they represent
            load-bearing physical or identity information.
        out_dir: Directory in which the best model checkpoint is written.
            The directory is created if it does not already exist.
        params: Optional model and optimizer configuration. Missing values are
            populated from :data:`visqai.constants.DEFAULT_PARAMS`.
        max_epochs: Maximum number of training epochs.
        patience: Number of consecutive epochs without improvement in the
            mixed validation loss before early stopping.
        device: Torch device on which the model and training tensors are
            placed. If omitted, CUDA is used when available; otherwise CPU is
            used.
        verbose: Whether to print epoch progress, validation diagnostics, and
            early-stopping information.

    Returns:
        A dictionary containing:

        * `best_loss` -- lowest mixed validation loss observed during
          training.
        * `epochs_run` -- number of epochs actually executed.
        * `group_weights` -- final adaptive sampling weight for each
          training group.

    """
    params = dict(params or DEFAULT_PARAMS)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CrossSampleCNP(
        static_dim,
        hidden_dim=params["hidden_dim"],
        latent_dim=params["latent_dim"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25
    )

    best_loss = float("inf")
    patience_counter = 0
    best_state = None
    epochs_run = 0

    train_set, stop_set = [], []
    groups_dict = defaultdict(list)
    for s in samples:
        groups_dict[s["group"]].append(s)

    for _, g_samples in groups_dict.items():
        np.random.shuffle(g_samples)
        n_val = max(1, int(len(g_samples) * 0.1))
        if len(g_samples) < 2:
            train_set.extend(g_samples)
        else:
            stop_set.extend(g_samples[:n_val])
            train_set.extend(g_samples[n_val:])

    if verbose:
        print(
            f"Final Train: {len(train_set)} samples | Early Stop Watchlist: {len(stop_set)} samples"
        )

    group_weights = {g: 1.0 for g in {s["group"] for s in train_set}}
    ema_alpha = 0.3

    for ep in range(max_epochs):
        epochs_run = ep + 1
        train_loss, per_group_mse = train_epoch(
            model,
            train_set,
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

        # Combine context-informed and literal zero-shot validation for checkpoint
        # selection. `validate()` always evaluates with a non-empty context and is
        # therefore blind to the quality of the prior-only zero-shot path. Including
        # `validate_zero_shot()` in the selection metric ensures the chosen checkpoint
        # must perform well in both deployment regimes rather than improving
        # context-conditioned predictions at the expense of zero-shot performance.
        val_loss_ctx = validate(model, stop_set, device, n_repeats=10)
        val_loss_zero = validate_zero_shot(model, stop_set, device, latent_dim=params["latent_dim"])
        val_loss = 0.5 * val_loss_ctx + 0.5 * val_loss_zero
        scheduler.step(val_loss)

        if verbose and ep % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            latent_var = log_latent_variance(model, train_set, device)
            shear_shape, cross_sample = log_flatness(model, train_set, device)
            top_hard = sorted(group_weights.items(), key=lambda x: -x[1])[:3]
            hard_str = ", ".join(f"{g}:{w:.2f}" for g, w in top_hard)
            print(
                f"Epoch {ep:3d}: Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"LR {current_lr:.2e} | LatentVar {latent_var:.3f} | "
                f"Flatness [Shape: {shear_shape:.3f} | XGroup: {cross_sample:.3f}] | "
                f"Top hard: [{hard_str}]"
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Stopping early at epoch {ep}. Best Val Loss: {best_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "best_model.pth")
    torch.save(
        {"state_dict": model.state_dict(), "config": params, "static_dim": static_dim}, save_path
    )

    return {"best_loss": best_loss, "epochs_run": epochs_run, "group_weights": group_weights}
