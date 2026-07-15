"""
run.py
======
train_final_model: the stratified-early-stopping training loop, extracted
from cli/train.py's __main__ body so it can be called per-fold by the LOGO
eval harness (visqai.eval.cnp_logo) without duplicating it. cli/train.py now
calls this too -- same behavior, single copy.
"""

from __future__ import annotations

import copy
import os
from collections import defaultdict

import numpy as np
import torch

from visqai.models.cnp import CrossSampleCNP
from visqai.training.loop import log_flatness, log_latent_variance, train_epoch, validate

DEFAULT_PARAMS = {
    "hidden_dim": 128,
    "latent_dim": 64,
    "dropout": 0.15,
    "lr": 5e-4,
    "weight_decay": 1e-4,
}


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
    """Train a CrossSampleCNP to convergence on `samples` (stratified 10%
    per-group early-stopping split) and save best_model.pth to `out_dir`.

    Returns dict(best_loss, epochs_run, group_weights).
    """
    params = dict(params or DEFAULT_PARAMS)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CrossSampleCNP(
        static_dim,
        hidden_dim=params["hidden_dim"],
        latent_dim=params["latent_dim"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None
    epochs_run = 0

    train_set, stop_set = [], []
    groups_dict = defaultdict(list)
    for s in samples:
        groups_dict[s["group"]].append(s)

    for g, g_samples in groups_dict.items():
        np.random.shuffle(g_samples)
        n_val = max(1, int(len(g_samples) * 0.1))
        if len(g_samples) < 2:
            train_set.extend(g_samples)
        else:
            stop_set.extend(g_samples[:n_val])
            train_set.extend(g_samples[n_val:])

    if verbose:
        print(f"Final Train: {len(train_set)} samples | Early Stop Watchlist: {len(stop_set)} samples")

    group_weights = {g: 1.0 for g in set(s["group"] for s in train_set)}
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

        val_loss = validate(model, stop_set, device, n_repeats=10)
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
    torch.save({"state_dict": model.state_dict(), "config": params, "static_dim": static_dim}, save_path)

    return {"best_loss": best_loss, "epochs_run": epochs_run, "group_weights": group_weights}
