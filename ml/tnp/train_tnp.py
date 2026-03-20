"""
train_tnp.py
============
Entrypoint for training and evaluating TransformerNP (TNP).

This script is deliberately thin — it only orchestrates the pipeline by
calling into the dedicated modules.  All logic lives in:

  constants.py   — domain lookup tables and config
  model.py       — TransformerNP architecture          [TNP-1 - TNP-3]
  data.py        — feature engineering and data loading
  training.py    — train_epoch, validate, tensor builders
  diagnostics.py — latent variance, attention stats     [TNP-4]
  tuning.py      — Optuna cross-validation objective    [TNP-3]
  evaluation.py  — parity evaluation and feature importance

Usage
-----
  python train_tnp.py

Adjust DATA, OUT, and TRIALS at the top of __main__ as needed.
Set TRIALS > 0 to enable Optuna hyperparameter search before final training.
"""

import copy
import os
from collections import defaultdict

import joblib
import numpy as np
import optuna
import torch
from tnp.constants import NON_PROTEIN_GROUPS
from tnp.data import load_and_preprocess
from tnp.diagnostics import (
    log_attention_stats,
    log_latent_variance,
    save_attention_heatmap,
)
from tnp.evaluation import run_feature_importance, run_parity_evaluation
from tnp.model import TransformerNP, _encode_latent, _forward
from tnp.training import (
    _build_ctx_encoded,
    _build_ctx_tensor,
    _build_tgt_tensors,
    train_epoch,
    validate,
)
from tnp.tuning import objective_cv

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    DATA = "data/raw/formulation_data_03042026.csv"
    OUT = "./models/experiments/tnp_v1"
    TRIALS = 0  # set > 0 to enable Optuna search

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Default / seed hyperparameters ----
    best_params = {
        "hidden_dim": 128,
        "latent_dim": 128,
        "n_heads": 4,
        "dropout": 0.30,
        "lr": 5e-4,
        "weight_decay": 5e-3,
        "lambda_triplet": 0.10,
        "epochs": 150,
    }

    # ---- Data loading ----
    # [TNP-ATTN-6] Returns 3 values: full ctx dim and reduced qry dim
    samples, static_ctx_dim, static_qry_dim = load_and_preprocess(DATA, save_dir=OUT)
    print(
        f"Loaded {len(samples)} samples from "
        f"{len(set(s['group'] for s in samples))} protein groups."
    )
    print(
        f"TransformerNP | static_ctx_dim={static_ctx_dim} | "
        f"static_qry_dim={static_qry_dim} | device={device}"
    )

    # ---- Optuna hyperparameter search ----
    if TRIALS > 0:
        print("Starting Group-Held-Out Optuna Optimisation (TNP)...")
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        study.optimize(
            lambda t: objective_cv(t, samples, static_ctx_dim, device),
            n_trials=TRIALS,
        )
        print("\n--- Tuning Complete ---")
        print("Best params:", study.best_params)
        best_params.update(study.best_params)
    else:
        print("Skipping hyperparameter tuning (TRIALS=0).  Using default params.")
        print("Default params:", best_params)

    # Ensure latent_dim divisibility by n_heads
    latent_dim = best_params["latent_dim"]
    n_heads = best_params["n_heads"]
    if latent_dim % n_heads != 0:
        latent_dim = (latent_dim // n_heads) * n_heads
        print(f"Adjusted latent_dim -> {latent_dim} for n_heads={n_heads} divisibility")
        best_params["latent_dim"] = latent_dim

    # ---- Final model construction ----
    print(
        f"\nRetraining final TransformerNP "
        f"(latent_dim={latent_dim}, n_heads={n_heads}) on ALL data..."
    )
    final_model = TransformerNP(
        static_ctx_dim,
        static_qry_dim=static_qry_dim,
        hidden_dim=best_params["hidden_dim"],
        latent_dim=latent_dim,
        n_heads=n_heads,
        dropout=best_params["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=25,
    )

    # ---- Stratified 10 % early-stop watchlist [FIX-WATCHLIST] ----
    final_train_set: list = []
    final_stop_set: list = []
    groups_dict: dict[str, list] = defaultdict(list)
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
        f"Final Train: {len(final_train_set)} | "
        f"Early Stop Watchlist (stratified 10%): {len(final_stop_set)}"
    )

    # ---- Training loop ----
    best_loss = float("inf")
    patience_counter = 0
    patience_limit = 80
    best_state = None
    group_weights = {g: 1.0 for g in set(s["group"] for s in final_train_set)}
    ema_alpha = 0.3

    for ep in range(500):
        train_loss, per_group_mse = train_epoch(
            final_model,
            final_train_set,
            optimizer,
            device,
            iterations=100,
            group_weights=group_weights,
            lambda_triplet=best_params.get("lambda_triplet", 0.10),
        )

        # [FIX-6] EMA difficulty reweighting
        for g, mse in per_group_mse.items():
            group_weights[g] = ema_alpha * mse + (1 - ema_alpha) * group_weights[g]
        total_w = sum(group_weights.values())
        n_g = len(group_weights)
        for g in group_weights:
            group_weights[g] = group_weights[g] / total_w * n_g

        val_loss = validate(final_model, final_stop_set, device, n_repeats=10)
        scheduler.step(val_loss)

        # ---- Periodic diagnostic logging (every 10 epochs) ----
        if ep % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            current_temp = final_model.get_temperature()
            latent_var = log_latent_variance(final_model, final_train_set, device)

            # log_attention_stats now returns 3 values: entropy, w_std, h_max [TNP-ATTN-4]
            mean_entropy, mean_w_std, mean_h_max = log_attention_stats(
                final_model, final_train_set, device
            )
            # Fractional saturation: 0 = perfectly focused, 1 = uniform/degenerate
            attn_frac = mean_entropy / mean_h_max if mean_h_max > 0 else 1.0

            # Pembrolizumab latent spread (sentinel hard-group monitor)
            pembro_samples = [
                s for s in final_train_set if s["group"] == "pembrolizumab"
            ]
            pembro_norm_str = pembro_spread_str = "N/A"
            if len(pembro_samples) > 1:
                final_model.eval()
                with torch.no_grad():
                    idx = np.random.permutation(len(pembro_samples))[
                        : min(10, len(pembro_samples))
                    ]
                    r_list = []
                    for i in idx:
                        s = pembro_samples[i]
                        stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                        ctx_item = (
                            torch.cat([s["points"], stat], dim=1)
                            .unsqueeze(0)
                            .to(device)
                        )
                        r_list.append(_encode_latent(final_model, ctx_item))
                    if r_list:
                        r_p = torch.cat(r_list, dim=0)
                        pembro_norm_str = (
                            f"{torch.norm(r_p, p=2, dim=-1).mean().item():.3f}"
                        )
                        dists = [
                            torch.norm(r_p[i] - r_p[j], p=2).item()
                            for i in range(len(r_p))
                            for j in range(i + 1, len(r_p))
                        ]
                        if dists:
                            pembro_spread_str = f"{np.mean(dists):.3f}"
                final_model.train()

            top_hard = sorted(group_weights.items(), key=lambda x: -x[1])[:3]
            hard_str = ", ".join(f"{g}:{w:.2f}" for g, w in top_hard)

            print(
                f"Epoch {ep:3d}: Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"LR {current_lr:.2e} | T={current_temp:.3f} | "
                f"LatentVar {latent_var:.3f} | "
                f"Attn [H={mean_entropy:.3f} H/Hmax={attn_frac:.2f} σ={mean_w_std:.3f}] | "
                f"Pembro [Norm: {pembro_norm_str} | Spread: {pembro_spread_str}] | "
                f"Top hard: [{hard_str}]"
            )

            if ep >= 30 and latent_var < 0.2:
                print(
                    f"  *** WARNING: LatentVar={latent_var:.3f} very low — "
                    "context collapse may be occurring. ***"
                )

            # [TNP-ATTN-4] Entropy warning uses the actual ceiling (ln(N_ctx_samples)),
            # not the old placeholder ln(600) which was always wrong.
            # Threshold: H/H_max > 0.85 means the model is near-uniformly attending.
            if ep >= 20 and attn_frac > 0.85:
                print(
                    f"  *** WARNING: Attn H/Hmax={attn_frac:.2f} "
                    f"(H={mean_entropy:.3f}, ceil={mean_h_max:.3f}). "
                    "Model is still average-pooling context. ***"
                )

        # ---- Early stopping ----
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(final_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Stopping early at epoch {ep}.  Best Val Loss: {best_loss:.4f}")
            break

    if best_state is not None:
        final_model.load_state_dict(best_state)

    # ---- Save checkpoint [TNP-6] ----
    os.makedirs(OUT, exist_ok=True)
    save_path = os.path.join(OUT, "best_model.pth")
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "config": best_params,
            "static_dim": static_ctx_dim,  # backward compat key
            "static_ctx_dim": static_ctx_dim,  # [TNP-ATTN-6]
            "static_qry_dim": static_qry_dim,  # [TNP-ATTN-6]
            "model_class": "TransformerNP",
        },
        save_path,
    )
    print(f"Model saved to {save_path}")
    print(
        f"Final group difficulty weights: "
        f"{dict(sorted(group_weights.items(), key=lambda x: -x[1]))}"
    )

    # ================================================================
    # Post-training analysis
    # ================================================================

    # ---- [TNP-4] Attention analysis ----
    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS")
    print("=" * 60)

    save_attention_heatmap(
        final_model,
        samples,
        device,
        save_path=os.path.join(OUT, "attention_entropy_heatmap.png"),
    )

    # Per-group attention entropy summary using sample-level context [TNP-ATTN-4]
    groups_eval: dict[str, list] = defaultdict(list)
    for s in samples:
        groups_eval[s["group"]].append(s)

    protein_groups = sorted(
        g
        for g, sl in groups_eval.items()
        if len(sl) >= 4 and g not in NON_PROTEIN_GROUPS
    )

    print("\nPer-group attention entropy (k=8 context samples, sample-level tokens):")
    print(f"  {'Group':<24} {'H':>8} {'H/Hmax':>8} {'W σ':>8}  Interpretation")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")

    final_model.eval()
    with torch.no_grad():
        for prot in protein_groups:
            task_samples = groups_eval[prot]
            k_eff = min(8, len(task_samples) - 1)
            if k_eff < 1:
                continue
            idx_c = np.random.choice(len(task_samples), k_eff, replace=False)
            remain = [i for i in range(len(task_samples)) if i not in set(idx_c)]
            idx_q = np.random.choice(remain, min(4, len(remain)), replace=False)

            # [TNP-ATTN-4] Sample-level context encoding
            ctx_t = _build_ctx_encoded(final_model, task_samples, idx_c, device)
            n_ctx_samples = ctx_t.shape[1]

            q_shear, q_stat, _ = _build_tgt_tensors(task_samples, idx_q, device)
            if q_shear is None:
                continue

            _, aw = _forward(final_model, ctx_t, q_shear, q_stat, ctx_is_encoded=True)
            w = aw.squeeze(0).cpu().numpy()
            eps = 1e-8
            H = (-np.sum(w * np.log(w + eps), axis=-1)).mean()
            W_std = w.std(axis=-1).mean()
            h_max = np.log(n_ctx_samples)
            frac = H / h_max if h_max > 0 else 1.0
            interp = (
                "focused" if frac < 0.50 else "moderate" if frac < 0.85 else "diffuse"
            )
            print(f"  {prot:<24} {H:>8.4f} {frac:>8.2f} {W_std:>8.4f}  {interp}")

    # ---- Parity evaluation ----
    run_parity_evaluation(final_model, samples, DATA, OUT, device)

    # ---- Feature importance ----
    physics_scaler = joblib.load(os.path.join(OUT, "physics_scaler.pkl"))
    import pandas as pd

    raw_df = pd.read_csv(DATA)

    parity_shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1_000.0,
        "Viscosity_10000": 10_000.0,
        "Viscosity_100000": 100_000.0,
        "Viscosity_15000000": 1.5e7,
    }
    key_log_shears = np.log10(list(parity_shear_map.values()))
    n_shears = len(key_log_shears)
    shear_mean = physics_scaler.mean_[0]
    shear_scale = physics_scaler.scale_[0]
    visc_mean = physics_scaler.mean_[1]
    visc_scale = physics_scaler.scale_[1]

    scaled_log_shears = torch.tensor(
        [(ls - shear_mean) / shear_scale for ls in key_log_shears],
        dtype=torch.float32,
    ).to(device)

    run_feature_importance(
        final_model,
        samples,
        raw_df,
        OUT,
        device,
        static_qry_dim,  # [TNP-ATTN-6] permute query features, not ctx features
        scaled_log_shears,
        visc_mean,
        visc_scale,
        n_shears,
        parity_shear_map,
    )
