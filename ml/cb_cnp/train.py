"""
train.py
========
Main training script for the CBM-CNP viscosity prediction pipeline.

Execution stages
----------------
1.  Load and preprocess formulation data.
2.  (Optional) Optuna hyperparameter search over group-held-out folds.
3.  Final ConceptBottleneckCNP training with cosine-annealed concept
    supervision, EMA difficulty reweighting, and early stopping.
4.  Checkpoint saving.
5.  Concept analysis: gate diagnostics, activation heatmap, per-sample
    proxy correlations, free-concept statistics.
6.  Concept intervention demo (Δlog-viscosity per concept x shear rate).
7.  Parity evaluation (LOO context, per-group and per-shear RMSE/R²).
8.  Permutation feature importance through the concept pathway.

Usage
-----
    python train.py

Adjust ``data``, ``out``, ``trials``, and ``best_params`` at the top of the
``if __name__ == "__main__"`` block for different experiments.
"""

from __future__ import annotations

from ast import Import
import copy
import os
from collections import defaultdict

import joblib
import numpy as np
import optuna
import pandas as pd
import torch


from cb_cnp.constants import (
    CONCEPT_ACTIVATIONS,
    CONCEPT_NAMES,
    N_CONCEPTS_SUPERVISED,
    CONCEPT_DEFS,
)

from cb_cnp.data_pipeline import load_and_preprocess
from cb_cnp.diagnostics import (
    log_concept_activations,
    log_latent_variance,
    run_concept_intervention_demo,
    run_feature_importance,
    run_parity_evaluation,
    save_concept_heatmap,
)
from cb_cnp.models import ConceptBottleneckCNP, _encode_latent
from cb_cnp.trainer import train_epoch, validate
from cb_cnp.tuning import objective_cv


# ============================================================
# Pembrolizumab latent diagnostics helper
# ============================================================


def _pembro_latent_stats(
    model: ConceptBottleneckCNP,
    train_set: list[dict],
    device: torch.device,
) -> tuple[str, str]:
    """
    Compute and return norm/spread strings for pembrolizumab samples.

    Used for the per-epoch console log. Returns ("N/A", "N/A") if no
    pembrolizumab samples are present in train_set.
    """
    pembro_samples = [s for s in train_set if s["group"] == "pembrolizumab"]
    if len(pembro_samples) < 2:
        return "N/A", "N/A"

    model.eval()
    with torch.no_grad():
        idx = np.random.permutation(len(pembro_samples))[: min(10, len(pembro_samples))]
        r_list = []
        for i in idx:
            s = pembro_samples[i]
            stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
            ctx_item = torch.cat([s["points"], stat], dim=1).unsqueeze(0).to(device)
            r_list.append(_encode_latent(model, ctx_item))
        if not r_list:
            return "N/A", "N/A"
        r_pembro = torch.cat(r_list, dim=0)
        norm_str = f"{torch.norm(r_pembro, p=2, dim=-1).mean().item():.3f}"
        dists = [
            torch.norm(r_pembro[i] - r_pembro[j], p=2).item()
            for i in range(len(r_pembro))
            for j in range(i + 1, len(r_pembro))
        ]
        spread_str = f"{np.mean(dists):.3f}" if dists else "N/A"
    model.train()
    return norm_str, spread_str


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------
    # data = "data/raw/formulation_data_03042026.csv"
    data = "data/processed/formulation_data_no_ibal.csv"
    out = "./models/experiments/cbm_cnp_v4"
    trials = 0  # Set > 0 to run Optuna search before final training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_params: dict = {
        "hidden_dim": 128,
        "latent_dim": 128,
        "dropout": 0.15,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "n_free_concepts": 4,
        "lambda_concept_sup_init": 0.30,  # annealed from this value
        "lambda_concept_sup_min": 0.20,  # annealed to this value
        "lambda_triplet": 0.10,
        "lambda_decov": 0.03,
        "lambda_sparsity": 0.05,
        "sup_anneal_epochs": 120,
        "epochs": 150,
        "meta_holdout_prob": 0.20,
    }

    # ----------------------------------------------------------
    # Stage 1: Data loading and preprocessing
    # ----------------------------------------------------------
    samples, static_dim = load_and_preprocess(data, save_dir=out)
    n_groups = len(set(s["group"] for s in samples))
    print(f"Loaded {len(samples)} samples from {n_groups} protein groups.")

    n_free = best_params["n_free_concepts"]
    n_concepts = N_CONCEPTS_SUPERVISED + n_free
    print(
        f"Concept bottleneck: {N_CONCEPTS_SUPERVISED} supervised + "
        f"{n_free} free = {n_concepts} total concepts"
    )

    # ----------------------------------------------------------
    # Stage 2: Optional Optuna hyperparameter search
    # ----------------------------------------------------------
    if trials > 0:
        print("Starting Group-Held-Out Optuna Optimization (CBM)...")
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        study.optimize(
            lambda t: objective_cv(t, samples, static_dim, device),
            n_trials=trials,
        )
        print("\n--- Tuning Complete ---")
        print("Best params:", study.best_params)
        best_params.update(study.best_params)
    else:
        print("Skipping hyperparameter tuning (trials=0). Using default params.")
        print("Default params:", best_params)

    # Recompute n_concepts in case Optuna updated n_free_concepts
    n_concepts = N_CONCEPTS_SUPERVISED + best_params.get("n_free_concepts", n_free)
    lambda_sup_init = best_params.get("lambda_concept_sup_init", 0.30)
    lambda_sup_min = best_params.get("lambda_concept_sup_min", 0.01)
    sup_anneal_epochs = best_params.get("sup_anneal_epochs", 80)

    # ----------------------------------------------------------
    # Stage 3: Final training
    # ----------------------------------------------------------
    print(f"\nRetraining final ConceptBottleneckCNP (n_concepts={n_concepts}) on ALL data...")
    print(
        f"Concept supervision annealing: {lambda_sup_init:.3f} -> {lambda_sup_min:.3f} "
        f"over {sup_anneal_epochs} epochs (cosine)"
    )

    all_concept_activations = CONCEPT_ACTIVATIONS[: min(n_concepts, N_CONCEPTS_SUPERVISED)] + [
        "tanh"
    ] * max(0, n_concepts - N_CONCEPTS_SUPERVISED)

    final_model = ConceptBottleneckCNP(
        static_dim,
        hidden_dim=best_params["hidden_dim"],
        latent_dim=best_params["latent_dim"],
        n_concepts=n_concepts,
        concept_names=CONCEPT_NAMES
        + [f"latent_{i}" for i in range(max(0, n_concepts - N_CONCEPTS_SUPERVISED))],
        concept_activations=all_concept_activations,
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

    # Stratified 10% early-stop split (FIX-WATCHLIST)
    final_train_set: list[dict] = []
    final_stop_set: list[dict] = []
    groups_dict: dict[str, list[dict]] = defaultdict(list)
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

    best_loss = float("inf")
    patience_counter = 0
    patience_limit = 80
    best_state = None

    group_weights: dict[str, float] = {g: 1.0 for g in set(s["group"] for s in final_train_set)}
    ema_alpha = 0.3

    n_epochs = best_params.get("epochs", 500)
    for ep in range(n_epochs):

        # Cosine annealing of concept supervision weight
        if ep < sup_anneal_epochs:
            anneal_frac = ep / sup_anneal_epochs
            current_lambda_sup = lambda_sup_min + (lambda_sup_init - lambda_sup_min) * (
                0.5 * (1 + np.cos(np.pi * anneal_frac))
            )
        else:
            current_lambda_sup = lambda_sup_min

        train_loss, per_group_mse = train_epoch(
            final_model,
            final_train_set,
            optimizer,
            device,
            iterations=100,
            group_weights=group_weights,
            lambda_concept_sup=current_lambda_sup,
            lambda_triplet=best_params.get("lambda_triplet", 0.10),
            lambda_decov=best_params.get("lambda_decov", 0.03),
            lambda_sparsity=best_params.get("lambda_sparsity", 0.01),
            meta_holdout_prob=best_params.get("meta_holdout_prob", 0.20),
        )

        # EMA difficulty reweighting (FIX-6)
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

            gate_vals = final_model.concept_gates().cpu().numpy()
            gate_str = " ".join(
                f"{final_model.concept_names[i][:8]}:{gate_vals[i]:.2f}"
                for i in range(min(len(gate_vals), 6))
            )
            if len(gate_vals) > 6:
                gate_str += f" +{len(gate_vals)-6} more"

            pembro_norm_str, pembro_spread_str = _pembro_latent_stats(
                final_model, final_train_set, device
            )

            top_hard = sorted(group_weights.items(), key=lambda x: -x[1])[:3]
            hard_str = ", ".join(f"{g}:{w:.2f}" for g, w in top_hard)

            print(
                f"Epoch {ep:3d}: Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"LR {current_lr:.2e} | LatentVar {latent_var:.3f} | "
                f"λ_sup {current_lambda_sup:.3f} | "
                f"Pembro [Norm: {pembro_norm_str} | Spread: {pembro_spread_str}] | "
                f"Top hard: [{hard_str}]"
            )
            if ep % 30 == 0:
                print(f"  Gates: [{gate_str}]")

            if ep >= 30 and latent_var < 0.2:
                print(
                    f"  *** WARNING: LatentVar={latent_var:.3f} is very low. "
                    "Context collapse may still be occurring. ***"
                )

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(final_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Stopping early at epoch {ep}. Best Val Loss: {best_loss:.4f}")
            break

    if best_state is not None:
        final_model.load_state_dict(best_state)

    # ----------------------------------------------------------
    # Stage 4: Save checkpoint
    # ----------------------------------------------------------
    os.makedirs(out, exist_ok=True)
    save_path = os.path.join(out, "best_model.pth")
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "config": best_params,
            "static_dim": static_dim,
            "n_concepts": n_concepts,
            "concept_names": final_model.concept_names,
            "concept_activations": final_model._concept_activations,
            "concept_gate_values": final_model.concept_gates().cpu().numpy().tolist(),
            "model_class": "ConceptBottleneckCNP",
        },
        save_path,
    )
    print(f"Model saved to {save_path}")
    print(
        f"Final group difficulty weights: "
        f"{dict(sorted(group_weights.items(), key=lambda x: -x[1]))}"
    )

    # ----------------------------------------------------------
    # Stage 5: Concept analysis
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONCEPT ANALYSIS (v3)")
    print("=" * 60)

    gate_vals = final_model.concept_gates().cpu().numpy()
    print("\nLearned concept gates (0=unused, 1=fully open):")
    print(f"  {'Concept':<28} {'Gate':>8} {'Status':>12}")
    print("  " + "-" * 50)
    for i, (cname, gval) in enumerate(zip(final_model.concept_names, gate_vals)):
        status = "OPEN" if gval > 0.5 else ("PARTIAL" if gval > 0.1 else "CLOSED")
        bar = "█" * int(gval * 20)
        print(f"  {cname:<28} {gval:>8.3f} {status:>12}  {bar}")

    gate_df = pd.DataFrame(
        {
            "Concept": final_model.concept_names,
            "Gate_value": gate_vals,
            "Activation_type": final_model._concept_activations,
        }
    )
    gate_df.to_csv(os.path.join(out, "concept_gates.csv"), index=False)

    group_concepts, concept_matrix, group_names = log_concept_activations(
        final_model,
        samples,
        device,
        n_draws=20,
        k=8,
    )

    if concept_matrix is not None:
        save_concept_heatmap(
            concept_matrix,
            group_names,
            final_model.concept_names,
            save_path=os.path.join(out, "concept_heatmap.png"),
        )
        df_concepts = pd.DataFrame(
            concept_matrix,
            index=group_names,
            columns=final_model.concept_names,
        )
        df_concepts.index.name = "Group"
        concepts_csv = os.path.join(out, "concept_activations.csv")
        df_concepts.to_csv(concepts_csv)
        print(f"Concept activations saved to {concepts_csv}")

        # Summary table
        print("\nConcept activations by protein group (mean over 20 context draws):")
        col_w = 13
        header = f"  {'Group':<22}" + "".join(
            f"{c[:col_w]:>{col_w}}" for c in final_model.concept_names
        )
        print(header)
        print("  " + "-" * (22 + col_w * len(final_model.concept_names)))
        for gname in group_names:
            vals = group_concepts[gname]
            row = f"  {gname:<22}" + "".join(f"{v:>{col_w}.3f}" for v in vals)
            print(row)

        # Per-sample concept-proxy Pearson correlations
        print("\nPer-sample concept-proxy correlation (individual encoding):")
        final_model.eval()
        per_sample_concepts: list[np.ndarray] = []
        per_sample_proxies: list[np.ndarray] = []
        with torch.no_grad():
            for s in samples:
                if "concept_targets" not in s:
                    continue
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_single = torch.cat([s["points"], stat], dim=1).unsqueeze(0).to(device)
                c_single = final_model.encode_memory(ctx_single).squeeze(0).cpu().numpy()
                per_sample_concepts.append(c_single[:N_CONCEPTS_SUPERVISED])
                per_sample_proxies.append(s["concept_targets"].numpy())

        if per_sample_concepts:
            concept_arr = np.stack(per_sample_concepts)
            proxy_arr = np.stack(per_sample_proxies)
            print(f"  {'Concept':<28} {'Proxy column':<22} {'Act':>6} {'Pearson r':>10}")
            print("  " + "-" * 70)
            for ci, (cname, pcol, _, act_type) in enumerate(CONCEPT_DEFS):
                if concept_arr.shape[0] > 2:
                    r_val = np.corrcoef(proxy_arr[:, ci], concept_arr[:, ci])[0, 1]
                    print(f"  {cname:<28} {pcol:<22} {act_type:>6} {r_val:>10.3f}")

        # Free concept residual statistics
        if n_concepts > N_CONCEPTS_SUPERVISED and per_sample_concepts:
            print("\nFree concept residual correlation (candidates for naming):")
            free_concepts_all: list[np.ndarray] = []
            with torch.no_grad():
                for s in samples:
                    stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                    ctx_single = torch.cat([s["points"], stat], dim=1).unsqueeze(0).to(device)
                    c_full = final_model.encode_memory(ctx_single).squeeze(0).cpu().numpy()
                    free_concepts_all.append(c_full[N_CONCEPTS_SUPERVISED:])

            if free_concepts_all:
                free_arr = np.stack(free_concepts_all)
                print(f"  {'Free concept':<20} {'Gate':>8} {'Mean |c|':>10} {'Std':>8}")
                print("  " + "-" * 48)
                for fi in range(free_arr.shape[1]):
                    fname = final_model.concept_names[N_CONCEPTS_SUPERVISED + fi]
                    fgate = gate_vals[N_CONCEPTS_SUPERVISED + fi]
                    fmean = np.abs(free_arr[:, fi]).mean()
                    fstd = free_arr[:, fi].std()
                    print(f"  {fname:<20} {fgate:>8.3f} {fmean:>10.3f} {fstd:>8.3f}")

    # ----------------------------------------------------------
    # Stage 6: Concept intervention demo
    # ----------------------------------------------------------
    run_concept_intervention_demo(
        final_model,
        samples,
        device,
        physics_scaler_path=os.path.join(out, "physics_scaler.pkl"),
        save_dir=out,
    )

    # ----------------------------------------------------------
    # Stage 7: Parity evaluation
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("PARITY EVALUATION")
    print("=" * 60)
    print(f"Data: {data}")

    physics_scaler_eval = joblib.load(os.path.join(out, "physics_scaler.pkl"))
    raw_df = pd.read_csv(data)

    run_parity_evaluation(
        final_model,
        samples,
        raw_df,
        physics_scaler_eval,
        device,
        out,
    )

    # ----------------------------------------------------------
    # Stage 8: Feature importance
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Permutation, Concept-pathway)")
    print("=" * 60)

    parity_shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1000.0,
        "Viscosity_10000": 10000.0,
        "Viscosity_100000": 100000.0,
        "Viscosity_15000000": 1.5e7,
    }

    run_feature_importance(
        final_model,
        samples,
        raw_df,
        preprocessor_path=os.path.join(out, "preprocessor.pkl"),
        static_dim=static_dim,
        physics_scaler=physics_scaler_eval,
        parity_shear_map=parity_shear_map,
        device=device,
        out=out,
    )
