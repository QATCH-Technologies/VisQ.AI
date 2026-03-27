"""
diagnostics.py
==============
Post-training diagnostics and evaluation for the CBM-CNP pipeline.

Functions
---------
log_latent_variance(model, samples, device)
    Inter-group L2 distance in pre-concept latent space.
log_concept_activations(model, samples, device, ...)
    Mean concept activations per protein group (for heatmap / summary table).
save_concept_heatmap(concept_matrix, group_names, concept_names, save_path)
    Save a colour-coded concept x group heatmap PNG.
run_concept_intervention_demo(model, samples, device, ...)
    Sweep each concept independently and record Δlog-viscosity per shear rate.
run_parity_evaluation(model, samples, raw_df, physics_scaler, device, out)
    Predict on all samples (LOO context), compute RMSE/R², save CSV + plot.
run_feature_importance(model, samples, raw_df, preprocessor, ..., out)
    Permutation feature importance through the concept pathway.
"""

from __future__ import annotations

import os
from collections import defaultdict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cb_cnp.batch_utils import _build_ctx_tensor
from cb_cnp.constants import CONCEPT_DEFS, N_CONCEPTS_SUPERVISED, NON_PROTEIN_GROUPS
from cb_cnp.models import ConceptBottleneckCNP, _encode_latent


# ============================================================
# Latent-space diagnostics
# ============================================================


def log_latent_variance(
    model: torch.nn.Module,
    samples: list[dict],
    device: torch.device,
) -> float:
    """
    Compute mean inter-group L2 distance in pre-concept latent r-space.

    A healthy model has high inter-group distance (well-separated protein
    representations). Very low values (< 0.2) indicate context collapse.

    Parameters
    ----------
    model : CrossSampleCNP or ConceptBottleneckCNP
    samples : list[dict]
    device : torch.device

    Returns
    -------
    float
        Mean pairwise L2 distance between group centroid latent vectors.
        Returns 0.0 if fewer than two groups are available.
    """
    model.eval()
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    group_r: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for prot, task_samples in groups.items():
            if len(task_samples) < 2 or prot in NON_PROTEIN_GROUPS:
                continue
            idx = np.random.permutation(len(task_samples))[: min(5, len(task_samples))]
            ctx_items = []
            for i in idx:
                s = task_samples[i]
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_items.append(torch.cat([s["points"], stat], dim=1))
            ctx_t = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)
            r = _encode_latent(model, ctx_t).squeeze(0).cpu().numpy()
            group_r[prot] = r

    if len(group_r) < 2:
        return 0.0

    vecs = np.stack(list(group_r.values()))
    dists = [
        np.linalg.norm(vecs[i] - vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))
    ]
    return float(np.mean(dists))


# ============================================================
# Concept activation analysis
# ============================================================


def log_concept_activations(
    model: torch.nn.Module,
    samples: list[dict],
    device: torch.device,
    n_draws: int = 10,
    k: int = 8,
) -> tuple[dict[str, np.ndarray], np.ndarray | None, list[str]]:
    """
    Compute mean concept activations per protein group.

    Averages ``n_draws`` random context draws of up to ``k`` samples each
    to reduce variance in the concept estimate.

    Parameters
    ----------
    model : ConceptBottleneckCNP (returns empty results for other types)
    samples : list[dict]
    device : torch.device
    n_draws : int
        Number of random context sub-samples per group.
    k : int
        Maximum context samples per draw.

    Returns
    -------
    group_concepts : dict[str, np.ndarray]
        {group_name: mean concept vector [n_concepts]}.
    concept_matrix : np.ndarray [n_groups, n_concepts] or None
    group_names : list[str]
        Row order for concept_matrix (sorted alphabetically).
    """
    if not isinstance(model, ConceptBottleneckCNP):
        return {}, None, []

    model.eval()
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    group_concepts: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for prot, task_samples in groups.items():
            if len(task_samples) < 2:
                continue
            draw_concepts = []
            for _ in range(n_draws):
                k_eff = min(k, len(task_samples))
                idx = np.random.choice(len(task_samples), size=k_eff, replace=False)
                ctx_items = []
                for i in idx:
                    s = task_samples[i]
                    stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                    ctx_items.append(torch.cat([s["points"], stat], dim=1))
                ctx_t = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)
                c = model.encode_concepts(ctx_t).squeeze(0).cpu().numpy()
                draw_concepts.append(c)
            group_concepts[prot] = np.stack(draw_concepts).mean(axis=0)

    if not group_concepts:
        return {}, None, []

    group_names = sorted(group_concepts.keys())
    concept_matrix = np.stack([group_concepts[g] for g in group_names])
    return group_concepts, concept_matrix, group_names


def save_concept_heatmap(
    concept_matrix: np.ndarray,
    group_names: list[str],
    concept_names: list[str],
    save_path: str,
) -> None:
    """
    Save a colour-coded concept activation heatmap to a PNG file.

    Parameters
    ----------
    concept_matrix : np.ndarray [n_groups, n_concepts]
    group_names : list[str]
    concept_names : list[str]
    save_path : str
    """
    fig, ax = plt.subplots(
        figsize=(max(8, len(concept_names) * 0.9), max(4, len(group_names) * 0.45))
    )
    im = ax.imshow(concept_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(concept_names)))
    ax.set_xticklabels(concept_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names, fontsize=9)
    plt.colorbar(im, ax=ax, label="Concept activation [-1, 1]")
    ax.set_title("Concept activations by protein group", fontsize=11)

    for i in range(len(group_names)):
        for j in range(len(concept_names)):
            val = concept_matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if abs(val) < 0.6 else "white",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Concept heatmap saved to {save_path}")


# ============================================================
# Concept intervention demo
# ============================================================


def run_concept_intervention_demo(
    model: torch.nn.Module,
    samples: list[dict],
    device: torch.device,
    physics_scaler_path: str,
    save_dir: str,
) -> None:
    """
    Sweep each concept independently from its activation minimum to maximum
    and record the resulting change in predicted log-viscosity.

    For each protein group, the full group context is encoded to obtain a
    baseline concept vector c. Each concept dimension is then clamped to a
    sweep of values while all others are held fixed, and the resulting
    Δlog-viscosity is recorded at five shear rates.

    This analysis reveals:
    - Which concepts have the strongest influence on viscosity.
    - Whether concept sensitivity differs across shear rates (e.g. a concept
      that dominates at high shear but is inert at low shear has likely learned
      a shear-thinning correction).

    Outputs
    -------
    Saves ``concept_intervention.csv`` and prints a per-concept x shear-rate
    sensitivity summary to stdout.

    Parameters
    ----------
    model : ConceptBottleneckCNP (silently skipped for other types)
    samples : list[dict]
    device : torch.device
    physics_scaler_path : str
        Path to the saved ``physics_scaler.pkl``.
    save_dir : str
        Directory for output CSV.
    """
    if not isinstance(model, ConceptBottleneckCNP):
        print("Concept intervention requires ConceptBottleneckCNP — skipping.")
        return

    physics_scaler = joblib.load(physics_scaler_path)
    shear_mean = physics_scaler.mean_[0]
    shear_scale = physics_scaler.scale_[0]
    visc_mean = physics_scaler.mean_[1]
    visc_scale = physics_scaler.scale_[1]

    intervention_shears = {
        100.0: "100",
        1000.0: "1k",
        10000.0: "10k",
        100000.0: "100k",
        15000000.0: "15M",
    }
    shear_tensors: dict[str, torch.Tensor] = {}
    for shear_val, shear_label in intervention_shears.items():
        log_shear = np.log10(shear_val)
        shear_scaled = (log_shear - shear_mean) / shear_scale
        shear_tensors[shear_label] = torch.tensor([[[shear_scaled]]], dtype=torch.float32).to(
            device
        )

    model.eval()
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    records: list[dict] = []

    with torch.no_grad():
        for prot, task_samples in sorted(groups.items()):
            if len(task_samples) < 2 or prot in NON_PROTEIN_GROUPS:
                continue

            ctx_items = []
            for s in task_samples:
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_items.append(torch.cat([s["points"], stat], dim=1))
            ctx_t = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)

            q_static = (
                torch.stack([s["static"] for s in task_samples])
                .mean(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            c_base = model.encode_concepts(ctx_t)

            for shear_label, query_shear in shear_tensors.items():
                pred_base_sc = model.decode_from_concepts(c_base, query_shear, q_static)
                pred_base_lv = float(pred_base_sc.squeeze()) * visc_scale + visc_mean

                for ci, cname in enumerate(model.concept_names):
                    act_type = model._concept_activations[ci]
                    sweep = (
                        [0.0, 0.25, 0.5, 0.75, 1.0]
                        if act_type == "sigmoid"
                        else [-1.0, -0.5, 0.0, 0.5, 1.0]
                    )
                    for cval in sweep:
                        c_mod = c_base.clone()
                        c_mod[:, ci] = cval
                        pred_int_sc = model.decode_from_concepts(c_mod, query_shear, q_static)
                        pred_int_lv = float(pred_int_sc.squeeze()) * visc_scale + visc_mean
                        records.append(
                            {
                                "Group": prot,
                                "Shear_rate": shear_label,
                                "Concept": cname,
                                "Concept_idx": ci,
                                "Activation_type": act_type,
                                "Intervention_value": cval,
                                "Baseline_log_visc": pred_base_lv,
                                "Predicted_log_visc": pred_int_lv,
                                "Delta_log_visc": pred_int_lv - pred_base_lv,
                            }
                        )

    df_int = pd.DataFrame(records)
    save_path = os.path.join(save_dir, "concept_intervention.csv")
    df_int.to_csv(save_path, index=False)
    print(f"Concept intervention results saved to {save_path}")

    # Per-concept x shear-rate sensitivity summary
    print("\nMean |Δlog-visc| per concept x shear rate (intervention sensitivity):")
    pivot = (
        df_int.groupby(["Concept", "Shear_rate"])["Delta_log_visc"]
        .apply(lambda x: x.abs().mean())
        .unstack(fill_value=0.0)
    )
    shear_order = [lbl for lbl in intervention_shears.values() if lbl in pivot.columns]
    pivot = pivot.reindex(columns=shear_order)
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=False).drop(columns=["_total"])

    header = f"  {'Concept':<28}" + "".join(f"{s:>10}" for s in shear_order)
    print(header)
    print("  " + "-" * (28 + 10 * len(shear_order)))
    for cname, row in pivot.iterrows():
        vals = "".join(f"{row[s]:>10.4f}" for s in shear_order)
        print(f"  {cname:<28}{vals}")


# ============================================================
# Parity evaluation
# ============================================================


def run_parity_evaluation(
    model: torch.nn.Module,
    samples: list[dict],
    raw_df: pd.DataFrame,
    physics_scaler: object,
    device: torch.device,
    out: str,
) -> None:
    """
    Predict on all samples using leave-one-out context and evaluate parity.

    For each sample, every other sample in the same protein group is used as
    context. Predictions are made at the five canonical shear rates and
    compared against raw measurements from the original CSV.

    Outputs saved to ``out/``
    -------------------------
    parity_results.csv
        Per-measurement actual vs predicted viscosity and log-viscosity.
    parity_plot.png
        Colour-coded scatter plot with parity line, RMSE, and R² annotation.

    Parameters
    ----------
    model : ConceptBottleneckCNP
    samples : list[dict]
    raw_df : pd.DataFrame
        Original (unscaled) CSV used to retrieve true viscosity values.
    physics_scaler : sklearn StandardScaler
        Fitted on (log-shear, log-viscosity) to decode predictions.
    device : torch.device
    out : str
        Output directory.
    """
    shear_mean = physics_scaler.mean_[0]
    shear_scale = physics_scaler.scale_[0]
    visc_mean = physics_scaler.mean_[1]
    visc_scale = physics_scaler.scale_[1]

    parity_shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1000.0,
        "Viscosity_10000": 10000.0,
        "Viscosity_100000": 100000.0,
        "Viscosity_15000000": 1.5e7,
    }
    key_shears_eval = list(parity_shear_map.values())
    key_log_shears_eval = np.log10(key_shears_eval)

    scaled_log_shears = torch.tensor(
        [(ls - shear_mean) / shear_scale for ls in key_log_shears_eval],
        dtype=torch.float32,
    ).to(device)

    eval_groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        eval_groups[s["group"]].append(s)

    all_actual: list[float] = []
    all_predicted: list[float] = []
    all_eval_groups: list[str] = []
    all_sample_ids: list[str] = []
    all_shear_rates: list[float] = []
    n_shears = len(key_shears_eval)

    model.eval()
    with torch.no_grad():
        for sample in samples:
            sid = sample["id"]
            group = sample["group"]
            task_samples = eval_groups[group]

            ctx_samples = [s for s in task_samples if s["id"] != sid] or task_samples

            ctx_items = []
            for s in ctx_samples:
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_items.append(torch.cat([s["points"], stat], dim=1))
            ctx_tensor = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)

            q_shear = scaled_log_shears.view(1, n_shears, 1)
            q_static = sample["static"].unsqueeze(0).unsqueeze(0).repeat(1, n_shears, 1).to(device)

            memory = model.encode_memory(ctx_tensor)
            pred_sc = model.decode_from_memory(memory, q_shear, q_static)
            pred_sc = pred_sc.squeeze().cpu().numpy()
            pred_log = pred_sc * visc_scale + visc_mean
            pred_visc = 10.0**pred_log

            row_mask = raw_df["ID"] == sid
            if not row_mask.any():
                continue
            row = raw_df[row_mask].iloc[0]
            for i, (col, shear) in enumerate(parity_shear_map.items()):
                if col in raw_df.columns and pd.notna(row[col]) and row[col] > 0:
                    all_actual.append(float(row[col]))
                    all_predicted.append(float(pred_visc[i]))
                    all_eval_groups.append(group)
                    all_sample_ids.append(sid)
                    all_shear_rates.append(shear)

    all_actual_arr = np.array(all_actual)
    all_predicted_arr = np.array(all_predicted)
    all_eval_groups_a = np.array(all_eval_groups)
    all_shear_rates_a = np.array(all_shear_rates)

    log_actual = np.log10(np.clip(all_actual_arr, 1e-6, None))
    log_predicted = np.log10(np.clip(all_predicted_arr, 1e-6, None))

    ss_res = np.sum((log_actual - log_predicted) ** 2)
    ss_tot = np.sum((log_actual - log_actual.mean()) ** 2)
    rmse_log = float(np.sqrt(np.mean((log_actual - log_predicted) ** 2)))
    r2_log = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    print(f"\nOverall ({len(all_actual)} sample-shear pairs):")
    print(f"  RMSE (log10 viscosity): {rmse_log:.4f}")
    print(f"  R²   (log10 viscosity): {r2_log:.4f}")

    print("\nPer-group parity (RMSE in log10 space):")
    for g in sorted(set(all_eval_groups)):
        mask = all_eval_groups_a == g
        g_rmse = float(np.sqrt(np.mean((log_actual[mask] - log_predicted[mask]) ** 2)))
        print(f"  {g:28s}: RMSE={g_rmse:.4f}  (n={mask.sum()})")

    print("\nPer-shear-rate parity (RMSE in log10 space):")
    for shear in key_shears_eval:
        mask = all_shear_rates_a == shear
        if not mask.any():
            continue
        s_rmse = float(np.sqrt(np.mean((log_actual[mask] - log_predicted[mask]) ** 2)))
        print(f"  {shear:12.0f} s⁻¹: RMSE={s_rmse:.4f}  (n={mask.sum()})")

    # Save CSV
    parity_df = pd.DataFrame(
        {
            "ID": all_sample_ids,
            "Group": all_eval_groups,
            "Shear_Rate": all_shear_rates,
            "Actual_Viscosity": all_actual,
            "Predicted_Viscosity": all_predicted,
            "Log10_Actual": log_actual,
            "Log10_Predicted": log_predicted,
            "Log10_Error": log_predicted - log_actual,
        }
    )
    parity_csv_path = os.path.join(out, "parity_results.csv")
    parity_df.to_csv(parity_csv_path, index=False)
    print(f"\nParity results saved to {parity_csv_path}")

    # Save parity plot
    try:
        import matplotlib.cm as cm

        unique_groups_plot = sorted(set(all_eval_groups))
        cmap = cm.get_cmap("tab20", len(unique_groups_plot))
        color_map = {g: cmap(i) for i, g in enumerate(unique_groups_plot)}

        fig, ax = plt.subplots(figsize=(7, 7))
        for g in unique_groups_plot:
            mask = all_eval_groups_a == g
            ax.scatter(
                log_actual[mask],
                log_predicted[mask],
                color=color_map[g],
                label=g,
                alpha=0.65,
                s=20,
            )
        lims = [
            min(log_actual.min(), log_predicted.min()) - 0.1,
            max(log_actual.max(), log_predicted.max()) + 0.1,
        ]
        ax.plot(lims, lims, "k--", linewidth=1, label="Parity (y=x)")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_xlabel("log₁₀(Actual Viscosity)")
        ax.set_ylabel("log₁₀(Predicted Viscosity)")
        ax.set_title(
            f"Parity Plot — All Samples & Shear Rates\n" f"RMSE={rmse_log:.4f}, R²={r2_log:.4f}"
        )
        ax.legend(fontsize=7, markerscale=1.5, loc="upper left", ncol=2)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "parity_plot.png"), dpi=150)
        plt.close(fig)
        print(f"Parity plot saved to {os.path.join(out, 'parity_plot.png')}")
    except Exception as e:
        print(f"(Parity plot skipped: {e})")


# ============================================================
# Permutation feature importance
# ============================================================


def run_feature_importance(
    model: torch.nn.Module,
    samples: list[dict],
    raw_df: pd.DataFrame,
    preprocessor_path: str,
    static_dim: int,
    physics_scaler: object,
    parity_shear_map: dict[str, float],
    device: torch.device,
    out: str,
) -> None:
    """
    Permutation feature importance through the concept bottleneck pathway.

    For each static feature dimension, the column is randomly permuted across
    samples and the increase in decoder MSE (ΔMSE) is recorded. A large ΔMSE
    indicates an important feature.

    Because concept vectors are pre-computed once and the permutation only
    re-runs the cheap decoder, this is efficient even for high-dimensional
    feature vectors.

    Outputs saved to ``out/``
    -------------------------
    feature_importance.csv
        Per-feature ΔMSE, sorted descending.
    feature_importance_grouped.csv
        Per-feature-group ΔMSE (one-hot columns summed back to their source column).

    Parameters
    ----------
    model : ConceptBottleneckCNP
    samples : list[dict]
    raw_df : pd.DataFrame
        Original CSV for retrieving true viscosity values.
    preprocessor_path : str
        Path to saved ``preprocessor.pkl``.
    static_dim : int
        Number of static feature dimensions.
    physics_scaler : sklearn StandardScaler
        Fitted physics scaler.
    parity_shear_map : dict[str, float]
        Mapping of viscosity column name -> shear rate value.
    device : torch.device
    out : str
        Output directory.
    """
    visc_mean = physics_scaler.mean_[1]
    visc_scale = physics_scaler.scale_[1]
    shear_mean = physics_scaler.mean_[0]
    shear_scale = physics_scaler.scale_[0]

    key_shears_eval = list(parity_shear_map.values())
    key_log_shears = np.log10(key_shears_eval)
    scaled_log_shears = torch.tensor(
        [(ls - shear_mean) / shear_scale for ls in key_log_shears],
        dtype=torch.float32,
    ).to(device)
    n_shears = len(key_shears_eval)

    preprocessor_fi = joblib.load(preprocessor_path)
    try:
        feature_names_fi = list(preprocessor_fi.get_feature_names_out())
    except Exception:
        feature_names_fi = [f"feature_{i}" for i in range(static_dim)]

    eval_groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        eval_groups[s["group"]].append(s)

    fi_ctx_tensors: list[torch.Tensor] = []
    fi_static_vecs: list[torch.Tensor] = []
    fi_true_log_visc: list[list[float]] = []
    fi_valid_masks: list[list[bool]] = []

    model.eval()
    with torch.no_grad():
        for sample in samples:
            sid = sample["id"]
            group = sample["group"]
            task_samples = eval_groups[group]
            ctx_samples = [s for s in task_samples if s["id"] != sid] or task_samples

            ctx_items = []
            for s in ctx_samples:
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_items.append(torch.cat([s["points"], stat], dim=1))
            fi_ctx_tensors.append(torch.cat(ctx_items, dim=0).unsqueeze(0).to(device))
            fi_static_vecs.append(sample["static"])

            row_mask = raw_df["ID"] == sid
            true_lv, valid = [0.0] * 5, [False] * 5
            if row_mask.any():
                row_fi = raw_df[row_mask].iloc[0]
                for j, col in enumerate(parity_shear_map):
                    if col in raw_df.columns and pd.notna(row_fi[col]) and row_fi[col] > 0:
                        true_lv[j] = np.log10(float(row_fi[col]))
                        valid[j] = True
            fi_true_log_visc.append(true_lv)
            fi_valid_masks.append(valid)

    fi_static_matrix = torch.stack(fi_static_vecs)
    fi_true_log_visc_a = np.array(fi_true_log_visc)
    fi_valid_masks_a = np.array(fi_valid_masks)
    q_shear_fi = scaled_log_shears.view(1, n_shears, 1)

    # Pre-compute concept vectors once — permutation only re-runs the decoder
    fi_memory_list: list[torch.Tensor] = []
    with torch.no_grad():
        for ctx_t in fi_ctx_tensors:
            fi_memory_list.append(model.encode_memory(ctx_t))

    def _decoder_mse(static_mat: torch.Tensor) -> float:
        errs: list[float] = []
        with torch.no_grad():
            for i, (mem, true_lv, valid) in enumerate(
                zip(fi_memory_list, fi_true_log_visc_a, fi_valid_masks_a, strict=False)
            ):
                if not any(valid):
                    continue
                q_st = static_mat[i].unsqueeze(0).unsqueeze(0).repeat(1, n_shears, 1).to(device)
                pred_sc = model.decode_from_memory(mem, q_shear_fi, q_st).squeeze().cpu().numpy()
                pred_lv = pred_sc * visc_scale + visc_mean
                for j in range(5):
                    if valid[j]:
                        errs.append((pred_lv[j] - true_lv[j]) ** 2)
        return float(np.mean(errs)) if errs else float("nan")

    baseline_fi_mse = _decoder_mse(fi_static_matrix)
    print(f"Baseline decoder MSE (log10 viscosity): {baseline_fi_mse:.6f}")
    print(f"Permuting {static_dim} features across {len(samples)} samples...")

    fi_importances = np.zeros(static_dim)
    for j in range(static_dim):
        perm = fi_static_matrix.clone()
        perm[:, j] = fi_static_matrix[torch.randperm(len(samples)), j]
        fi_importances[j] = _decoder_mse(perm) - baseline_fi_mse

    ranked_idx = np.argsort(-fi_importances)
    print("\nTop 20 most important features (individual):")
    print(f"  {'Feature':<55} {'ΔMSE':>10}")
    print(f"  {'-'*55} {'-'*10}")
    for k in ranked_idx[:20]:
        fname = feature_names_fi[k] if k < len(feature_names_fi) else f"feature_{k}"
        print(f"  {fname:<55} {fi_importances[k]:>10.6f}")

    # Grouped rollup: sum one-hot columns back to their source column name
    cat_cols_fi = [
        "Protein_type",
        "Protein_class_type",
        "Buffer_type",
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
        "Excipient_type",
    ]
    grouped_imp: dict[str, float] = defaultdict(float)
    for k, imp in enumerate(fi_importances):
        fname = feature_names_fi[k] if k < len(feature_names_fi) else f"feature_{k}"
        if fname.startswith("cat__"):
            rest = fname[5:]
            matched = next(
                (col for col in cat_cols_fi if rest.startswith(col + "_") or rest == col),
                None,
            )
            grouped_imp[matched if matched else rest] += imp
        elif fname.startswith("num__"):
            grouped_imp[fname[5:]] += imp
        else:
            grouped_imp[fname] += imp

    grouped_ranked = sorted(grouped_imp.items(), key=lambda x: -x[1])
    print("\nGrouped feature importance (categoricals summed by column):")
    print(f"  {'Feature':<45} {'ΔMSE':>10}")
    print(f"  {'-'*45} {'-'*10}")
    for fname, imp in grouped_ranked:
        print(f"  {fname:<45} {imp:>10.6f}")

    fi_df = pd.DataFrame(
        {
            "Feature": [
                feature_names_fi[k] if k < len(feature_names_fi) else f"feature_{k}"
                for k in range(static_dim)
            ],
            "Importance_dMSE": fi_importances,
        }
    ).sort_values("Importance_dMSE", ascending=False)
    fi_df.to_csv(os.path.join(out, "feature_importance.csv"), index=False)

    fi_grp_df = pd.DataFrame(grouped_ranked, columns=["Feature_Group", "Importance_dMSE"])
    fi_grp_df.to_csv(os.path.join(out, "feature_importance_grouped.csv"), index=False)
    print(f"\nFeature importance saved to {out}/feature_importance*.csv")
