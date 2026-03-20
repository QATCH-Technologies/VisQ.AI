"""
evaluation.py
=============
Post-training evaluation routines for TransformerNP.

Public API
----------
run_parity_evaluation(final_model, samples, data_csv, out_dir, device)
    -> None
    Leave-one-out parity pass at 5 canonical shear rates.
    Saves: parity_results.csv, parity_plot.png

run_feature_importance(final_model, samples, raw_df, out_dir, device,
                       static_dim, scaled_log_shears, visc_mean, visc_scale,
                       n_shears, parity_shear_map)
    -> None
    Decoder-side permutation importance via pre-computed context encodings.
    Saves: feature_importance.csv, feature_importance_grouped.csv

[TNP-ATTN-4] Both routines now use encode_context_samples() to build one
latent token per context sample rather than concatenating all points into a
single long sequence.  This is consistent with the training forward pass and
ensures the parity metrics reflect the model's actual inference behaviour.
"""

import os
from collections import defaultdict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tnp.model import TransformerNP

# ---------------------------------------------------------------------------
# Internal helper: build sample-level context encoding
# ---------------------------------------------------------------------------


def _encode_ctx_samples(
    model: TransformerNP, sample_list: list, device: torch.device
) -> torch.Tensor:
    """
    Encode a list of sample dicts into [1, N_samples, latent_dim].

    Each sample's dense curve points are encoded independently and mean-pooled
    to one latent token.  This is the evaluation-time equivalent of
    _build_ctx_encoded() in training.py.
    """
    ctx_items_list = []
    for s in sample_list:
        stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
        raw = torch.cat([s["points"], stat], dim=1).to(device)
        ctx_items_list.append(raw)
    return model.encode_context_samples(ctx_items_list)


# ---------------------------------------------------------------------------
# Parity evaluation
# ---------------------------------------------------------------------------


def run_parity_evaluation(
    final_model: TransformerNP,
    samples: list,
    data_csv: str,
    out_dir: str,
    device: torch.device,
) -> None:
    """
    Leave-one-out parity evaluation at five canonical shear rates.

    For each sample, the remaining members of its protein group form the
    context set.  The context is encoded at sample level [TNP-ATTN-4] before
    being passed to decode_from_memory(), matching the training forward pass.

    Args:
        final_model: Trained TransformerNP (set to eval mode internally).
        samples:     Full preprocessed sample list.
        data_csv:    Path to the raw formulation CSV.
        out_dir:     Directory where artefacts are written.
        device:      Computation device.
    """
    print("\n" + "=" * 60)
    print("PARITY EVALUATION")
    print("=" * 60)
    print(f"Data: {data_csv}")

    physics_scaler = joblib.load(os.path.join(out_dir, "physics_scaler.pkl"))
    raw_df = pd.read_csv(data_csv)

    parity_shear_map = {
        "Viscosity_100": 100.0,
        "Viscosity_1000": 1_000.0,
        "Viscosity_10000": 10_000.0,
        "Viscosity_100000": 100_000.0,
        "Viscosity_15000000": 1.5e7,
    }
    key_shears_eval = list(parity_shear_map.values())
    key_log_shears_eval = np.log10(key_shears_eval)
    n_shears = len(key_shears_eval)

    shear_mean = physics_scaler.mean_[0]
    shear_scale = physics_scaler.scale_[0]
    visc_mean = physics_scaler.mean_[1]
    visc_scale = physics_scaler.scale_[1]

    scaled_log_shears = torch.tensor(
        [(ls - shear_mean) / shear_scale for ls in key_log_shears_eval],
        dtype=torch.float32,
    ).to(device)

    eval_groups: dict[str, list] = defaultdict(list)
    for s in samples:
        eval_groups[s["group"]].append(s)

    all_actual, all_predicted = [], []
    all_eval_groups, all_sample_ids, all_shear_rates = [], [], []

    final_model.eval()
    with torch.no_grad():
        for sample in samples:
            sid = sample["id"]
            group = sample["group"]
            task_samples = eval_groups[group]

            ctx_samples = [s for s in task_samples if s["id"] != sid] or task_samples

            # [TNP-ATTN-4] One token per context sample (uses s["static"] — full features)
            ctx_enc = _encode_ctx_samples(final_model, ctx_samples, device)

            q_shear = scaled_log_shears.view(1, n_shears, 1)
            # [TNP-ATTN-6] Query uses reduced static (no protein identity)
            q_static = (
                sample["static_qry"]
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, n_shears, 1)
                .to(device)
            )

            pred_sc = final_model.decode_from_memory(ctx_enc, q_shear, q_static)
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

    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    all_eval_groups = np.array(all_eval_groups)
    all_shear_rates = np.array(all_shear_rates)

    log_actual = np.log10(np.clip(all_actual, 1e-6, None))
    log_predicted = np.log10(np.clip(all_predicted, 1e-6, None))

    ss_res = np.sum((log_actual - log_predicted) ** 2)
    ss_tot = np.sum((log_actual - log_actual.mean()) ** 2)
    rmse_log = float(np.sqrt(np.mean((log_actual - log_predicted) ** 2)))
    r2_log = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    print(f"\nOverall ({len(all_actual)} sample-shear pairs):")
    print(f"  RMSE (log10 viscosity): {rmse_log:.4f}")
    print(f"  R2   (log10 viscosity): {r2_log:.4f}")

    print("\nPer-group parity (RMSE in log10 space):")
    for g in sorted(set(all_eval_groups)):
        mask = all_eval_groups == g
        g_rmse = float(np.sqrt(np.mean((log_actual[mask] - log_predicted[mask]) ** 2)))
        print(f"  {g:28s}: RMSE={g_rmse:.4f}  (n={mask.sum()})")

    print("\nPer-shear-rate parity (RMSE in log10 space):")
    for shear in key_shears_eval:
        mask = all_shear_rates == shear
        if not mask.any():
            continue
        s_rmse = float(np.sqrt(np.mean((log_actual[mask] - log_predicted[mask]) ** 2)))
        print(f"  {shear:12.0f} s-1: RMSE={s_rmse:.4f}  (n={mask.sum()})")

    # ---- CSV ----
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
    parity_df.to_csv(os.path.join(out_dir, "parity_results.csv"), index=False)
    print(f"\nParity results saved to {os.path.join(out_dir, 'parity_results.csv')}")

    # ---- Scatter plot ----
    try:
        import matplotlib.cm as cm

        unique_groups_plot = sorted(set(all_eval_groups))
        cmap = cm.get_cmap("tab20", len(unique_groups_plot))
        color_map = {g: cmap(i) for i, g in enumerate(unique_groups_plot)}

        fig, ax = plt.subplots(figsize=(7, 7))
        for g in unique_groups_plot:
            mask = all_eval_groups == g
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
        ax.set_xlabel("log10(Actual Viscosity)")
        ax.set_ylabel("log10(Predicted Viscosity)")
        ax.set_title(
            f"TNP Parity — All Samples & Shear Rates\n"
            f"RMSE={rmse_log:.4f}, R2={r2_log:.4f}"
        )
        ax.legend(fontsize=7, markerscale=1.5, loc="upper left", ncol=2)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "parity_plot.png"), dpi=150)
        plt.close(fig)
        print("Parity plot saved.")
    except Exception as e:
        print(f"(Parity plot skipped: {e})")


# ---------------------------------------------------------------------------
# Feature importance (decoder-side permutation)
# ---------------------------------------------------------------------------


def run_feature_importance(
    final_model: TransformerNP,
    samples: list,
    raw_df: pd.DataFrame,
    out_dir: str,
    device: torch.device,
    static_dim: int,
    scaled_log_shears: torch.Tensor,
    visc_mean: float,
    visc_scale: float,
    n_shears: int,
    parity_shear_map: dict,
) -> None:
    """
    Decoder-side permutation feature importance for TransformerNP.

    Context encodings are pre-computed once per sample using encode_context_samples()
    [TNP-ATTN-4], giving one token per context sample.  Permutation only re-runs
    the cross-attention + decoder path.

    Args:
        final_model:        Trained TransformerNP.
        samples:            Full preprocessed sample list.
        raw_df:             Raw formulation DataFrame.
        out_dir:            Directory where artefacts are written.
        device:             Computation device.
        static_dim:         Number of static features.
        scaled_log_shears:  [n_shears] tensor of pre-scaled log shear rates.
        visc_mean:          Physics scaler mean for viscosity dimension.
        visc_scale:         Physics scaler scale for viscosity dimension.
        n_shears:           Number of canonical shear rates.
        parity_shear_map:   {viscosity_col: shear_rate} mapping.
    """
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Permutation, Decoder + Cross-Attention Pathway)")
    print("=" * 60)

    preprocessor = joblib.load(os.path.join(out_dir, "preprocessor.pkl"))
    # Feature importance permutes the QUERY static — load query preprocessor for names
    qry_preprocessor_path = os.path.join(out_dir, "query_preprocessor.pkl")
    if os.path.exists(qry_preprocessor_path):
        qry_preprocessor = joblib.load(qry_preprocessor_path)
        try:
            feature_names = list(qry_preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"feature_{i}" for i in range(static_dim)]
    else:
        # Fallback: use full preprocessor names (old checkpoint without split static)
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"feature_{i}" for i in range(static_dim)]

    eval_groups: dict[str, list] = defaultdict(list)
    for s in samples:
        eval_groups[s["group"]].append(s)

    fi_ctx_encodings: list = []
    fi_static_vecs: list = []
    fi_true_log_visc: list = []
    fi_valid_masks: list = []

    final_model.eval()
    with torch.no_grad():
        for sample in samples:
            sid = sample["id"]
            group = sample["group"]
            task_samples = eval_groups[group]
            ctx_samples = [s for s in task_samples if s["id"] != sid] or task_samples

            # [TNP-ATTN-4] Sample-level encoding: [1, N_ctx_samples, latent_dim]
            ctx_enc = _encode_ctx_samples(final_model, ctx_samples, device)
            fi_ctx_encodings.append(ctx_enc)
            fi_static_vecs.append(
                sample["static_qry"]
            )  # [TNP-ATTN-6] permute query features

            row_mask = raw_df["ID"] == sid
            true_lv, valid = [0.0] * 5, [False] * 5
            if row_mask.any():
                row_fi = raw_df[row_mask].iloc[0]
                for j, col in enumerate(parity_shear_map):
                    if (
                        col in raw_df.columns
                        and pd.notna(row_fi[col])
                        and row_fi[col] > 0
                    ):
                        true_lv[j] = np.log10(float(row_fi[col]))
                        valid[j] = True
            fi_true_log_visc.append(true_lv)
            fi_valid_masks.append(valid)

    fi_static_matrix = torch.stack(fi_static_vecs)  # [N, static_qry_dim]
    fi_true_log_visc = np.array(fi_true_log_visc)  # [N, 5]
    fi_valid_masks = np.array(fi_valid_masks)  # [N, 5]
    q_shear_fi = scaled_log_shears.view(1, n_shears, 1)
    static_qry_dim = fi_static_matrix.shape[1]  # may differ from static_dim (ctx)

    def _full_mse(static_mat: torch.Tensor) -> float:
        """MSE with pre-computed ctx encodings; re-runs cross-attn for permuted static."""
        errs = []
        with torch.no_grad():
            for i, (ctx_enc, true_lv, valid) in enumerate(
                zip(fi_ctx_encodings, fi_true_log_visc, fi_valid_masks, strict=False)
            ):
                if not any(valid):
                    continue
                q_st = (
                    static_mat[i]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(1, n_shears, 1)
                    .to(device)
                )
                pred_sc = final_model.decode_from_memory(ctx_enc, q_shear_fi, q_st)
                pred_lv = pred_sc.squeeze().cpu().numpy() * visc_scale + visc_mean
                for j in range(5):
                    if valid[j]:
                        errs.append((pred_lv[j] - true_lv[j]) ** 2)
        return float(np.mean(errs)) if errs else float("nan")

    baseline_mse = _full_mse(fi_static_matrix)
    print(f"Baseline decoder MSE (log10 viscosity): {baseline_mse:.6f}")
    print(f"Permuting {static_qry_dim} query features across {len(samples)} samples...")

    importances = np.zeros(static_qry_dim)
    for j in range(static_qry_dim):
        perm = fi_static_matrix.clone()
        perm[:, j] = fi_static_matrix[torch.randperm(len(samples)), j]
        importances[j] = _full_mse(perm) - baseline_mse

    ranked_idx = np.argsort(-importances)
    print("\nTop 20 most important query features:")
    print(f"  {'Feature':<55} {'dMSE':>10}")
    print(f"  {'-'*55} {'-'*10}")
    for k in ranked_idx[:20]:
        fname = feature_names[k] if k < len(feature_names) else f"feature_{k}"
        print(f"  {fname:<55} {importances[k]:>10.6f}")

    # ---- Grouped importance ----
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
    for k, imp in enumerate(importances):
        fname = feature_names[k] if k < len(feature_names) else f"feature_{k}"
        if fname.startswith("cat__"):
            rest = fname[5:]
            matched = next(
                (
                    col
                    for col in cat_cols_fi
                    if rest.startswith(col + "_") or rest == col
                ),
                None,
            )
            grouped_imp[matched if matched else rest] += imp
        elif fname.startswith("num__"):
            grouped_imp[fname[5:]] += imp
        else:
            grouped_imp[fname] += imp

    grouped_ranked = sorted(grouped_imp.items(), key=lambda x: -x[1])
    print("\nGrouped feature importance:")
    print(f"  {'Feature':<45} {'dMSE':>10}")
    print(f"  {'-'*45} {'-'*10}")
    for fname, imp in grouped_ranked:
        print(f"  {fname:<45} {imp:>10.6f}")

    # ---- Save CSVs ----
    pd.DataFrame(
        {
            "Feature": [
                feature_names[k] if k < len(feature_names) else f"feature_{k}"
                for k in range(static_qry_dim)
            ],
            "Importance_dMSE": importances,
        }
    ).sort_values("Importance_dMSE", ascending=False).to_csv(
        os.path.join(out_dir, "feature_importance.csv"), index=False
    )
    pd.DataFrame(grouped_ranked, columns=["Feature_Group", "Importance_dMSE"]).to_csv(
        os.path.join(out_dir, "feature_importance_grouped.csv"), index=False
    )
    print(f"\nFeature importance saved to {out_dir}/feature_importance*.csv")
