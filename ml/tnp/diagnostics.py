"""
diagnostics.py
==============
Diagnostic utilities for monitoring TransformerNP training health.  [TNP-4]

Public API
----------
log_latent_variance(model, samples, device) -> float
    Inter-group mean L2 distance of mean-pooled context encodings.
    Low value (< 0.2 after epoch 30) may indicate context collapse.

log_attention_stats(model, samples, device, n_groups=6)
    -> (mean_entropy, mean_w_std, mean_h_max)
    Per-query Shannon entropy, weight std, and theoretical entropy ceiling.
    After [TNP-ATTN-4] the ceiling is ln(N_ctx_samples), not ln(N_pts).
    Healthy target: mean_entropy / mean_h_max < 0.5  (model is focused).

save_attention_heatmap(model, samples, device, save_path, n_proteins=6) -> None
    Protein x context-size entropy heatmap.  Values should decrease as k grows
    once sample-level aggregation is in effect.
"""

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from constants import NON_PROTEIN_GROUPS
from model import TransformerNP, _encode_latent, _forward
from training import _build_ctx_encoded, _build_ctx_tensor, _build_tgt_tensors

# ---------------------------------------------------------------------------
# Latent variance
# ---------------------------------------------------------------------------


def log_latent_variance(
    model: TransformerNP,
    samples: list,
    device: torch.device,
) -> float:
    """
    Compute mean pairwise L2 distance between per-group mean-pooled encodings.

    Uses raw _build_ctx_tensor -> encode_latent_mean (point-level pooling).
    This is intentional: latent variance measures whether groups occupy distinct
    regions of the encoding space, which is a property of the encoder MLP and
    does not depend on the cross-attention path.

    Args:
        model:   TransformerNP (switched to eval internally).
        samples: Full list of sample dicts.
        device:  Computation device.

    Returns:
        Mean pairwise L2 distance; 0.0 if fewer than 2 qualifying groups.
    """
    model.eval()
    groups: dict[str, list] = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    group_r: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for prot, task_samples in groups.items():
            if len(task_samples) < 2 or prot in NON_PROTEIN_GROUPS:
                continue
            idx = np.random.permutation(len(task_samples))[: min(5, len(task_samples))]
            ctx_t = _build_ctx_tensor(task_samples, idx, device)
            r = _encode_latent(model, ctx_t).squeeze(0).cpu().numpy()
            group_r[prot] = r

    if len(group_r) < 2:
        return 0.0

    vecs = np.stack(list(group_r.values()))
    dists = [
        np.linalg.norm(vecs[i] - vecs[j])
        for i in range(len(vecs))
        for j in range(i + 1, len(vecs))
    ]
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Attention statistics
# ---------------------------------------------------------------------------


def log_attention_stats(
    model: TransformerNP,
    samples: list,
    device: torch.device,
    n_groups: int = 6,
) -> tuple:
    """
    Compute mean cross-attention entropy, weight std, and entropy ceiling.

    Uses _build_ctx_encoded() so the context has one token per sample [TNP-ATTN-4].
    The entropy ceiling is now ln(N_ctx_samples) ~= ln(half the group size),
    not ln(N_pts_total).  The third return value is this ceiling so callers
    can compute the fractional saturation without hard-coding a token count.

    Interpretation (after sample-level fix):
      entropy / h_max < 0.50  -> focused attention (ideal)
      entropy / h_max > 0.85  -> near-uniform weights; model still average-pooling

    Args:
        model:    TransformerNP instance.
        samples:  Full list of sample dicts.
        device:   Computation device.
        n_groups: Maximum number of protein groups to evaluate.

    Returns:
        (mean_entropy, mean_weight_std, mean_h_max) -- all floats.
    """
    model.eval()
    groups: dict[str, list] = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    protein_groups = [
        g for g, sl in groups.items() if len(sl) >= 4 and g not in NON_PROTEIN_GROUPS
    ][:n_groups]

    all_entropy: list[float] = []
    all_w_std: list[float] = []
    all_h_max: list[float] = []

    with torch.no_grad():
        for prot in protein_groups:
            task_samples = groups[prot]
            idx = np.random.permutation(len(task_samples))
            mid = max(1, len(idx) // 2)
            ctx_idx = idx[:mid]
            tgt_idx = idx[mid:]
            if len(tgt_idx) == 0:
                continue

            # [TNP-ATTN-4] Sample-level encoded context: [1, mid, latent_dim]
            ctx_t = _build_ctx_encoded(model, task_samples, ctx_idx, device)
            n_ctx_samples = ctx_t.shape[1]

            tgt_shear, tgt_stat = [], []
            for i in tgt_idx:
                s = task_samples[i]
                n = s["points"].shape[0]
                tgt_shear.append(s["points"][:, [0]])
                tgt_stat.append(s["static"].unsqueeze(0).repeat(n, 1))
            q_x = torch.cat(tgt_shear, dim=0).unsqueeze(0).to(device)
            q_stat = torch.cat(tgt_stat, dim=0).unsqueeze(0).to(device)

            _, attn_weights = _forward(model, ctx_t, q_x, q_stat, ctx_is_encoded=True)
            # attn_weights: [B, N_q, N_ctx_samples]
            w = attn_weights.squeeze(0).cpu().numpy()  # [N_q, N_ctx_samples]

            eps = 1e-8
            entropy = -np.sum(w * np.log(w + eps), axis=-1)  # [N_q]
            h_max = np.log(n_ctx_samples)  # theoretical ceiling for this draw

            all_entropy.extend(entropy.tolist())
            all_w_std.extend(w.std(axis=-1).tolist())
            all_h_max.extend([h_max] * len(entropy))

    mean_entropy = float(np.mean(all_entropy)) if all_entropy else 0.0
    mean_w_std = float(np.mean(all_w_std)) if all_w_std else 0.0
    mean_h_max = float(np.mean(all_h_max)) if all_h_max else 1.0
    return mean_entropy, mean_w_std, mean_h_max


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------


def save_attention_heatmap(
    model: TransformerNP,
    samples: list,
    device: torch.device,
    save_path: str,
    n_proteins: int = 6,
) -> None:
    """
    Save a protein x context-size entropy heatmap.  [TNP-4]

    After [TNP-ATTN-4], context sizes (k=2,4,8,12) refer to the number of
    sample-level tokens.  A well-functioning model shows entropy decreasing
    (or flat at low values) as k grows — the model focuses more sharply when
    it has more samples to compare against.

    The colour scale is calibrated to [0, ln(k_max)] so values near zero
    (focused) are green and values near the ceiling are red.

    Args:
        model:      TransformerNP instance.
        samples:    Full list of sample dicts.
        device:     Computation device.
        save_path:  Output PNG file path.
        n_proteins: Maximum number of protein groups to include.
    """
    model.eval()
    groups: dict[str, list] = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    protein_list = sorted(
        [g for g, sl in groups.items() if len(sl) >= 4 and g not in NON_PROTEIN_GROUPS]
    )[:n_proteins]

    ctx_sizes = [2, 4, 8, 12]
    entropy_matrix = np.zeros((len(protein_list), len(ctx_sizes)))
    # Store each cell's theoretical ceiling for annotation
    ceiling_matrix = np.zeros_like(entropy_matrix)

    with torch.no_grad():
        for pi, prot in enumerate(protein_list):
            task_samples = groups[prot]
            for ci, k in enumerate(ctx_sizes):
                k_eff = min(k, len(task_samples) - 1)
                if k_eff < 1:
                    continue
                idx_c = np.random.choice(len(task_samples), k_eff, replace=False)
                remaining = [i for i in range(len(task_samples)) if i not in set(idx_c)]
                if not remaining:
                    continue
                idx_q = np.random.choice(
                    remaining, min(4, len(remaining)), replace=False
                )

                # [TNP-ATTN-4] Sample-level encoded context
                ctx_t = _build_ctx_encoded(model, task_samples, idx_c, device)
                n_ctx_samples = ctx_t.shape[1]
                ceiling_matrix[pi, ci] = np.log(n_ctx_samples)

                q_shear, q_stat, _ = _build_tgt_tensors(task_samples, idx_q, device)
                if q_shear is None:
                    continue

                _, attn_weights = _forward(
                    model, ctx_t, q_shear, q_stat, ctx_is_encoded=True
                )
                w = attn_weights.squeeze(0).cpu().numpy()
                eps = 1e-8
                entropy = -np.sum(w * np.log(w + eps), axis=-1)
                entropy_matrix[pi, ci] = entropy.mean()

    # Normalise by ceiling so colours are comparable across different k values
    frac_matrix = np.where(ceiling_matrix > 0, entropy_matrix / ceiling_matrix, 0.0)

    fig, ax = plt.subplots(figsize=(7, max(3, len(protein_list) * 0.45)))
    im = ax.imshow(frac_matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(ctx_sizes)))
    ax.set_xticklabels([f"k={k}" for k in ctx_sizes], fontsize=9)
    ax.set_yticks(range(len(protein_list)))
    ax.set_yticklabels(protein_list, fontsize=9)
    plt.colorbar(im, ax=ax, label="Entropy / ceiling (0=focused, 1=uniform)")
    ax.set_title(
        "Attention entropy / ceiling  (protein group x context size)", fontsize=11
    )

    for pi in range(len(protein_list)):
        for ci in range(len(ctx_sizes)):
            h = entropy_matrix[pi, ci]
            c = ceiling_matrix[pi, ci]
            label = f"{h:.2f}\n/{c:.2f}" if c > 0 else "—"
            ax.text(
                ci,
                pi,
                label,
                ha="center",
                va="center",
                fontsize=6,
                color=("white" if frac_matrix[pi, ci] > 0.6 else "black"),
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Attention entropy heatmap saved to {save_path}")
