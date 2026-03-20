"""
training.py
===========
Training loop, validation, and tensor-building utilities for TransformerNP.

Public API
----------
_build_ctx_tensor(task_samples, indices, device)
    -> Tensor [1, N_pts, 2+static_dim]
    Raw point-level concatenation.  Used ONLY for triplet / consistency losses
    that call encode_latent_mean() directly.

_build_ctx_encoded(model, task_samples, indices, device)
    -> Tensor [1, N_samples, latent_dim]   [TNP-ATTN-4]
    Sample-level encoding: each sample's points are encoded + mean-pooled to
    one token before cross-attention.  Use this everywhere a context tensor
    is passed to forward() or _forward().

_build_tgt_tensors(task_samples, indices, device)
    -> (q_x, q_stat, q_y) | (None, None, None)

train_epoch(model, samples, optimizer, device, **kwargs)
    -> (avg_loss, per_group_mse)

validate(model, samples, device, n_repeats=3) -> float

Loss components
---------------
  mse_loss         -- primary prediction objective (MSE in scaled log space)
  utility_loss     -- impostor-context baseline penalty   [FIX-5 / TNP-ATTN-5]
  norm_penalty     -- soft L2 cap on mean-pooled r         [FIX-NORM]
  consistency_loss -- cosine alignment for two halves      [FIX-4]
  triplet_loss     -- hard-negative mining on r            [FIX-3]

Mixed-context strategy  [TNP-ATTN-5]
--------------------------------------
40% of training iterations use a cross-protein mixed context: half the context
slots are genuine anchor-protein samples, half are "impostor" samples drawn from
a randomly selected different protein.  Targets are always anchor-protein only.

This forces the attention to discriminate — the model must upweight anchor tokens
and downweight impostors to minimise prediction error.  Without this signal, all
context tokens come from the same protein and mean-pooling is the optimal strategy,
keeping attention permanently at H/H_max ≈ 1.0.

The utility loss baseline in mixed iterations uses the impostor-only context rather
than a zero token.  This measures "how much does the anchor context help beyond
having only irrelevant context?" — a much sharper training signal than zeros.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tnp.constants import NON_PROTEIN_GROUPS, PROTEIN_CLASS_MAP
from tnp.model import TransformerNP, _encode_latent, _forward

# ---------------------------------------------------------------------------
# Tensor builders
# ---------------------------------------------------------------------------


def _build_ctx_tensor(
    task_samples: list,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Concatenate context sample points into a single raw batched tensor.

    Returns [1, N_total_points, 2+static_dim].

    NOTE: Do NOT pass this to forward() or _forward() directly — it produces
    one attention token per dense-curve point, inflating the context sequence
    to ~N_samples x 40 tokens and collapsing attention to near-uniform weights.
    Use _build_ctx_encoded() for forward passes.
    This function is kept for encode_latent_mean() calls (triplet / consistency).
    """
    ctx_items = []
    for i in indices:
        s = task_samples[i]
        stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
        ctx_items.append(torch.cat([s["points"], stat], dim=1))
    return torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)


def _build_ctx_encoded(
    model: TransformerNP,
    task_samples: list,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a sample-level encoded context tensor.  [TNP-ATTN-4]

    Each context sample's dense curve points are encoded independently through
    the context MLP and mean-pooled to one latent token.  The result has shape
    [1, N_samples, latent_dim] — one token per sample, not one per point.

    This reduces the cross-attention sequence from ~N_samples x 40 tokens to
    N_samples tokens, lowering the entropy ceiling from ln(320) to ln(N_samples)
    and allowing the model to attend selectively.

    Always pass ctx_is_encoded=True when using this output with forward() or
    _forward() — the encoding step has already been applied.

    Args:
        model:        TransformerNP (in train or eval mode, preserved as-is).
        task_samples: List of sample dicts for one protein group.
        indices:      Indices into task_samples to use as context.
        device:       Target device.

    Returns:
        [1, N_samples, latent_dim] — LayerNorm'd, ready for cross-attention.
    """
    ctx_items_list = []
    for i in indices:
        s = task_samples[i]
        stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
        raw = torch.cat([s["points"], stat], dim=1).to(device)
        ctx_items_list.append(raw)
    return model.encode_context_samples(ctx_items_list)


def _build_tgt_tensors(
    task_samples: list,
    indices: np.ndarray,
    device: torch.device,
):
    """
    Build query input tensors and ground-truth viscosity for target samples.

    Uses s["static_qry"] — the reduced static vector without protein identity.
    [TNP-ATTN-6]

    Returns:
        q_x    : [1, N_total_pts, 1]
        q_stat : [1, N_total_pts, static_qry_dim]
        q_y    : [1, N_total_pts, 1]
        or (None, None, None) if indices is empty.
    """
    shear_list, y_list, stat_list = [], [], []
    for i in indices:
        s = task_samples[i]
        n = s["points"].shape[0]
        shear_list.append(s["points"][:, [0]])
        y_list.append(s["points"][:, [1]])
        stat_list.append(s["static_qry"].unsqueeze(0).repeat(n, 1))
    if not shear_list:
        return None, None, None
    q_x = torch.cat(shear_list, dim=0).unsqueeze(0).to(device)
    q_stat = torch.cat(stat_list, dim=0).unsqueeze(0).to(device)
    q_y = torch.cat(y_list, dim=0).unsqueeze(0).to(device)
    return q_x, q_stat, q_y


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------


def train_epoch(
    model: TransformerNP,
    samples: list,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    iterations: int = 100,
    group_weights: dict | None = None,
    lambda_triplet: float = 0.10,
    lambda_consistency: float = 0.10,
    lambda_utility: float = 2.5,  # Raised from 1.0; impostor baseline is sharper
    triplet_margin: float = 3.0,
    lambda_norm: float = 0.05,
    norm_target: float = 5.0,
    mixed_ctx_prob: float = 0.40,  # [TNP-ATTN-5] Fraction of iters using cross-protein context
):
    """
    Run one training epoch for TransformerNP.

    Each iteration samples a random protein group (weighted by group_weights),
    splits its samples into context / target, and computes the combined loss.

    [TNP-ATTN-4] Context tensors for the forward pass are built with
    _build_ctx_encoded(), producing one latent token per context sample.

    [TNP-ATTN-5] Mixed-context iterations: with probability mixed_ctx_prob the
    context is a blend of anchor-protein samples and impostor samples from a
    different protein.  Targets are always anchor-protein only.  The utility
    loss baseline in these iterations uses the impostor-only context (not zeros),
    measuring the specific improvement from attending to relevant samples.

    Args:
        model:              TransformerNP instance.
        samples:            Full list of sample dicts.
        optimizer:          PyTorch optimizer.
        device:             Computation device.
        iterations:         Number of random task draws per epoch.
        group_weights:      Per-group EMA difficulty weights; uniform if None.
        lambda_triplet:     Weight for the triplet contrastive loss  [FIX-3].
        lambda_consistency: Weight for the intra-group cosine loss   [FIX-4].
        lambda_utility:     Weight for the impostor-context penalty  [FIX-5 / TNP-ATTN-5].
        triplet_margin:     Margin for the triplet loss.
        lambda_norm:        Weight for the soft norm penalty          [FIX-NORM].
        norm_target:        Target L2 norm for mean-pooled encodings.
        mixed_ctx_prob:     Probability of using cross-protein mixed context [TNP-ATTN-5].

    Returns:
        avg_loss      : float
        per_group_mse : dict[str, float]
    """
    model.train()
    total_loss = 0.0
    count = 0

    groups: dict[str, list] = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    all_protein_list = [
        g for g, sl in groups.items() if len(sl) >= 4 and g not in NON_PROTEIN_GROUPS
    ]
    protein_list = [g for g, sl in groups.items() if len(sl) >= 4]

    raw_w = (
        np.array([group_weights.get(g, 1.0) for g in protein_list], dtype=float)
        if group_weights is not None
        else np.ones(len(protein_list), dtype=float)
    )
    sampling_probs = raw_w / raw_w.sum()

    group_loss_accum: dict[str, float] = defaultdict(float)
    group_loss_count: dict[str, int] = defaultdict(int)

    for _ in range(iterations):
        if len(protein_list) < 2:
            continue

        # ---- Sample anchor protein and split into context / target ----
        idx_anchor = np.random.choice(len(protein_list), p=sampling_probs)
        prot_A = protein_list[idx_anchor]
        task_A = groups[prot_A]

        idx_A = np.random.permutation(len(task_A))
        n_ctx_A = np.random.randint(1, min(12, len(idx_A) - 1))

        # ---- [TNP-ATTN-5] Context construction ----
        # Two modes controlled by mixed_ctx_prob:
        #
        # STANDARD (60%): all context tokens from anchor protein.
        #   Null baseline = single zero latent token (original FIX-5).
        #
        # MIXED (40%): half tokens from anchor, half from a random other protein
        #   ("impostors").  Targets are always anchor-protein only.
        #   Null baseline = impostor-only context — measures the specific
        #   improvement from attending to relevant samples vs irrelevant ones.
        #   This forces attention to discriminate; mean-pooling over mixed context
        #   yields higher loss than upweighting anchor tokens.

        use_mixed = (
            np.random.random() < mixed_ctx_prob
            and len(all_protein_list) >= 3
            and len(task_A) >= 4
        )

        if use_mixed:
            # Split context budget: at least 1 anchor, at least 1 impostor
            n_anchor = max(1, n_ctx_A // 2)
            n_impostor = max(1, n_ctx_A - n_anchor)

            anchor_indices = idx_A[:n_anchor]
            anchor_items = [task_A[i] for i in anchor_indices]

            # Pick impostor protein (different from anchor)
            other_prots = [g for g in all_protein_list if g != prot_A]
            imp_prot = np.random.choice(other_prots)
            imp_task = groups[imp_prot]
            imp_idx = np.random.choice(
                len(imp_task), min(n_impostor, len(imp_task)), replace=False
            )
            impostor_items = [imp_task[i] for i in imp_idx]

            # Build mixed context list — shuffle so position doesn't signal identity
            # Context encoding uses s["static"] (full features with protein identity)
            mixed_items = anchor_items + impostor_items
            np.random.shuffle(mixed_items)

            ctx_items_list = []
            for s in mixed_items:
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_items_list.append(torch.cat([s["points"], stat], dim=1).to(device))
            ctx_A = model.encode_context_samples(ctx_items_list)

            # Impostor-only context for utility baseline
            imp_items_list = []
            for s in impostor_items:
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                imp_items_list.append(torch.cat([s["points"], stat], dim=1).to(device))
            with torch.no_grad():
                imp_ctx = model.encode_context_samples(imp_items_list)
        else:
            # Standard: all context from anchor protein [TNP-ATTN-4]
            ctx_A = _build_ctx_encoded(model, task_A, idx_A[:n_ctx_A], device)
            imp_ctx = None  # sentinel: use zero-token baseline below

        qx_A, qstat_A, qy_A = _build_tgt_tensors(task_A, idx_A[n_ctx_A:], device)
        if qx_A is None:
            continue

        # ---- Forward pass with optional static feature masking [FIX-5] ----
        if np.random.random() < 0.60:
            mask = torch.bernoulli(torch.full_like(qstat_A, 0.5))
            qstat_A_in = qstat_A * mask
        else:
            qstat_A_in = qstat_A

        pred_A, _ = _forward(model, ctx_A, qx_A, qstat_A_in, ctx_is_encoded=True)
        mse_loss = F.mse_loss(pred_A, qy_A)

        # ---- Context utility loss [FIX-5 / TNP-ATTN-5] ----
        # Baseline: impostor-only context (mixed mode) or single zero token (standard).
        # Both paths use ctx_is_encoded=True.
        if imp_ctx is not None:
            baseline_ctx = imp_ctx
        else:
            baseline_ctx = torch.zeros(1, 1, model.latent_dim, device=device)

        with torch.no_grad():
            pred_baseline, _ = _forward(
                model, baseline_ctx, qx_A, qstat_A, ctx_is_encoded=True
            )
        mse_baseline = F.mse_loss(pred_baseline, qy_A).detach()

        pred_ctx_unmasked, _ = _forward(
            model, ctx_A, qx_A, qstat_A, ctx_is_encoded=True
        )
        mse_ctx_unmasked = F.mse_loss(pred_ctx_unmasked, qy_A)
        utility_loss = torch.clamp(mse_ctx_unmasked - mse_baseline + 1e-3, min=0.0)

        # ---- Soft norm penalty [FIX-NORM] ----
        # ctx_A is already [1, N_samples, latent_dim]; pool across samples.
        r_current = ctx_A.mean(dim=1)  # [1, latent_dim]
        r_norm = torch.norm(r_current, p=2, dim=-1)
        norm_penalty = torch.mean(torch.clamp(r_norm - norm_target, min=0.0) ** 2)

        # ---- Triplet [FIX-3] + latent consistency [FIX-4] ----
        # These use raw ctx tensors -> encode_latent_mean, not the cross-attention path.
        triplet_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)

        if prot_A in all_protein_list and len(all_protein_list) >= 2:
            perm_full = np.random.permutation(len(task_A))
            half = max(1, len(perm_full) // 2)

            # Raw tensors for latent-space losses (not used in cross-attention)
            ctx_anchor_raw = _build_ctx_tensor(task_A, perm_full[:half], device)
            ctx_pos_raw = _build_ctx_tensor(task_A, perm_full[half:], device)

            # Intra-group cosine consistency [FIX-4]
            enc_anchor = _encode_latent(model, ctx_anchor_raw)
            enc_pos = _encode_latent(model, ctx_pos_raw)
            cos_within = F.cosine_similarity(enc_anchor, enc_pos, dim=-1)
            consistency_loss = (1.0 - cos_within).mean()

            r_anchor = enc_anchor
            r_pos = enc_pos

            # Hard negative mining [FIX-3]
            class_A = PROTEIN_CLASS_MAP.get(prot_A, "unknown")
            same_class_negs = [
                g
                for g in all_protein_list
                if g != prot_A and PROTEIN_CLASS_MAP.get(g, "") == class_A
            ]
            diff_class_negs = [g for g in all_protein_list if g != prot_A]

            if same_class_negs and np.random.random() < 0.70:
                prot_B = np.random.choice(same_class_negs)
            elif diff_class_negs:
                prot_B = np.random.choice(diff_class_negs)
            else:
                prot_B = prot_A

            task_B = groups[prot_B]
            idx_B = np.random.permutation(len(task_B))
            n_ctx_B = np.random.randint(1, min(8, len(idx_B)))
            r_neg = _encode_latent(
                model, _build_ctx_tensor(task_B, idx_B[:n_ctx_B], device)
            )

            d_pos = torch.sum((r_anchor - r_pos) ** 2, dim=-1).sqrt()
            d_neg = torch.sum((r_anchor - r_neg) ** 2, dim=-1).sqrt()
            triplet_loss = torch.clamp(d_pos - d_neg + triplet_margin, min=0.0).mean()

        # ---- Combined loss ----
        loss = (
            mse_loss
            + lambda_utility * utility_loss
            + lambda_triplet * triplet_loss
            + lambda_consistency * consistency_loss
            + lambda_norm * norm_penalty
        )

        if torch.isnan(loss):
            print("Warning: NaN loss encountered. Skipping batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1
        group_loss_accum[prot_A] += mse_loss.item()
        group_loss_count[prot_A] += 1

    per_group_mse = {
        g: group_loss_accum[g] / group_loss_count[g]
        for g in group_loss_accum
        if group_loss_count[g] > 0
    }
    return total_loss / max(1, count), per_group_mse


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    model: TransformerNP,
    samples: list,
    device: torch.device,
    n_repeats: int = 3,
) -> float:
    """
    Compute mean validation MSE using randomised context selection.  [FIX-8]

    Uses _build_ctx_encoded() so attention operates on sample-level tokens,
    consistent with the training forward pass.

    Args:
        model:     TransformerNP instance (switched to eval internally).
        samples:   List of sample dicts.
        device:    Computation device.
        n_repeats: Number of random 50/50 splits per group.

    Returns:
        Mean validation MSE across all qualifying groups.
    """
    model.eval()
    total_error = 0.0
    count = 0

    groups: dict[str, list] = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    with torch.no_grad():
        for prot, task_samples in groups.items():
            if len(task_samples) < 2:
                continue
            group_errors = []
            for _ in range(n_repeats):
                idx = np.random.permutation(len(task_samples))
                mid = max(1, len(idx) // 2)
                ctx_idx = idx[:mid]
                tgt_idx = idx[mid:]
                if len(tgt_idx) == 0:
                    continue

                # [TNP-ATTN-4] Sample-level context encoding
                ctx_tensor = _build_ctx_encoded(model, task_samples, ctx_idx, device)

                tgt_shear, tgt_y, tgt_stat = [], [], []
                for i in tgt_idx:
                    s = task_samples[i]
                    tgt_shear.append(s["points"][:, [0]])
                    tgt_y.append(s["points"][:, [1]])
                    tgt_stat.append(
                        s["static_qry"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                    )

                q_x = torch.cat(tgt_shear, dim=0).unsqueeze(0).to(device)
                q_stat = torch.cat(tgt_stat, dim=0).unsqueeze(0).to(device)
                true_y = torch.cat(tgt_y, dim=0).unsqueeze(0).to(device)

                pred, _ = _forward(model, ctx_tensor, q_x, q_stat, ctx_is_encoded=True)
                loss = F.mse_loss(pred, true_y)
                if not torch.isnan(loss):
                    group_errors.append(loss.item())

            if group_errors:
                total_error += float(np.mean(group_errors))
                count += 1

    return total_error / max(1, count)
