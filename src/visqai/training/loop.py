"""
loop.py
=======
Training/validation loops for CrossSampleCNP: train_epoch (the combined
weighted-MSE + contrastive + consistency + context-utility + norm-penalty
loss, with hard-group EMA oversampling), validate, validate_fewshot, and the
latent-collapse diagnostics (log_latent_variance, log_flatness).

Originally moved verbatim from ml/cnp_mk2/train_o_net_v4_rung1.py; the data-
fit term in train_epoch has since been split to match the prior/correction
decoder in visqai.models.cnp:

    prior_loss      = MSE(prior_head(query), y)
    correction_loss = MSE(g(query, r), (y - prior_head(query)).detach())
    mse_loss        = prior_loss + correction_loss

The `.detach()` on the residual target is load-bearing: it stops the
correction head's gradient from flowing back through prior_head, so g is only
ever rewarded for explaining what the CURRENT prior snapshot gets wrong, not
for jointly re-deriving y with prior_head. That is what forces the pooled
context to carry information the prior lacks instead of just re-fitting the
target a second time -- the failure mode that let a bad/OOD context make a
good zero-shot prediction worse (belatacept 0.077->0.125 log MAE at 1-shot).

SEPARATE GRADIENT CLIPPING FOR prior_head (P0 fix)
----------------------------------------------------
mse_loss = prior_loss + correction_loss is roughly double the magnitude of
the old single-decoder MSE term, and it shares one clip_grad_norm_(model.
parameters(), max_norm=1.0) call with the utility/triplet/consistency/norm
losses. Clipping rescales the WHOLE gradient vector by one scalar once its
norm exceeds max_norm -- so on any iteration where the encoder/pooler/
correction_head's combined gradient (from correction_loss + the auxiliary
losses, all of which route through the shared context-encoding path) is
large, prior_head's own gradient gets crushed by the same factor even though
prior_loss never spiked. Since prior_head no longer shares any parameters
with that path (forward() never routes prior_head's output through r), there
is no reason for it to share a clipping budget with it either. Clipping
prior_head's parameters separately from everything else removes this
cross-talk: a hard-to-fit high-N group's correction/utility gradient can no
longer throttle the very iteration that group needed to move prior_head.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from visqai.training.data import (
    PROTEIN_CLASS_MAP,
    NON_PROTEIN_GROUPS,
    _build_ctx_tensor,
    _build_tgt_tensors,
    compute_viscosity_weights,
)


def train_epoch(
    model,
    samples,
    optimizer,
    device,
    iterations=100,
    group_weights=None,
    lambda_triplet=0.30,
    lambda_consistency=0.10,
    lambda_utility=2.5,
    triplet_margin=1.0,
    lambda_norm=0.02,
    norm_target=12.0,
    # Indices of physically load-bearing static features that must survive
    # masking. Passed through from load_and_preprocess. None => protect
    # nothing (legacy behaviour).
    protected_indices=None,
    mask_prob=0.15,
    physics_scaler=None,
    visc_threshold=1.3,
    visc_max_weight=4.0,
):
    if physics_scaler is not None:
        _visc_mean = float(physics_scaler.mean_[1])
        _visc_scale = float(physics_scaler.scale_[1])
    else:
        _visc_mean = None
        _visc_scale = None

    _protected = set(protected_indices) if protected_indices is not None else set()

    model.train()
    total_loss = 0
    count = 0

    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    all_protein_list = [g for g, sl in groups.items() if len(sl) >= 4 and g not in NON_PROTEIN_GROUPS]
    protein_list = [g for g, sl in groups.items() if len(sl) >= 4]

    if group_weights is not None:
        raw_w = np.array([group_weights.get(g, 1.0) for g in protein_list], dtype=float)
    else:
        raw_w = np.ones(len(protein_list), dtype=float)
    sampling_probs = raw_w / raw_w.sum()

    group_loss_accum = defaultdict(float)
    group_loss_count = defaultdict(int)

    for _ in range(iterations):
        if len(protein_list) < 2:
            continue
        idx_anchor = np.random.choice(len(protein_list), p=sampling_probs)
        prot_A = protein_list[idx_anchor]
        task_A = groups[prot_A]

        idx_A = np.random.permutation(len(task_A))
        n_ctx_A = np.random.randint(1, min(12, len(idx_A) - 1))
        ctx_A = _build_ctx_tensor(task_A, idx_A[:n_ctx_A], device)
        qx_A, qstat_A, qy_A = _build_tgt_tensors(task_A, idx_A[n_ctx_A:], device)
        if qx_A is None:
            continue

        # Partial masking that PRESERVES load-bearing physics columns: mask a
        # random subset of only the non-protected static features so the
        # decoder is nudged toward using r for excipient/buffer-identity
        # nuance without ever hiding concentration, pH, or protein identity.
        if np.random.random() < mask_prob:
            mask = torch.bernoulli(torch.full_like(qstat_A, 0.5))
            if _protected:
                prot_idx = torch.tensor(sorted(_protected), device=mask.device)
                mask[..., prot_idx] = 1.0
            qstat_A_in = qstat_A * mask
        else:
            qstat_A_in = qstat_A

        prior_A, correction_A = model.forward_split(ctx_A, qx_A, qstat_A_in)
        pred_A = prior_A + correction_A

        # residual_target is detached: correction_A is only ever graded
        # against what the CURRENT prior snapshot misses, never against y
        # directly, so its gradient can't just re-derive the target through
        # prior_head a second time (see module docstring).
        residual_target = (qy_A - prior_A).detach()
        if _visc_mean is not None:
            w_A = compute_viscosity_weights(
                qy_A, _visc_mean, _visc_scale, threshold=visc_threshold, max_weight=visc_max_weight
            )
            prior_loss = (w_A * (prior_A - qy_A) ** 2).mean()
            correction_loss = (w_A * (correction_A - residual_target) ** 2).mean()
        else:
            prior_loss = F.mse_loss(prior_A, qy_A)
            correction_loss = F.mse_loss(correction_A, residual_target)

        mse_loss = prior_loss + correction_loss
        mse_loss_unweighted = F.mse_loss(pred_A, qy_A).detach()

        # Graded multi-shot context utility: more context should reduce error
        # on HELD-OUT target points of the same protein. Encode r from a
        # 1-shot and a k-shot context, decode both on the same held-out
        # queries, and penalize when the larger context doesn't help.
        utility_loss = torch.tensor(0.0, device=device)
        if n_ctx_A >= 2:
            ctx_pool = idx_A[:n_ctx_A]
            split = max(1, len(ctx_pool) // 2)
            small_ctx = _build_ctx_tensor(task_A, ctx_pool[:1], device)
            large_ctx = _build_ctx_tensor(task_A, ctx_pool[: split + 1], device)

            r_small = model.encode_memory(small_ctx)
            r_large = model.encode_memory(large_ctx)
            pred_small = model.decode_from_memory(r_small, qx_A, qstat_A)
            pred_large = model.decode_from_memory(r_large, qx_A, qstat_A)

            mse_small = F.mse_loss(pred_small, qy_A)
            mse_large = F.mse_loss(pred_large, qy_A)
            utility_loss = torch.clamp(mse_large - mse_small.detach() + 1e-3, min=0.0)

            with torch.no_grad():
                pred_null = model(torch.zeros_like(large_ctx), qx_A, qstat_A)
                mse_null = F.mse_loss(pred_null, qy_A)
            utility_loss = utility_loss + 0.5 * torch.clamp(mse_large - mse_null + 1e-3, min=0.0)

        r_current = model.encode_memory(ctx_A)
        r_norm_current = torch.norm(r_current, p=2, dim=-1)
        norm_penalty = torch.mean(torch.clamp(r_norm_current - norm_target, min=0.0) ** 2)

        triplet_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)

        if prot_A in all_protein_list and len(all_protein_list) >= 2:
            perm_full = np.random.permutation(len(task_A))
            half = max(1, len(perm_full) // 2)

            ctx_anchor = _build_ctx_tensor(task_A, perm_full[:half], device)
            ctx_pos = _build_ctx_tensor(task_A, perm_full[half:], device)

            enc_anchor_mean = model.encoder(ctx_anchor).mean(dim=1)
            enc_pos_mean = model.encoder(ctx_pos).mean(dim=1)
            cos_within = F.cosine_similarity(enc_anchor_mean, enc_pos_mean, dim=-1)
            consistency_loss = (1.0 - cos_within).mean()

            r_anchor = model.encode_memory(ctx_anchor)
            r_pos = model.encode_memory(ctx_pos)

            class_A = PROTEIN_CLASS_MAP.get(prot_A, "unknown")
            same_class_negs = [
                g for g in all_protein_list if g != prot_A and PROTEIN_CLASS_MAP.get(g, "") == class_A
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
            r_neg = model.encode_memory(_build_ctx_tensor(task_B, idx_B[:n_ctx_B], device))

            d_pos = torch.sum((r_anchor - r_pos) ** 2, dim=-1).sqrt()
            d_neg = torch.sum((r_anchor - r_neg) ** 2, dim=-1).sqrt()

            triplet_loss = torch.clamp(d_pos - d_neg + triplet_margin, min=0.0).mean()

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
        # Clipped separately from the rest of the model -- see module
        # docstring ("SEPARATE GRADIENT CLIPPING FOR prior_head"): prior_head
        # shares no parameters with the r-dependent path, so a large
        # correction/utility/triplet gradient must not be allowed to shrink
        # prior_head's own update via a shared clipping budget.
        torch.nn.utils.clip_grad_norm_(model.prior_head.parameters(), max_norm=1.0)
        rest_params = [p for n, p in model.named_parameters() if not n.startswith("prior_head.")]
        torch.nn.utils.clip_grad_norm_(rest_params, max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

        group_loss_accum[prot_A] += mse_loss_unweighted.item()
        group_loss_count[prot_A] += 1

    per_group_mse = {
        g: group_loss_accum[g] / group_loss_count[g] for g in group_loss_accum if group_loss_count[g] > 0
    }
    return total_loss / max(1, count), per_group_mse


def validate_zero_shot(model, samples, device, latent_dim):
    """Pure zero-shot validation: predicts every held-out point using the
    EXACT r=0 path visqai.inference.predictor.py uses at deployment --
    model.decode_from_memory(torch.zeros((1, latent_dim)), ...), bypassing
    the encoder/pooler entirely. This is NOT the same as feeding an empty/
    zero context tensor through the encoder (AttentionPool's LayerNorm gives
    no guarantee that maps to r=0), and it is NOT what validate() measures --
    validate() always builds a non-empty context split.

    Why this exists: run_final_model's early-stopping/checkpoint selection
    (visqai.training.run) previously used validate()'s context-informed loss
    alone. Since the prior/correction decoder split (visqai.models.cnp) made
    prior_head the ONLY path to zero-shot predictions, and prior_head gets no
    gradient from anything validate() measures, checkpoint selection was
    blind to the metric that governs zero-shot quality -- it could freely
    pick a snapshot that looks great with context and mediocre without it.
    Mixing this into the early-stopping signal (see train_final_model) closes
    that blind spot."""
    model.eval()
    total_error = 0
    count = 0
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    with torch.no_grad():
        for prot, task_samples in groups.items():
            shear_list, y_list, stat_list = [], [], []
            for s in task_samples:
                shear_list.append(s["points"][:, [0]])
                y_list.append(s["points"][:, [1]])
                stat_list.append(s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1))
            if not shear_list:
                continue
            q_x = torch.cat(shear_list, dim=0).unsqueeze(0).to(device)
            q_stat = torch.cat(stat_list, dim=0).unsqueeze(0).to(device)
            true_y = torch.cat(y_list, dim=0).unsqueeze(0).to(device)

            r_zero = torch.zeros((1, latent_dim), device=device)
            pred = model.decode_from_memory(r_zero, q_x, q_stat)
            loss = F.mse_loss(pred, true_y)
            if not torch.isnan(loss):
                total_error += loss.item()
                count += 1

    return total_error / max(1, count)


def validate(model, samples, device, n_repeats=3):
    """Randomized-context validation: n_repeats random splits per group,
    averaged, removing the ordering bias of a fixed first-half-as-context
    split."""
    model.eval()
    total_error = 0
    count = 0
    groups = defaultdict(list)
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

                ctx_list = []
                for i in ctx_idx:
                    s = task_samples[i]
                    stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                    ctx_list.append(torch.cat([s["points"], stat], dim=1))
                ctx_tensor = torch.cat(ctx_list, dim=0).unsqueeze(0).to(device)

                tgt_shear, tgt_y, tgt_stat = [], [], []
                for i in tgt_idx:
                    s = task_samples[i]
                    tgt_shear.append(s["points"][:, [0]])
                    tgt_y.append(s["points"][:, [1]])
                    tgt_stat.append(s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1))

                q_x = torch.cat(tgt_shear, dim=0).unsqueeze(0).to(device)
                q_stat = torch.cat(tgt_stat, dim=0).unsqueeze(0).to(device)
                true_y = torch.cat(tgt_y, dim=0).unsqueeze(0).to(device)
                pred = model(ctx_tensor, q_x, q_stat)
                loss = F.mse_loss(pred, true_y)
                if not torch.isnan(loss):
                    group_errors.append(loss.item())

            if group_errors:
                total_error += np.mean(group_errors)
                count += 1

    return total_error / max(1, count)


def log_latent_variance(model, samples, device):
    """Mean pairwise L2 distance between PROTEIN-ONLY group latent centroids
    (buffer-only groups excluded -- protein-vs-buffer separation is trivially
    easy and masks whether protein-protein discrimination is actually
    happening). Healthy training should grow this over time; near-zero means
    context collapse."""
    model.eval()
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    group_r = {}
    with torch.no_grad():
        for prot, task_samples in groups.items():
            if len(task_samples) < 2:
                continue
            if prot in NON_PROTEIN_GROUPS:
                continue
            idx = np.random.permutation(len(task_samples))[: min(5, len(task_samples))]
            ctx_items = []
            for i in idx:
                s = task_samples[i]
                stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                ctx_items.append(torch.cat([s["points"], stat], dim=1))
            ctx_t = torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)
            r = model.encode_memory(ctx_t).squeeze(0).cpu().numpy()
            group_r[prot] = r

    if len(group_r) < 2:
        return 0.0

    vecs = np.stack(list(group_r.values()))
    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(np.linalg.norm(vecs[i] - vecs[j]))
    return float(np.mean(dists))


def log_flatness(model, samples, device, n_groups=6):
    """Diagnoses collapse-to-population-mean directly: for a sample of
    protein groups, predict the full shear-axis query with a NULL (zero)
    context (worst case) and measure (a) per-sample std of predicted
    log10-viscosity across the shear axis (any shear-thinning shape at all?)
    and (b) spread of per-sample mean predictions across groups (do
    different formulations get different levels from physics alone?).
    Returns (shear_shape_std, cross_sample_std); values near zero indicate
    collapse."""
    model.eval()
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)
    prot_groups = [g for g in groups if g not in NON_PROTEIN_GROUPS]
    if not prot_groups:
        return 0.0, 0.0
    np.random.shuffle(prot_groups)
    prot_groups = prot_groups[:n_groups]

    shear_shape_stds = []
    per_sample_means = []
    with torch.no_grad():
        for g in prot_groups:
            s = groups[g][np.random.randint(len(groups[g]))]
            n_pts = s["points"].shape[0]
            qx = s["points"][:, [0]].unsqueeze(0).to(device)
            qstat = s["static"].unsqueeze(0).repeat(n_pts, 1).unsqueeze(0).to(device)
            null_ctx = torch.zeros(1, n_pts, 2 + s["static"].shape[0], device=device)
            pred = model(null_ctx, qx, qstat).squeeze().cpu().numpy()
            if pred.ndim == 0:
                continue
            shear_shape_stds.append(float(np.std(pred)))
            per_sample_means.append(float(np.mean(pred)))

    shear_shape_std = float(np.mean(shear_shape_stds)) if shear_shape_stds else 0.0
    cross_sample_std = float(np.std(per_sample_means)) if len(per_sample_means) > 1 else 0.0
    return shear_shape_std, cross_sample_std


def validate_fewshot(model, val_samples, device, shots=(1, 2, 4), n_repeats=5):
    """Few-shot held-out validation -- the metric that matches deployment.
    For a held-out protein, randomly pick k context samples and predict the
    remaining samples, for each k in `shots`, averaged over n_repeats random
    draws."""
    model.eval()
    if len(val_samples) < 2:
        return float("inf")
    errors = []
    with torch.no_grad():
        for k in shots:
            if k >= len(val_samples):
                continue
            for _ in range(n_repeats):
                idx = np.random.permutation(len(val_samples))
                ctx_idx, tgt_idx = idx[:k], idx[k:]
                if len(tgt_idx) == 0:
                    continue
                ctx_list = []
                for i in ctx_idx:
                    s = val_samples[i]
                    stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
                    ctx_list.append(torch.cat([s["points"], stat], dim=1))
                ctx_t = torch.cat(ctx_list, dim=0).unsqueeze(0).to(device)

                tx, ty, ts = [], [], []
                for i in tgt_idx:
                    s = val_samples[i]
                    tx.append(s["points"][:, [0]])
                    ty.append(s["points"][:, [1]])
                    ts.append(s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1))
                q_x = torch.cat(tx, dim=0).unsqueeze(0).to(device)
                q_y = torch.cat(ty, dim=0).unsqueeze(0).to(device)
                q_s = torch.cat(ts, dim=0).unsqueeze(0).to(device)
                pred = model(ctx_t, q_x, q_s)
                loss = F.mse_loss(pred, q_y)
                if not torch.isnan(loss):
                    errors.append(loss.item())
    return float(np.mean(errors)) if errors else float("inf")
