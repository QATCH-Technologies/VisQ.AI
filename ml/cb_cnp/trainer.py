"""
trainer.py
==========
One-epoch training loop and validation for the CBM-CNP pipeline.

Changes from prior version
--------------------------
- Static masking reverted to moderate 50%/50% (aggressive 85%/80% crippled
  the decoder's ability to use static features).
- Concept consistency loss now uses ``encode_concepts()`` (not ``encode_memory()``
  which now returns latent r, not concept c).
- Concept decoder co-training: the lightweight concept decoder receives a
  small auxiliary loss so it can be used for causal intervention analysis.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from cb_cnp.batch_utils import _build_concept_targets, _build_ctx_tensor, _build_tgt_tensors
from cb_cnp.constants import N_CONCEPTS_SUPERVISED, NON_PROTEIN_GROUPS, PROTEIN_CLASS_MAP
from cb_cnp.models import ConceptBottleneckCNP, _encode_latent, _forward


# ============================================================
# Training epoch
# ============================================================


def train_epoch(
    model: torch.nn.Module,
    samples: list[dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    iterations: int = 100,
    group_weights: dict[str, float] | None = None,
    # --- Contrastive / regularisation lambdas ---
    lambda_triplet: float = 0.10,
    lambda_consistency: float = 0.10,
    lambda_utility: float = 2.50,
    triplet_margin: float = 3.00,
    lambda_norm: float = 0.05,
    norm_target: float = 5.00,
    # --- CBM-specific lambdas ---
    lambda_concept_sup: float = 0.10,
    lambda_concept_consist: float = 0.05,
    lambda_decov: float = 0.03,
    lambda_sparsity: float = 0.01,
    lambda_concept_pred: float = 0.30,
    meta_holdout_prob: float = 0.20,
    meta_n_ctx_samples: int = 6,
) -> tuple[float, dict[str, float]]:
    """
    Train one epoch over randomly sampled protein groups.

    Parameters
    ----------
    lambda_concept_pred : float
        Weight for the concept decoder auxiliary prediction loss. Trains
        the secondary decoder that is used for causal interventions.
    """
    model.train()
    total_loss = 0.0
    count = 0

    is_cbm = isinstance(model, ConceptBottleneckCNP)

    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    all_protein_list = [
        g for g, sl in groups.items() if len(sl) >= 4 and g not in NON_PROTEIN_GROUPS
    ]
    protein_list = [g for g, sl in groups.items() if len(sl) >= 4]

    if group_weights is not None:
        raw_w = np.array([group_weights.get(g, 1.0) for g in protein_list], dtype=float)
    else:
        raw_w = np.ones(len(protein_list), dtype=float)
    sampling_probs = raw_w / raw_w.sum()

    group_loss_accum: dict[str, float] = defaultdict(float)
    group_loss_count: dict[str, int] = defaultdict(int)

    batch_concepts: list[torch.Tensor] = []

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

        # ---- Static masking (moderate: 50% prob, 50% features) ----
        if np.random.random() < 0.50:
            mask = torch.bernoulli(torch.full_like(qstat_A, 0.5))
            qstat_A_in = qstat_A * mask
        else:
            qstat_A_in = qstat_A

        pred_A, concepts_A = _forward(model, ctx_A, qx_A, qstat_A_in)
        mse_loss = F.mse_loss(pred_A, qy_A)

        # ---- Concept decoder auxiliary loss ----
        # Co-train the concept decoder so it can be used for interventions.
        concept_pred_loss = torch.tensor(0.0, device=device)
        if is_cbm and concepts_A is not None:
            # Detach concepts so this loss only trains the concept decoder,
            # not the encoder/bottleneck (those are trained by main path).
            c_detached = concepts_A.detach()
            pred_from_c = model.decode_from_concepts(c_detached, qx_A, qstat_A)
            concept_pred_loss = F.mse_loss(pred_from_c, qy_A)

        # ---- Context utility loss (FIX-5) ----
        with torch.no_grad():
            pred_null, _ = _forward(model, torch.zeros_like(ctx_A), qx_A, qstat_A)
        mse_null = F.mse_loss(pred_null, qy_A).detach()

        pred_ctx_unmasked, _ = _forward(model, ctx_A, qx_A, qstat_A)
        mse_ctx_unmasked = F.mse_loss(pred_ctx_unmasked, qy_A)
        utility_loss = torch.clamp(mse_ctx_unmasked - mse_null + 1e-3, min=0.0)

        # ---- Soft latent norm penalty (FIX-NORM) ----
        r_current = _encode_latent(model, ctx_A)
        r_norm = torch.norm(r_current, p=2, dim=-1)
        norm_penalty = torch.mean(torch.clamp(r_norm - norm_target, min=0.0) ** 2)

        # ---- Concept supervision loss (CBM-3) ----
        concept_sup_loss = torch.tensor(0.0, device=device)
        if is_cbm and concepts_A is not None:
            ctx_concept_targets = _build_concept_targets(task_A, idx_A[:n_ctx_A], device)
            if ctx_concept_targets is not None:
                n_sup = min(N_CONCEPTS_SUPERVISED, model.n_concepts)
                concept_sup_loss = F.mse_loss(
                    concepts_A[:, :n_sup],
                    ctx_concept_targets[:, :n_sup],
                )

        # ---- DeCov decorrelation loss (v3) ----
        decov_loss = torch.tensor(0.0, device=device)
        if is_cbm and concepts_A is not None:
            batch_concepts.append(concepts_A.detach())
            decov_loss = model.decov_loss(concepts_A)

        # ---- L1 sparsity on concept gates (v4) ----
        sparsity_loss = torch.tensor(0.0, device=device)
        if is_cbm:
            gates = torch.sigmoid(model.concept_gate_logits)
            if model.n_concepts > N_CONCEPTS_SUPERVISED:
                free_gates = gates[N_CONCEPTS_SUPERVISED:]
                sup_gates = gates[:N_CONCEPTS_SUPERVISED]
                sparsity_loss = 0.10 * free_gates.abs().mean() + 0.05 * sup_gates.abs().mean()
            else:
                sparsity_loss = gates.abs().mean()

        # ---- Triplet (FIX-3) + consistency (FIX-4) + concept consistency (CBM-4) ----
        triplet_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)
        concept_consist_loss = torch.tensor(0.0, device=device)

        if prot_A in all_protein_list and len(all_protein_list) >= 2:
            perm_full = np.random.permutation(len(task_A))
            half = max(1, len(perm_full) // 2)

            ctx_anchor = _build_ctx_tensor(task_A, perm_full[:half], device)
            ctx_pos = _build_ctx_tensor(task_A, perm_full[half:], device)

            # Pre-pooled latent consistency (FIX-4)
            enc_anchor_mean = model.encoder(ctx_anchor).mean(dim=1)
            enc_pos_mean = model.encoder(ctx_pos).mean(dim=1)
            cos_within = F.cosine_similarity(enc_anchor_mean, enc_pos_mean, dim=-1)
            consistency_loss = (1.0 - cos_within).mean()

            r_anchor = _encode_latent(model, ctx_anchor)
            r_pos = _encode_latent(model, ctx_pos)

            # Concept consistency (CBM-4): same protein -> similar concept activations.
            # Uses encode_concepts (NOT encode_memory, which now returns r).
            if is_cbm:
                c_anchor = model.encode_concepts(ctx_anchor)
                c_pos = model.encode_concepts(ctx_pos)
                concept_consist_loss = F.mse_loss(c_anchor, c_pos)

            # Protein-class-aware hard-negative triplet (FIX-3)
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
            r_neg = _encode_latent(model, _build_ctx_tensor(task_B, idx_B[:n_ctx_B], device))

            d_pos = torch.sum((r_anchor - r_pos) ** 2, dim=-1).sqrt()
            d_neg = torch.sum((r_anchor - r_neg) ** 2, dim=-1).sqrt()
            triplet_loss = torch.clamp(d_pos - d_neg + triplet_margin, min=0.0).mean()

        # ---- Protein-novelty meta-learning (Option 4b) ----
        meta_loss = torch.tensor(0.0, device=device)
        if (
            meta_holdout_prob > 0.0
            and np.random.random() < meta_holdout_prob
            and len(all_protein_list) >= 3
        ):
            if group_weights is not None:
                raw_w_meta = np.array(
                    [group_weights.get(g, 1.0) for g in all_protein_list], dtype=float
                )
                meta_probs = raw_w_meta / raw_w_meta.sum()
            else:
                meta_probs = None

            meta_anchor_idx = np.random.choice(len(all_protein_list), p=meta_probs)
            prot_novel = all_protein_list[meta_anchor_idx]
            task_novel = groups[prot_novel]

            donor_proteins = [g for g in all_protein_list if g != prot_novel]
            cross_ctx_samples: list[dict] = []
            for dp in donor_proteins:
                pool = groups[dp]
                n_take = max(1, meta_n_ctx_samples // max(1, len(donor_proteins)))
                take_idx = np.random.choice(len(pool), size=min(n_take, len(pool)), replace=False)
                cross_ctx_samples.extend([pool[i] for i in take_idx])

            if len(cross_ctx_samples) >= 2 and len(task_novel) >= 2:
                cross_idx = np.arange(len(cross_ctx_samples))
                ctx_cross = _build_ctx_tensor(cross_ctx_samples, cross_idx, device)

                novel_idx = np.random.permutation(len(task_novel))
                qx_novel, qstat_novel, qy_novel = _build_tgt_tensors(task_novel, novel_idx, device)
                if qx_novel is not None:
                    pred_novel, _ = _forward(model, ctx_cross, qx_novel, qstat_novel)
                    meta_loss = F.mse_loss(pred_novel, qy_novel)

        # ---- Combined loss ----
        loss = (
            mse_loss
            + lambda_utility * utility_loss
            + lambda_triplet * triplet_loss
            + lambda_consistency * consistency_loss
            + lambda_norm * norm_penalty
            + lambda_concept_sup * concept_sup_loss
            + lambda_concept_consist * concept_consist_loss
            + lambda_decov * decov_loss
            + lambda_sparsity * sparsity_loss
            + lambda_concept_pred * concept_pred_loss
            + meta_loss
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


# ============================================================
# Validation
# ============================================================


def validate(
    model: torch.nn.Module,
    samples: list[dict],
    device: torch.device,
    n_repeats: int = 3,
) -> float:
    """
    Estimate generalisation loss via randomized held-out context splits.
    """
    model.eval()
    total_error = 0.0
    count = 0

    groups: dict[str, list[dict]] = defaultdict(list)
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
                ctx_idx, tgt_idx = idx[:mid], idx[mid:]
                if len(tgt_idx) == 0:
                    continue

                ctx_tensor = _build_ctx_tensor(task_samples, ctx_idx, device)

                tgt_shear, tgt_y, tgt_stat = [], [], []
                for i in tgt_idx:
                    s = task_samples[i]
                    tgt_shear.append(s["points"][:, [0]])
                    tgt_y.append(s["points"][:, [1]])
                    tgt_stat.append(s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1))

                q_x = torch.cat(tgt_shear, dim=0).unsqueeze(0).to(device)
                q_stat = torch.cat(tgt_stat, dim=0).unsqueeze(0).to(device)
                true_y = torch.cat(tgt_y, dim=0).unsqueeze(0).to(device)

                pred, _ = _forward(model, ctx_tensor, q_x, q_stat)
                loss = F.mse_loss(pred, true_y)
                if not torch.isnan(loss):
                    group_errors.append(loss.item())

            if group_errors:
                total_error += np.mean(group_errors)
                count += 1

    return total_error / max(1, count)
