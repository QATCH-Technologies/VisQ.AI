"""Training, validation, and latent-representation diagnostics for the
:class:`visqai.models.cnp.CrossSampleCNP` architecture.

The training loop optimizes a decomposition of the prediction into a
feature-only prior and a context-dependent correction:

    prediction = prior_head(query) + correction_head(query, context_memory)

The prior is trained directly against the target, while the correction head is
trained against the detached residual left by the current prior. This keeps
the two prediction paths functionally distinct: the prior learns the
context-independent prediction, while the pooled context learns only the
information that the prior does not explain.

The module provides:

* :func:`train_epoch` for stochastic multi-objective model optimization.
* :func:`validate_zero_shot` for the literal zero-memory deployment path.
* :func:`validate` for randomized context-informed validation.
* :func:`validate_fewshot` for deployment-oriented few-shot evaluation.
* :func:`log_latent_variance` for monitoring separation between protein-group
  latent representations.
* :func:`log_flatness` for detecting collapse toward flat or population-mean
  predictions.

Training combines weighted reconstruction with auxiliary objectives for
context utility, latent consistency, contrastive separation, and latent-norm
control. Static-feature masking can be applied during training while
protecting designated physically load-bearing features.

Gradient clipping is applied independently to `prior_head` and the
remaining model parameters. The prior path has no dependency on pooled
context, so allowing large context-path gradients to consume the same
clipping budget can unnecessarily suppress updates to the zero-shot path.
Separate clipping prevents this gradient cross-talk.

Validation deliberately exposes multiple operating regimes. In particular,
:func:`validate_zero_shot` evaluates the exact zero-memory path used during
deployment rather than approximating it with an empty encoded context.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as f
from sklearn.preprocessing import StandardScaler

from visqai.models.cnp import CrossSampleCNP
from visqai.training.data import (
    PROTEIN_CLASS_MAP,
    NON_PROTEIN_GROUPS,
    _build_ctx_tensor,
    _build_tgt_tensors,
    compute_viscosity_weights,
)


def train_epoch(
    model: CrossSampleCNP,
    samples: list[dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    iterations: int = 100,
    group_weights: dict[str, float] | None = None,
    lambda_triplet: float = 0.30,
    lambda_consistency: float = 0.10,
    lambda_utility: float = 2.5,
    triplet_margin: float = 1.0,
    lambda_norm: float = 0.02,
    norm_target: float = 12.0,
    protected_indices: list[int] | None = None,
    mask_prob: float = 0.15,
    physics_scaler: StandardScaler | None = None,
    visc_threshold: float = 1.3,
    visc_max_weight: float = 4.0,
) -> tuple[float, dict[str, float]]:
    """Run one stochastic training epoch for a CrossSampleCNP model.

    Samples formulation-level context/query tasks, computes the weighted
    prior and context-dependent correction losses, and jointly optimizes the
    reconstruction, context-utility, contrastive, consistency, and latent-
    norm objectives.

    The reconstruction objective is explicitly split between the feature-only
    prior and the context-dependent correction. The correction target is
    detached from the prior so the correction head learns the residual left by
    the current prior rather than jointly re-fitting the target. Gradients for
    `prior_head` are clipped independently from the remainder of the model
    to prevent context-path auxiliary losses from suppressing prior updates.

    Context sampling is performed at the protein-group level. Groups with
    sufficient samples are sampled according to optional group weights, while
    same-class protein groups are preferentially selected as contrastive
    negatives. Optional static-feature masking can hide non-protected
    descriptors while preserving designated physically load-bearing features.

    Args:
        model: :class:`visqai.models.cnp.CrossSampleCNP` instance to train.
        samples: Collection of preprocessed formulation samples. Each sample
            must provide `"group"`, `"points"`, and `"static"` entries.
        optimizer: Torch optimizer used to update model parameters.
        device: Torch device on which tensors and losses are evaluated.
        iterations: Number of stochastic context/query training tasks to
            sample for the epoch.
        group_weights: Optional mapping from protein-group name to relative
            sampling weight. Groups not present in the mapping receive weight
            `1.0`.
        lambda_triplet: Weight applied to the latent triplet loss.
        lambda_consistency: Weight applied to the within-protein latent
            consistency loss.
        lambda_utility: Weight applied to the multi-shot context-utility loss.
        triplet_margin: Margin used by the latent triplet objective.
        lambda_norm: Weight applied to the latent-norm penalty.
        norm_target: Target upper bound for the pooled latent vector norm.
        protected_indices: Optional indices of static features that must not
            be masked during static-feature augmentation.
        mask_prob: Probability of applying partial static-feature masking to a
            sampled query task.
        physics_scaler: Optional fitted scaler used to recover log10-viscosity
            values for magnitude-dependent loss weighting. If `None`,
            unweighted MSE is used for the reconstruction terms.
        visc_threshold: Log10-viscosity threshold at which viscosity loss
            weighting begins to increase.
        visc_max_weight: Maximum asymptotic relative weight for high-viscosity
            points.

    Returns:
        A tuple `(mean_loss, per_group_mse)` where `mean_loss` is the
        average total optimization loss across successful iterations and
        `per_group_mse` maps each sampled protein group to its mean
        unweighted reconstruction MSE.

    Notes:
        Training iterations that cannot construct a valid query task or that
        produce NaN losses are skipped. The reported per-group metric uses the
        unweighted MSE of the combined prior-plus-correction prediction, even
        when viscosity weighting is enabled, so it remains comparable across
        groups.
    """
    if (
        physics_scaler is not None
        and physics_scaler.mean_ is not None
        and physics_scaler.scale_ is not None
    ):
        _visc_mean = float(physics_scaler.mean_[1])
        _visc_scale = float(physics_scaler.scale_[1])
    else:
        _visc_mean = 0.0
        _visc_scale = 1.0

    _protected = set(protected_indices) if protected_indices is not None else set()

    model.train()
    total_loss = 0
    count = 0

    groups = defaultdict(list)
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

    group_loss_accum = defaultdict(float)
    group_loss_count = defaultdict(int)

    for _ in range(iterations):
        if len(protein_list) < 2:
            continue
        idx_anchor = np.random.choice(len(protein_list), p=sampling_probs)
        prot_a = protein_list[idx_anchor]
        task_a = groups[prot_a]

        idx_a = np.random.permutation(len(task_a))
        n_ctx_a = np.random.randint(1, min(12, len(idx_a) - 1))
        ctx_a = _build_ctx_tensor(task_a, idx_a[:n_ctx_a], device)
        qx_a, qstat_a, qy_a = _build_tgt_tensors(task_a, idx_a[n_ctx_a:], device)
        if qx_a is None:
            continue

        # Partial masking that PRESERVES load-bearing physics columns: mask a
        # random subset of only the non-protected static features so the
        # decoder is nudged toward using r for excipient/buffer-identity
        # nuance without ever hiding concentration, pH, or protein identity.
        if np.random.random() < mask_prob:
            assert qstat_a is not None, "qstat_a cannot be None when applying mask"

            mask = torch.bernoulli(torch.full_like(qstat_a, 0.5))
            if _protected:
                prot_idx = torch.tensor(sorted(_protected), device=mask.device)
                mask[..., prot_idx] = 1.0
            qstat_a_in = qstat_a * mask
        else:
            qstat_a_in = qstat_a

        assert qstat_a_in is not None, "qstat_a_in must be a Tensor"
        prior_a, correction_a = model.forward_split(ctx_a, qx_a, qstat_a_in)
        pred_a = prior_a + correction_a

        # residual_target is detached: correction_A is only ever graded
        # against what the CURRENT prior snapshot misses, never against y
        # directly, so its gradient can't just re-derive the target through
        # prior_head a second time (see module docstring).
        assert qy_a is not None, "qy_a cannot be None when computing residual target"

        residual_target = (qy_a - prior_a).detach()
        if _visc_mean is not None:
            w_a = compute_viscosity_weights(
                qy_a, _visc_mean, _visc_scale, threshold=visc_threshold, max_weight=visc_max_weight
            )
            prior_loss = (w_a * (prior_a - qy_a) ** 2).mean()
            correction_loss = (w_a * (correction_a - residual_target) ** 2).mean()
        else:
            prior_loss = f.mse_loss(prior_a, qy_a)
            correction_loss = f.mse_loss(correction_a, residual_target)

        mse_loss = prior_loss + correction_loss
        mse_loss_unweighted = f.mse_loss(pred_a, qy_a).detach()

        # Graded multi-shot context utility: more context should reduce error
        # on HELD-OUT target points of the same protein. Encode r from a
        # 1-shot and a k-shot context, decode both on the same held-out
        # queries, and penalize when the larger context doesn't help.
        utility_loss = torch.tensor(0.0, device=device)
        if n_ctx_a >= 2:
            ctx_pool = idx_a[:n_ctx_a]
            split = max(1, len(ctx_pool) // 2)
            small_ctx = _build_ctx_tensor(task_a, ctx_pool[:1], device)
            large_ctx = _build_ctx_tensor(task_a, ctx_pool[: split + 1], device)

            r_small = model.encode_memory(small_ctx)
            r_large = model.encode_memory(large_ctx)

            assert qstat_a is not None, "qstat_a must be a Tensor for decoding"
            pred_small = model.decode_from_memory(r_small, qx_a, qstat_a)
            pred_large = model.decode_from_memory(r_large, qx_a, qstat_a)

            mse_small = f.mse_loss(pred_small, qy_a)
            mse_large = f.mse_loss(pred_large, qy_a)
            utility_loss = torch.clamp(mse_large - mse_small.detach() + 1e-3, min=0.0)

            with torch.no_grad():
                pred_null = model(torch.zeros_like(large_ctx), qx_a, qstat_a)
                mse_null = f.mse_loss(pred_null, qy_a)
            utility_loss = utility_loss + 0.5 * torch.clamp(mse_large - mse_null + 1e-3, min=0.0)

        r_current = model.encode_memory(ctx_a)
        r_norm_current = torch.norm(r_current, p=2, dim=-1)
        norm_penalty = torch.mean(torch.clamp(r_norm_current - norm_target, min=0.0) ** 2)

        triplet_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)

        if prot_a in all_protein_list and len(all_protein_list) >= 2:
            perm_full = np.random.permutation(len(task_a))
            half = max(1, len(perm_full) // 2)

            ctx_anchor = _build_ctx_tensor(task_a, perm_full[:half], device)
            ctx_pos = _build_ctx_tensor(task_a, perm_full[half:], device)

            enc_anchor_mean = model.encoder(ctx_anchor).mean(dim=1)
            enc_pos_mean = model.encoder(ctx_pos).mean(dim=1)
            cos_within = f.cosine_similarity(enc_anchor_mean, enc_pos_mean, dim=-1)
            consistency_loss = (1.0 - cos_within).mean()

            r_anchor = model.encode_memory(ctx_anchor)
            r_pos = model.encode_memory(ctx_pos)

            class_a = PROTEIN_CLASS_MAP.get(prot_a, "unknown")
            same_class_negs = [
                g
                for g in all_protein_list
                if g != prot_a and PROTEIN_CLASS_MAP.get(g, "") == class_a
            ]
            diff_class_negs = [g for g in all_protein_list if g != prot_a]

            if same_class_negs and np.random.random() < 0.70:
                prot_b = np.random.choice(same_class_negs)
            elif diff_class_negs:
                prot_b = np.random.choice(diff_class_negs)
            else:
                prot_b = prot_a

            task_b = groups[prot_b]
            idx_b = np.random.permutation(len(task_b))
            n_ctx_b = np.random.randint(1, min(8, len(idx_b)))
            r_neg = model.encode_memory(_build_ctx_tensor(task_b, idx_b[:n_ctx_b], device))

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

        group_loss_accum[prot_a] += mse_loss_unweighted.item()
        group_loss_count[prot_a] += 1

    per_group_mse = {
        g: group_loss_accum[g] / group_loss_count[g]
        for g in group_loss_accum
        if group_loss_count[g] > 0
    }
    return total_loss / max(1, count), per_group_mse


def validate_zero_shot(
    model: CrossSampleCNP, samples: list[dict], device: torch.device, latent_dim: int
) -> float:
    """Evaluate the model's literal zero-shot prediction path.

    Scores held-out samples using a zero pooled-memory vector and
    :meth:`CrossSampleCNP.decode_from_memory`, exactly matching the deployment
    path used when no context memory is available. The encoder and attention
    pooler are bypassed entirely.

    This is intentionally distinct from passing an empty or zero-valued
    context tensor through the encoder, because the attention pooler's
    normalization does not guarantee that such a context produces a zero
    latent vector.

    Args:
        model: Trained :class:`visqai.models.cnp.CrossSampleCNP` model.
        samples: Collection of held-out formulation samples grouped by
            `"group"` and containing `"points"` and `"static"` tensors.
        device: Torch device on which evaluation is performed.
        latent_dim: Dimensionality of the model's pooled memory vector.

    Returns:
        Mean MSE across evaluable protein groups using the literal zero-shot
        path. Returns `0.0` when no valid groups can be evaluated.

    Notes:
        This metric is used independently of context-informed validation so
        checkpoint selection can explicitly monitor the quality of
        `prior_head` predictions. It should therefore exercise the same
        `r=0` behavior that inference uses for zero-shot prediction.
    """
    model.eval()
    total_error = 0
    count = 0
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    with torch.no_grad():
        for _, task_samples in groups.items():
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
            loss = f.mse_loss(pred, true_y)
            if not torch.isnan(loss):
                total_error += loss.item()
                count += 1

    return total_error / max(1, count)


def validate(
    model: CrossSampleCNP, samples: list[dict], device: torch.device, n_repeats: int = 3
) -> float:
    """Evaluate context-informed generalization with randomized splits.

    For each protein group, repeatedly partitions formulation samples into a
    non-empty context set and held-out target set, predicts the targets from
    the context, and averages the resulting MSEs. Randomized splits avoid the
    ordering bias that would result from always using a fixed portion of each
    group's samples as context.

    Args:
        model: Trained :class:`visqai.models.cnp.CrossSampleCNP` model.
        samples: Collection of preprocessed formulation samples containing
            `"group"`, `"points"`, and `"static"` entries.
        device: Torch device on which evaluation is performed.
        n_repeats: Number of independent random context/target splits to
            evaluate for each protein group.

    Returns:
        Mean context-informed MSE across protein groups with at least two
        samples. Returns `0.0` when no groups provide a valid split.

    Notes:
        The context and target sets are formed at the formulation level,
        preserving the distinction between independent formulations and their
        multiple measurement points.
    """
    model.eval()
    total_error = 0
    count = 0
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    with torch.no_grad():
        for _, task_samples in groups.items():
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
                loss = f.mse_loss(pred, true_y)
                if not torch.isnan(loss):
                    group_errors.append(loss.item())

            if group_errors:
                total_error += float(np.mean(group_errors))
                count += 1

    return total_error / max(1, count)


def log_latent_variance(model: CrossSampleCNP, samples: list[dict], device: torch.device) -> float:
    """Measure separation between protein-group latent representations.

    Constructs a pooled memory representation for each eligible protein group
    and computes the mean pairwise Euclidean distance between group-level
    latent vectors. Non-protein groups are excluded because separating them
    from protein groups is comparatively trivial and can obscure whether the
    model has learned meaningful protein-to-protein distinctions.

    Args:
        model: Trained :class:`visqai.models.cnp.CrossSampleCNP` model.
        samples: Collection of preprocessed formulation samples grouped by
            `"group"`.
        device: Torch device on which latent representations are computed.

    Returns:
        Mean pairwise L2 distance between eligible protein-group latent
        centroids. Returns `0.0` when fewer than two eligible groups are
        available.

    Notes:
        Upward movement of this diagnostic during training generally indicates
        increasing separation between protein-group representations, whereas
        values near zero are consistent with latent representation collapse.
        Up to five randomly selected formulations are used per group to
        construct each diagnostic context.
    """
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


def log_flatness(
    model: CrossSampleCNP, samples: list[dict], device: torch.device, n_groups: int = 6
) -> tuple[float, float]:
    """Diagnose collapse toward a population-mean zero-context prediction.

    Selects a subset of protein groups and evaluates each using a null context.
    Two complementary statistics are computed: variation in predicted
    log10-viscosity across the shear axis for each formulation, and variation
    in the mean predicted viscosity level across groups.

    Args:
        model: Trained :class:`visqai.models.cnp.CrossSampleCNP` model.
        samples: Collection of preprocessed formulation samples grouped by
            `"group"` and containing `"points"` and `"static"` tensors.
        device: Torch device on which evaluation is performed.
        n_groups: Maximum number of protein groups to sample for the
            diagnostic.

    Returns:
        A tuple `(shear_shape_std, cross_sample_std)` where:

        * `shear_shape_std` is the mean within-formulation standard
          deviation of predicted log10 viscosity across the shear axis.
        * `cross_sample_std` is the standard deviation of mean predictions
          across sampled formulation groups.

        Both values are `0.0` when the corresponding statistic cannot be
        computed.

    Notes:
        Values near zero indicate that the model is producing nearly flat,
        population-mean predictions under null context. The diagnostic is
        intended to distinguish loss of shear-dependent shape from loss of
        formulation-specific prediction levels.
    """
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


def validate_fewshot(
    model: CrossSampleCNP,
    val_samples: list[dict],
    device: torch.device,
    shots: tuple[int, ...] = (1, 2, 4),
    n_repeats: int = 5,
) -> float:
    """Evaluate few-shot generalization across multiple context sizes.

    For each requested number of context formulations, repeatedly samples that
    many formulations from a held-out validation group, uses them as context,
    and evaluates the remaining formulations as targets. The resulting MSEs
    are averaged across valid shot counts and random draws.

    Args:
        model: Trained :class:`visqai.models.cnp.CrossSampleCNP` model.
        val_samples: Preprocessed formulations belonging to the held-out
            validation group. Each sample must contain `"points"` and
            `"static"` tensors.
        device: Torch device on which evaluation is performed.
        shots: Iterable of context-set sizes to evaluate.
        n_repeats: Number of random context/target draws for each shot count.

    Returns:
        Mean MSE across all valid few-shot evaluations. Returns `float("inf")`
        when fewer than two validation samples are available or no valid
        evaluation can be performed.

    Notes:
        This metric is intended to approximate deployment behavior by
        explicitly measuring how prediction quality changes as progressively
        more context formulations become available.
    """
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
                loss = f.mse_loss(pred, q_y)
                if not torch.isnan(loss):
                    errors.append(loss.item())
    return float(np.mean(errors)) if errors else float("inf")
