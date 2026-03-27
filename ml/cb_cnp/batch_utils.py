"""
batch_utils.py
==============
Tensor construction helpers used by the training loop and evaluation code.

All three helpers take lists of sample dicts (as produced by
``data_pipeline.load_and_preprocess``) and convert them into batched
PyTorch tensors on the requested device.

Functions
---------
_build_ctx_tensor(task_samples, indices, device)
    Build a batched context tensor from a subset of samples.
_build_tgt_tensors(task_samples, indices, device)
    Build query shear, static, and target viscosity tensors.
_build_concept_targets(task_samples, indices, device)
    Average concept proxy targets over a set of context samples.
"""

from __future__ import annotations

import torch


def _build_ctx_tensor(
    task_samples: list[dict],
    indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Concatenate (points, static) pairs from the selected samples into a
    single context tensor.

    Parameters
    ----------
    task_samples : list[dict]
        All samples belonging to one protein group.
    indices : list[int] or array-like
        Indices into task_samples to use as context.
    device : torch.device

    Returns
    -------
    Tensor [1, N_points_total, 2 + static_dim]
        Batch dimension of 1 (single episode).
    """
    ctx_items = []
    for i in indices:
        s = task_samples[i]
        stat = s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1)
        ctx_items.append(torch.cat([s["points"], stat], dim=1))
    return torch.cat(ctx_items, dim=0).unsqueeze(0).to(device)


def _build_tgt_tensors(
    task_samples: list[dict],
    indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Build query tensors for the target (prediction) samples.

    Parameters
    ----------
    task_samples : list[dict]
        All samples belonging to one protein group.
    indices : list[int] or array-like
        Indices into task_samples to use as targets.
    device : torch.device

    Returns
    -------
    q_shear : Tensor [1, N_pts, 1] or None
    q_static : Tensor [1, N_pts, static_dim] or None
    q_y : Tensor [1, N_pts, 1] or None

    Returns (None, None, None) if no target indices were provided.
    """
    shear_list, y_list, stat_list = [], [], []
    for i in indices:
        s = task_samples[i]
        n = s["points"].shape[0]
        shear_list.append(s["points"][:, [0]])
        y_list.append(s["points"][:, [1]])
        stat_list.append(s["static"].unsqueeze(0).repeat(n, 1))

    if not shear_list:
        return None, None, None

    q_x = torch.cat(shear_list, dim=0).unsqueeze(0).to(device)
    q_stat = torch.cat(stat_list, dim=0).unsqueeze(0).to(device)
    q_y = torch.cat(y_list, dim=0).unsqueeze(0).to(device)
    return q_x, q_stat, q_y


def _build_concept_targets(
    task_samples: list[dict],
    indices: list[int],
    device: torch.device,
) -> torch.Tensor | None:
    """
    Compute the mean concept proxy target over a set of context samples.

    Used to generate the supervision signal for the concept bottleneck loss.
    Samples lacking a ``"concept_targets"`` key are silently skipped
    (backward-compatibility guard for datasets produced without concept proxies).

    Parameters
    ----------
    task_samples : list[dict]
        All samples belonging to one protein group.
    indices : list[int] or array-like
        Indices of the context samples whose proxies should be averaged.
    device : torch.device

    Returns
    -------
    Tensor [1, N_CONCEPTS_SUPERVISED] or None
        Mean concept proxy values in [-1, 1] / [0, 1] depending on activation.
        Returns None if no samples carry concept_targets.
    """
    targets = [
        task_samples[i]["concept_targets"] for i in indices if "concept_targets" in task_samples[i]
    ]
    if not targets:
        return None
    return torch.stack(targets).mean(dim=0).unsqueeze(0).to(device)  # [1, N_SUP]
