"""
Neural network layers module for the VisQ.AI.
Contains building blocks like ResidualBlocks, EmbeddingDropout,
and the specialized LearnablePhysicsPrior.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-01-15

Version:
    1.0
"""

from typing import cast

import torch
import torch.nn as nn

try:
    from .config import TARGETS
except (ImportError, ModuleNotFoundError):
    try:
        from config import TARGETS
    except (ImportError, ModuleNotFoundError):
        from visq_ml.config import TARGETS


class LearnableSoftThresholdPrior(nn.Module):
    def __init__(self, n_classes, n_regimes, n_excipients, initial_thresholds=None):
        super().__init__()
        # 1. Store Static Scores
        self.register_buffer(
            "static_scores", torch.zeros(n_classes, n_regimes, n_excipients)
        )

        # 2. Learnable Modifiers
        self.delta = nn.Parameter(torch.zeros(n_classes, n_regimes, n_excipients))

        # 3. Learnable Thresholds
        if initial_thresholds is None:
            t_data = torch.ones(n_excipients)
        elif isinstance(initial_thresholds, torch.Tensor):
            t_data = initial_thresholds.clone().detach()
        else:
            t_data = torch.tensor(initial_thresholds)

        self.thresholds = nn.Parameter(t_data.to(dtype=torch.float32))

        # 4. Learnable Weights
        self.w_below = nn.Parameter(
            torch.ones(n_classes, n_regimes, n_excipients) * 0.1
        )
        self.w_above = nn.Parameter(
            torch.ones(n_classes, n_regimes, n_excipients) * 0.5
        )
        self.sharpness = nn.Parameter(torch.tensor(10.0))

    def expand_indices(self, dim: int, new_entries: int = 1, source_idx: int = -1):
        """
        Dynamically resizes internal tensors to accommodate new categories.
        Args:
            dim: Dimension to expand (0=Class, 1=Regime, 2=Excipient)
            new_entries: How many new categories to add
            source_idx: If >= 0, copy weights from this index. If -1, use MEAN initialization.
        """
        device = self.delta.device

        def _expand_tensor(tensor, d, n, init_val=0.0, src_idx=-1):
            shape = list(tensor.shape)
            shape[d] = n

            if src_idx >= 0:
                # Copy from specific source index
                indices = [slice(None)] * len(shape)
                indices[d] = src_idx
                source_slice = tensor[tuple(indices)].unsqueeze(d)
                new_data = source_slice.repeat_interleave(n, dim=d)

                # Add noise to Parameters (not buffers) to break symmetry
                if isinstance(tensor, nn.Parameter):
                    noise = torch.randn_like(new_data) * 0.01
                    new_data = new_data + noise
            else:
                # Fallback: Use MEAN of existing entries instead of 0.0
                if tensor.size(d) > 0:
                    mean_slice = tensor.mean(dim=d, keepdim=True)
                    new_data = mean_slice.repeat_interleave(n, dim=d)
                    if isinstance(tensor, nn.Parameter):
                        noise = torch.randn_like(new_data) * 0.01
                        new_data = new_data + noise
                else:
                    new_data = (
                        torch.ones(shape, device=device, dtype=tensor.dtype) * init_val
                    )

            res = torch.cat([tensor, new_data], dim=d)
            if isinstance(tensor, nn.Parameter):
                return nn.Parameter(res)
            return res

        # Apply expansion
        self.static_scores = _expand_tensor(
            self.static_scores, dim, new_entries, 0.0, source_idx
        )
        self.delta = _expand_tensor(self.delta, dim, new_entries, 0.0, source_idx)

        # Weights (init_val is ignored now that we use mean/copy)
        self.w_below = _expand_tensor(self.w_below, dim, new_entries, 0.1, source_idx)
        self.w_above = _expand_tensor(self.w_above, dim, new_entries, 0.5, source_idx)

        # If expanding excipients (dim 2), we must also resize thresholds
        if dim == 2:
            # Thresholds is 1D [n_excipients], so we expand dim 0 of the thresholds tensor
            self.thresholds = _expand_tensor(
                self.thresholds, 0, new_entries, 1.0, source_idx
            )

    def forward(self, p_idx, r_idx, e_idx, raw_concentration):
        scores_tensor = cast(torch.Tensor, self.static_scores)
        base_score = scores_tensor[p_idx, r_idx, e_idx].unsqueeze(1)

        # 1. Unclamped Delta (Safe range)
        d = torch.clamp(self.delta[p_idx, r_idx, e_idx], -5.0, 5.0).unsqueeze(1)

        # 2. BULLETPROOF THRESHOLD (The "Anti-NaN" Fix)
        # Use abs() to force positivity and clamp(min=0.1) to prevent division by zero
        thresh = self.thresholds[e_idx].abs().clamp(min=0.1).unsqueeze(1)

        w_b = self.w_below[p_idx, r_idx, e_idx].unsqueeze(1)
        w_a = self.w_above[p_idx, r_idx, e_idx].unsqueeze(1)

        # 3. Calculation
        conc_ratio = raw_concentration / thresh

        # Clamp sharpness to prevent sigmoid overflow/underflow
        s = torch.clamp(self.sharpness, min=1.0, max=20.0)
        gate = torch.sigmoid(s * (conc_ratio - 1.0))

        effect_below = torch.tanh(conc_ratio) * w_b
        effect_above = torch.log1p(conc_ratio) * w_a

        conc_term = ((1 - gate) * effect_below) + (gate * effect_above)

        result = (base_score + d) * conc_term
        return result, {"gate": gate, "conc_term": conc_term}


# class LearnablePhysicsPrior(nn.Module):
#     """
#     Implements the learnable weighted physics prior logic.
#     Scores are initialized by the Model class using the config priors.
#     """

#     def __init__(self, n_classes, n_regimes, n_excipients):
#         super().__init__()
#         self.register_buffer(
#             "static_scores", torch.zeros(n_classes, n_regimes, n_excipients)
#         )
#         self.delta = nn.Parameter(torch.zeros(n_classes, n_regimes, n_excipients))
#         self.w_L = nn.Parameter(torch.ones(n_classes, n_regimes, n_excipients) * 0.5)
#         self.w_H = nn.Parameter(torch.ones(n_classes, n_regimes, n_excipients) * 1.0)

#     def expand_indices(self, dim: int, new_entries: int = 1):
#         """Dynamically resizes internal tensors."""
#         device = self.delta.device

#         def _expand_param(param, d, n, init_val=0.0):
#             shape = list(param.shape)
#             shape[d] = n
#             new_data = torch.ones(shape, device=device, dtype=param.dtype) * init_val
#             res = torch.cat([param, new_data], dim=d)
#             return nn.Parameter(res)

#         def _expand_buffer(buff, d, n, init_val=0.0):
#             shape = list(buff.shape)
#             shape[d] = n
#             new_data = torch.ones(shape, device=device, dtype=buff.dtype) * init_val
#             return torch.cat([buff, new_data], dim=d)

#         self.static_scores = _expand_buffer(self.static_scores, dim, new_entries, 0.0)
#         self.delta = _expand_param(self.delta, dim, new_entries, 0.0)
#         self.w_L = _expand_param(self.w_L, dim, new_entries, 0.5)
#         self.w_H = _expand_param(self.w_H, dim, new_entries, 1.0)

#     def forward(self, p_idx, r_idx, e_idx, e_low_norm, e_high_norm):
#         scores_tensor = cast(torch.Tensor, self.static_scores)
#         score = scores_tensor[p_idx, r_idx, e_idx]
#         d = torch.clamp(self.delta[p_idx, r_idx, e_idx], min=-2.0, max=2.0)
#         wl = self.w_L[p_idx, r_idx, e_idx]
#         wh = self.w_H[p_idx, r_idx, e_idx]

#         el = e_low_norm.view(-1)
#         eh = torch.tanh(e_high_norm.view(-1))
#         base_term = score + d
#         conc_term = (wl * el) + (wh * eh)
#         result = base_term * conc_term
#         details = {
#             "static_score": score,
#             "delta": d,
#             "w_L": wl,
#             "w_H": wh,
#             "e_low_norm": el,
#             "e_high_norm": eh,
#             "base_term": base_term,
#             "conc_term": conc_term,
#             "result": result,
#         }
#         return result.unsqueeze(1), details


class EmbeddingDropout(nn.Module):
    """Drop entire embedding vectors, not individual dimensions."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand(x.size(0), 1, device=x.device) > self.p).float()
        return x * mask


class ResidualBlock(nn.Module):
    """Standard Residual Block with Pre-Activation structure."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x + residual


class ResidualAdapter(nn.Module):
    """
    Enhanced Residual Adapter for VisQ.AI.
    Increased capacity to handle large shifts in log-space residuals
    caused by new protein types or classes.
    """

    def __init__(self, numeric_dim, cat_dims, embed_dim=32):  # Increased from 16
        super().__init__()
        # Project numeric features to a higher dimension to match categorical resolution
        self.num_proj = nn.Sequential(
            nn.Linear(numeric_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU()
        )

        # Embeddings for new categorical levels
        self.embeddings = nn.ModuleList([nn.Embedding(n, embed_dim) for n in cat_dims])

        input_dim = embed_dim * (len(cat_dims) + 1)

        # Deep residual mapping with bottleneck and normalization
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # Increased from 64
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # Increased from 32
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(TARGETS)),
        )

    def forward(self, x_num, x_cat):
        # Generate embeddings for all categorical features
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]

        # Process numeric features through local projection
        num_emb = self.num_proj(x_num)

        # Global concatenation of features
        x = torch.cat(embs + [num_emb], dim=1)

        # Calculate the residual delta
        return self.net(x)
