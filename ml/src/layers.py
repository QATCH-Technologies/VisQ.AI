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

from .config import TARGETS


class LearnablePhysicsPrior(nn.Module):
    """
    Implements the learnable weighted physics prior logic.
    Scores are initialized by the Model class using the config priors.
    """

    def __init__(self, n_classes, n_regimes, n_excipients):
        super().__init__()
        # Buffer for static scores (initialized by Model._init_static_priors)
        self.register_buffer(
            "static_scores", torch.zeros(n_classes, n_regimes, n_excipients)
        )

        # Learnable parameters
        self.delta = nn.Parameter(torch.zeros(n_classes, n_regimes, n_excipients))
        self.w_L = nn.Parameter(torch.ones(n_classes, n_regimes, n_excipients) * 0.5)
        self.w_H = nn.Parameter(torch.ones(n_classes, n_regimes, n_excipients) * 1.0)

    def forward(self, p_idx, r_idx, e_idx, e_low_norm, e_high_norm):
        # Fetch parameters
        scores_tensor = cast(torch.Tensor, self.static_scores)
        score = scores_tensor[p_idx, r_idx, e_idx]
        d = torch.clamp(self.delta[p_idx, r_idx, e_idx], min=-2.0, max=2.0)
        wl = self.w_L[p_idx, r_idx, e_idx]
        wh = self.w_H[p_idx, r_idx, e_idx]

        # Compute components
        el = e_low_norm.view(-1)

        # This prevents the 'explosion' by capping the magnitude of the
        # high-concentration input vector between -1 and 1.
        eh = torch.tanh(e_high_norm.view(-1))

        base_term = score + d
        conc_term = (wl * el) + (wh * eh)
        result = base_term * conc_term
        details = {
            "static_score": score,
            "delta": d,
            "w_L": wl,
            "w_H": wh,
            "e_low_norm": el,
            "e_high_norm": eh,
            "base_term": base_term,
            "conc_term": conc_term,
            "result": result,
        }

        return result.unsqueeze(1), details


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
    Residual Adapter network for domain adaptation.
    Uses a separate embedding set and projection for numeric inputs.
    """

    def __init__(self, numeric_dim, cat_dims, embed_dim=16):
        super().__init__()
        self.num_proj = nn.Linear(numeric_dim, embed_dim)
        self.embeddings = nn.ModuleList([nn.Embedding(n, embed_dim) for n in cat_dims])

        # Output dim is len(TARGETS) imported from config
        self.net = nn.Sequential(
            nn.Linear(embed_dim * (len(cat_dims) + 1), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(TARGETS)),
        )

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        num_emb = self.num_proj(x_num)
        x = torch.cat(embs + [num_emb], dim=1)
        return self.net(x)
