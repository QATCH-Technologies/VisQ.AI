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
import torch.nn.functional as F  # Add this import

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
        self.w_below = nn.Parameter(torch.ones(n_classes, n_regimes, n_excipients) * 0.1)
        self.w_above = nn.Parameter(torch.ones(n_classes, n_regimes, n_excipients) * 0.5)
        
        # --- NEW: Linear Term Weight ---
        # Initialize small to avoid disrupting initial training
        self.w_linear = nn.Parameter(torch.zeros(n_classes, n_regimes, n_excipients)) 
        
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
        self.w_linear = _expand_tensor(self.w_linear, dim, new_entries, 0.0, source_idx)
        # If expanding excipients (dim 2), we must also resize thresholds
        if dim == 2:
            # Thresholds is 1D [n_excipients], so we expand dim 0 of the thresholds tensor
            self.thresholds = _expand_tensor(
                self.thresholds, 0, new_entries, 1.0, source_idx
            )

    def forward(self, p_idx, r_idx, e_idx, raw_concentration):
        scores_tensor = cast(torch.Tensor, self.static_scores)
        base_score = scores_tensor[p_idx, r_idx, e_idx].unsqueeze(1)

        d = torch.clamp(self.delta[p_idx, r_idx, e_idx], -5.0, 5.0).unsqueeze(1)
        
        # Existing safe threshold logic
        thresh = self.thresholds[e_idx].abs().clamp(min=0.1).unsqueeze(1)

        w_b = self.w_below[p_idx, r_idx, e_idx].unsqueeze(1)
        w_a = self.w_above[p_idx, r_idx, e_idx].unsqueeze(1)

        # --- FIX STARTS HERE ---
        # 1. Project standardized input to positive space. 
        #    Softplus is differentiable and mimics "0 concentration" behavior smoothly.
        #    You can also add a small epsilon to be safe.
        safe_concentration = F.softplus(raw_concentration) 
        conc_ratio = safe_concentration / thresh
        # --- FIX ENDS HERE ---

        s = torch.clamp(self.sharpness, min=1.0, max=20.0)
        gate = torch.sigmoid(s * (conc_ratio - 1.0))

        # 1. Saturation Effect (Logarithmic)
        effect_below = torch.tanh(conc_ratio) * w_b
        effect_above = torch.log1p(conc_ratio) * w_a 
        
        # 2. Crowding Effect (Linear in Log-Space)
        # This allows modeling: Viscosity ~ exp(k * Conc)
        w_lin = self.w_linear[p_idx, r_idx, e_idx].unsqueeze(1)
        effect_linear = conc_ratio * w_lin

        # Combine: (Gated Saturation) + (Linear Crowding)
        conc_term = ((1 - gate) * effect_below) + (gate * effect_above) + effect_linear

        result = (base_score + d) * conc_term
        return result, {"gate": gate, "conc_term": conc_term}


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


# class ResidualAdapter(nn.Module):
#     """
#     "Bottleneck" Residual Adapter for VisQ.AI.

#     OPTIMIZED FOR: Small Data (~500 samples)
#     STRATEGY: High bias, low variance. Forces the model to find
#               simple, robust physical corrections rather than memorizing noise.
#     """

#     def __init__(self, numeric_dim, cat_dims, embed_dim=4):  # REDUCED: 32 -> 4
#         super().__init__()

#         # 1. REMOVED: numeric projection (self.num_proj).
#         # Reason: Projecting 8 floats to 32 dimensions adds parameters
#         # without adding information. We use raw numeric values now.

#         # 2. SHRUNK: Embeddings.
#         # We only need enough dimensions to separate "High Salt" from "Low Salt".
#         # 4 dimensions is plenty for this.
#         self.embeddings = nn.ModuleList([nn.Embedding(n, embed_dim) for n in cat_dims])

#         # Calculate input size: Raw Numeric + (Cats * Embed Size)
#         input_dim = numeric_dim + (len(cat_dims) * embed_dim)

#         # 3. FLATTENED: The Network.
#         # Old: 128 -> 64 -> 32 -> Out (4 layers, ~50k params)
#         # New: 16 -> Out (2 layers, ~1k params)
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 16),  # Bottleneck layer
#             nn.LayerNorm(16),  # Stability
#             nn.ReLU(),  # Non-linearity
#             nn.Dropout(0.2),  # Higher dropout for small data
#             nn.Linear(16, len(TARGETS)),  # Direct mapping to targets
#         )

#         # Initialize weights to be near-zero so the adapter starts
#         # by outputting ~0 residual (trusting the PINN backbone initially).
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.net.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#         # specifically zero out the last layer so initial predictions change little
#         nn.init.constant_(self.net[-1].weight, 0)

#     def forward(self, x_num, x_cat):
#         # Generate small embeddings
#         embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]

#         # Concatenate raw numeric features with embeddings
#         # Note: x_num is used directly now!
#         x = torch.cat(embs + [x_num], dim=1)


#         return self.net(x)
# File: visq_ml/layers.py
# class ResidualAdapter(nn.Module):
#     """
#     Bottleneck Adapter optimized for small-data adaptation.
#     """

#     def __init__(self, numeric_dim, cat_dims, embed_dim=4):  # Low dim embeddings
#         super().__init__()

#         self.embeddings = nn.ModuleList([nn.Embedding(n, embed_dim) for n in cat_dims])

#         # Input = Raw Numerics + Concatenated Embeddings
#         input_dim = numeric_dim + (len(cat_dims) * embed_dim)

#         # Shallow Network: Only 1 hidden layer
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 16),  # Bottleneck (High Bias/Low Variance)
#             nn.LayerNorm(16),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(16, 5),  # Maps directly to 5 targets
#         )

#     def forward(self, x_num, x_cat):
#         embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
#         # Use raw numeric features + embeddings
#         x = torch.cat(embs + [x_num], dim=1)
#         return self.net(x)
# File: ml/visq_ml/layers.py



# class ResidualAdapter(nn.Module):
#     """
#     Bottleneck Residual Adapter.
#     Simplified inputs to preserve directionality of standardized features.
#     """

#     def __init__(self, numeric_dim, cat_dims, embed_dim=4):
#         super().__init__()
        
#         self.embeddings = nn.ModuleList([nn.Embedding(n, embed_dim) for n in cat_dims])
        
#         # --- FIX: Remove x_log dimensions ---
#         # Input = (Raw Numerics) + (Embeddings)
#         # We removed the "+ numeric_dim" that was previously there for the log features
#         input_dim = numeric_dim + (len(cat_dims) * embed_dim)

#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 32),
#             nn.LayerNorm(32),
#             nn.ReLU(),
#             nn.Dropout(0.2), 
#             nn.Linear(32, 5),
#         )

#     def forward(self, x_num, x_cat):
#         # 1. Embed Categoricals
#         embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        
#         # --- FIX: Removed Broken Log-Transform ---
#         # x_num is standardized (Z-score). We use it directly to preserve 
#         # the sign (directionality) of the features.
        
#         # 2. Concatenate: Embeddings + Raw Numerics
#         x = torch.cat(embs + [x_num], dim=1)
        
#         return self.net(x)

class ResidualAdapter(nn.Module):
    """
    Bottleneck MLP Adapter.
    Restored capacity for full-dataset learning.
    """

    def __init__(self, numeric_dim, cat_dims, embed_dim=4):
        super().__init__()
        
        self.embeddings = nn.ModuleList([nn.Embedding(n, embed_dim) for n in cat_dims])
        
        # Input = (Raw Numerics) + (Embeddings)
        input_dim = numeric_dim + (len(cat_dims) * embed_dim)

        # Standard MLP Structure
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(32, 5),  # Index [-1] is this layer
        )

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.net(x)