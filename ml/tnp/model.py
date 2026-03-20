"""
model.py
========
TransformerNP architecture for viscosity prediction.  [TNP-1 through TNP-3]

Split-static architecture  [TNP-ATTN-6]
-----------------------------------------
Context encoder  : input = 2 + static_ctx_dim  (full features, protein identity included)
Query encoder    : input = 1 + static_qry_dim  (reduced features, NO protein identity)
Decoder          : input = 1 + static_qry_dim + latent_dim

By excluding Protein_type and Protein_class_type from the query static, the model
cannot infer protein identity from the query alone — it must attend selectively to
context tokens (which DO carry protein identity) to predict accurately.  This makes
selective cross-attention necessary and learnable.

Architecture overview
---------------------
context_tensor [B, N_ctx, 2+static_ctx_dim]
    -> context_encoder MLP -> LayerNorm -> ctx_enc [B, N_ctx, latent_dim]

  -- OR -- encode_context_samples(ctx_items_list)  [TNP-ATTN-4]
    per-sample encode + mean-pool -> [1, N_samples, latent_dim]

[query_shear | query_static_qry] [B, N_q, 1+static_qry_dim]
    -> query_encoder MLP -> q_enc [B, N_q, latent_dim]

cross_attention(Q=q_enc / T_learned, K=V=ctx_enc)  [TNP-ATTN-1]
    -> attended -> LayerNorm -> FFN -> LayerNorm    [TNP-ATTN-2/3]

decoder([query_shear | query_static_qry | attended]) -> prediction [B, N_q, 1]
"""

import math

import torch
import torch.nn as nn


class TransformerNP(nn.Module):
    """
    Transformer Neural Process for biopharmaceutical viscosity prediction.

    Args:
        static_ctx_dim (int):     Dimensionality of the full context static vector
                                  (includes protein identity one-hots).
        static_qry_dim (int):     Dimensionality of the reduced query static vector
                                  (excludes Protein_type / Protein_class_type).
                                  If None, defaults to static_ctx_dim (backward compat).
        hidden_dim (int):         MLP layer width.
        latent_dim (int):         Cross-attention space dim; must be divisible by n_heads.
        n_heads (int):            Attention heads [TNP-3].
        dropout (float):          Dropout inside MLPs and cross-attention.
        init_temperature (float): Initial learned temperature [TNP-ATTN-1].
        T_min (float):            Hard lower bound on learned temperature.
    """

    def __init__(
        self,
        static_ctx_dim: int,
        static_qry_dim: int = None,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.0,
        init_temperature: float = 0.5,
        T_min: float = 0.05,
    ) -> None:
        super().__init__()

        # Backward compat: if static_qry_dim not given, use same as ctx
        if static_qry_dim is None:
            static_qry_dim = static_ctx_dim

        assert (
            latent_dim % n_heads == 0
        ), f"latent_dim ({latent_dim}) must be divisible by n_heads ({n_heads})"
        self.static_ctx_dim = static_ctx_dim
        self.static_qry_dim = static_qry_dim
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.T_min = T_min

        # [TNP-ATTN-1] Learned temperature in log-space.
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(init_temperature), dtype=torch.float32)
        )

        # Context encoder: (shear, visc, static_ctx) -> latent
        self.context_encoder = nn.Sequential(
            nn.Linear(2 + static_ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # [TNP-ATTN-2] Normalise context encodings before cross-attention.
        self.ctx_enc_norm = nn.LayerNorm(latent_dim)

        # Query encoder: (shear, static_qry) -> latent
        # Uses static_qry_dim — NO protein identity features.  [TNP-ATTN-6]
        self.query_encoder = nn.Sequential(
            nn.Linear(1 + static_qry_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            latent_dim, n_heads, batch_first=True, dropout=dropout
        )
        self.cross_attn_norm = nn.LayerNorm(latent_dim)

        # [TNP-ATTN-3] Post-attention FFN sublayer.
        self.attn_ffn = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.attn_ffn_norm = nn.LayerNorm(latent_dim)

        # Decoder: (shear, static_qry, attended) -> prediction
        # Uses static_qry_dim so decoder also cannot read protein identity.
        self.decoder = nn.Sequential(
            nn.Linear(1 + static_qry_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Diagnostic
    # ------------------------------------------------------------------

    def get_temperature(self) -> float:
        with torch.no_grad():
            return float(torch.exp(self.log_temperature).clamp(self.T_min, 1.0).item())

    # ------------------------------------------------------------------
    # [TNP-ATTN-4] Sample-level context encoding
    # ------------------------------------------------------------------

    def encode_context_samples(self, ctx_items_list: list) -> torch.Tensor:
        """
        Encode a list of per-sample raw tensors into one latent token per sample.

        Args:
            ctx_items_list: list of [n_pts_i, 2+static_ctx_dim] tensors on device.

        Returns:
            [1, N_samples, latent_dim] — LayerNorm'd.
        """
        tokens = []
        for item in ctx_items_list:
            enc = self.context_encoder(item)  # [n_pts, latent_dim]
            token = enc.mean(dim=0, keepdim=True)  # [1, latent_dim]
            tokens.append(token)
        stacked = torch.cat(tokens, dim=0).unsqueeze(0)  # [1, N_samples, latent_dim]
        return self.ctx_enc_norm(stacked)

    # ------------------------------------------------------------------
    # Internal cross-attention + FFN block
    # ------------------------------------------------------------------

    def _attend(self, ctx_enc, q_enc, temperature_override):
        if temperature_override is not None:
            T = max(float(temperature_override), self.T_min)
        else:
            T = torch.exp(self.log_temperature).clamp(self.T_min, 1.0)

        attended, attn_weights = self.cross_attn(q_enc / T, ctx_enc, ctx_enc)
        attended = self.cross_attn_norm(attended + q_enc)
        attended = self.attn_ffn_norm(attended + self.attn_ffn(attended))
        return attended, attn_weights

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        context_tensor: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
        temperature=None,
        ctx_is_encoded: bool = False,
    ):
        """
        Args:
            context_tensor: [B, N_ctx, 2+static_ctx_dim]  (ctx_is_encoded=False)
                            [B, N_ctx, latent_dim]         (ctx_is_encoded=True)
            query_shear:    [B, N_q, 1]
            query_static:   [B, N_q, static_qry_dim]  -- reduced, NO protein identity
            temperature:    None -> use learned; float -> override.
            ctx_is_encoded: True -> skip context_encoder (already from encode_context_samples).

        Returns:
            pred [B, N_q, 1], attn_weights [B, N_q, N_ctx]
        """
        if ctx_is_encoded:
            ctx_enc = context_tensor
        else:
            ctx_enc = self.ctx_enc_norm(self.context_encoder(context_tensor))

        q_enc = self.query_encoder(torch.cat([query_shear, query_static], dim=-1))
        attended, attn_weights = self._attend(ctx_enc, q_enc, temperature)

        dec_in = torch.cat([query_shear, query_static, attended], dim=-1)
        return self.decoder(dec_in), attn_weights

    # ------------------------------------------------------------------
    # Memory API
    # ------------------------------------------------------------------

    def encode_memory(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode a raw point-level context tensor [B, N_pts, 2+static_ctx_dim].
        Returns [B, N_pts, latent_dim] LayerNorm'd.
        For the preferred sample-level approach use encode_context_samples().
        """
        return self.ctx_enc_norm(self.context_encoder(context_tensor))

    def decode_from_memory(
        self,
        ctx_encodings: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
        temperature=None,
    ) -> torch.Tensor:
        """
        query_static must be the REDUCED static vector (static_qry_dim).
        ctx_encodings must already be in latent space (from encode_context_samples
        or encode_memory).
        """
        q_enc = self.query_encoder(torch.cat([query_shear, query_static], dim=-1))
        attended, _ = self._attend(ctx_encodings, q_enc, temperature)
        return self.decoder(torch.cat([query_shear, query_static, attended], dim=-1))

    # ------------------------------------------------------------------
    # Latent mean for triplet / consistency / norm losses
    # ------------------------------------------------------------------

    def encode_latent_mean(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """
        Mean-pooled, LayerNorm'd encoding [B, latent_dim] from raw context tensor.
        context_tensor: [B, N_pts, 2+static_ctx_dim]
        """
        return self.ctx_enc_norm(self.context_encoder(context_tensor)).mean(dim=1)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def _forward(model, ctx, qx, qstat, ctx_is_encoded: bool = False):
    """
    qstat must be the REDUCED query static (static_qry_dim), not the full static.
    """
    return model(ctx, qx, qstat, ctx_is_encoded=ctx_is_encoded)


def _encode_latent(model, context_tensor):
    """Mean-pooled latent [B, latent_dim] from raw [B, N_pts, 2+static_ctx_dim]."""
    return model.encode_latent_mean(context_tensor)
