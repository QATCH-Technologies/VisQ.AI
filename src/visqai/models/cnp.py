"""Cross-Sample Conditional Neural Process architecture for conditional
viscosity prediction.

The module provides the shared neural architecture used by model training and
inference. A context encoder maps observed samples into a latent representation
that is aggregated by :class:`AttentionPool`, while the decoder separates
query-only prediction from context-dependent correction.

The decoder is composed of two heads:

    * `prior_head` predicts the feature-conditioned zero-shot baseline from
      query shear and static features without access to the context latent.
    * `correction_head` predicts a context-dependent residual from the query
      features and pooled context representation.

The final layer of `correction_head` is initialized to zero so that the
initial prediction is exactly the query-only prior. This architectural
separation prevents context information from directly altering the zero-shot
path and allows the correction head to learn residual structure independently.

The resulting prediction has the form::

    prediction = prior_head(query) + correction_head(query, context)

The split decoder also exposes separate prior and correction outputs for
training procedures that optimize the prior against the target and the
correction against the residual relative to that prior.

Checkpoint compatibility:
    The decoder parameter namespace uses `prior_head` and
    `correction_head` instead of the former combined `decoder` module.
    Checkpoints created before this decoder split therefore require
    retraining and are not directly compatible with the current architecture.
"""

from __future__ import annotations

import torch
from typing import cast
import torch.nn as nn


class AttentionPool(nn.Module):
    """Aggregate encoded context samples into a fixed-size latent memory vector.

    Uses multi-head self-attention with a learned query token to summarize a
    variable number of context samples into a single latent representation.
    Layer normalization is applied to the resulting representation to control
    its scale before it is consumed by downstream objectives and decoder
    components.

    Args:
        latent_dim: Dimensionality of each encoded context representation and
            of the resulting pooled memory vector.
        n_heads: Number of attention heads used by the multi-head attention
            layer.
    """

    def __init__(self, latent_dim: int, n_heads: int = 4) -> None:
        """Initialize the attention-based context pooling module.

        Args:
            latent_dim: Dimensionality of the input context representations and
                the pooled latent memory vector.
            n_heads: Number of attention heads used by the multi-head attention
                layer.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate a variable-sized context sequence into one latent vector.

        Args:
            x: Context representations with shape
                `(batch_size, n_context, latent_dim)`.

        Returns:
            A pooled context representation with shape
            `(batch_size, latent_dim)`. The representation is layer-normalized
            to control its magnitude while preserving the learned directional
            information used by downstream objectives.
        """
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class CrossSampleCNP(nn.Module):
    """Conditional Neural Process for cross-sample viscosity prediction.

    Encodes observed context samples into a shared latent memory
    representation and combines that representation with query-specific
    shear and static features to produce viscosity predictions.

    The decoder is explicitly split into a context-independent prior and a
    context-dependent residual correction. The prior provides the model's
    zero-shot prediction, while the correction head captures information
    learned from the supplied context. Because the correction head's final
    layer is zero-initialized, the model initially behaves as the prior alone.

    Args:
        static_dim: Number of static formulation features supplied for each
            query or context sample.
        hidden_dim: Width of the hidden layers in the context encoder and
            context-dependent correction head.
        latent_dim: Dimensionality of the pooled context representation.
        dropout: Dropout probability applied within the encoder and decoder
            heads.
    """

    def __init__(
        self,
        static_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the cross-sample conditional neural process.

        Args:
            static_dim: Number of static feature dimensions supplied alongside the
                query shear value and context observations.
            hidden_dim: Width of the hidden layers used by the context encoder and
                context-dependent correction head.
            latent_dim: Dimensionality of the pooled context representation used as
                the model's latent memory vector.
            dropout: Dropout probability applied within the encoder and decoder
                heads.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.pooler = AttentionPool(latent_dim)

        # Build the feature-only prior head.
        #
        # The prior receives only the query shear and static features and has no access
        # to the pooled context representation. It therefore defines the model's
        # context-independent, zero-shot prediction path and cannot be directly
        # affected by an uninformative or misleading context.

        # The hidden width is set to `hidden_dim + latent_dim` rather than
        # `hidden_dim`. This preserves the first-layer capacity that the previous
        # combined decoder provided to the query-dependent mapping: the previous
        # decoder accepted the latent context as an additional `latent_dim`-sized
        # input while using the same hidden width for both context-independent and
        # context-dependent behavior. Matching that capacity helps isolate the effect
        # of separating the prior and correction paths from any reduction in model
        # capacity.
        #
        # The prior is trained directly against the target, while the separate
        # correction head models context-dependent residual structure.

        prior_hidden_dim = hidden_dim + latent_dim
        self.prior_head = nn.Sequential(
            nn.Linear(1 + static_dim, prior_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prior_hidden_dim, prior_hidden_dim),
            nn.ReLU(),
            nn.Linear(prior_hidden_dim, 1),
        )

        # Build the context-dependent correction head.
        #
        # The correction head is the only prediction path that receives the pooled
        # context representation `r`. It models the context-dependent residual on top
        # of the feature-only prior, so the final prediction is the sum of the prior
        # and correction terms.
        #
        # The final linear layer is zero-initialized, causing the correction to start
        # at exactly zero for all inputs. This makes the initial model behavior equal
        # to the feature-only prior and allows the context-dependent path to learn only
        # deviations from that baseline rather than having to learn the full prediction
        # from the outset.
        self.correction_head = nn.Sequential(
            nn.Linear(1 + static_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        last_layer = cast(nn.Linear, self.correction_head[-1])
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def _decode_split(
        self, query_shear: torch.Tensor, query_static: torch.Tensor, r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode query predictions into prior and context correction terms.

        Args:
            query_shear: Query shear-rate features with shape
                `(batch_size, n_queries, 1)`.
            query_static: Query static features with shape
                `(batch_size, n_queries, static_dim)`.
            r: Pooled context representation with shape
                `(batch_size, latent_dim)`.

        Returns:
            A tuple `(prior, correction)` containing the query-only prior
            prediction and the context-dependent residual correction. Both
            tensors have shape `(batch_size, n_queries, 1)`.

        Notes:
            The context representation is expanded across query points so
            that each query is decoded using the same context memory while
            retaining its own shear and static features.
        """
        n_queries = query_shear.size(1)
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)
        prior = self.prior_head(torch.cat([query_shear, query_static], dim=-1))
        correction = self.correction_head(
            torch.cat([query_shear, query_static, r_expanded], dim=-1)
        )
        return prior, correction

    def forward(
        self,
        context_tensor: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> torch.Tensor:
        """Generate separate prior and context-correction predictions.

        Args:
            context_tensor: Features for the observed context samples.
            query_shear: Shear-rate features for the query samples.
            query_static: Static formulation features for the query samples.

        Returns:
            A tuple `(prior, correction)` containing the query-only prior
            and context-dependent correction tensors without summing them.

        Notes:
            This method is intended for training or diagnostics that need to
            optimize or inspect the prior and residual components separately.
        """
        r = self.pooler(self.encoder(context_tensor))
        prior, correction = self._decode_split(query_shear, query_static, r)
        return prior + correction

    def forward_split(
        self,
        context_tensor: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate separate prior and context-correction predictions.

        Args:
            context_tensor: Features for the observed context samples.
            query_shear: Shear-rate features for the query samples.
            query_static: Static formulation features for the query samples.

        Returns:
            A tuple `(prior, correction)` containing the query-only prior
            and context-dependent correction tensors without summing them.

        Notes:
            This method is intended for training or diagnostics that need to
            optimize or inspect the prior and residual components separately.
        """
        r = self.pooler(self.encoder(context_tensor))
        return self._decode_split(query_shear, query_static, r)

    def encode_memory(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """Encode context samples into a pooled latent memory representation.

        Args:
            context_tensor: Features for the observed context samples.

        Returns:
            A tensor containing the fixed-size latent representation produced
            by encoding and attention-pooling the supplied context samples.
        """
        return self.pooler(self.encoder(context_tensor))

    def decode_from_memory(
        self,
        memory_vector: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> torch.Tensor:
        """Decode queries using a precomputed context memory representation.

        Args:
            memory_vector: Pooled latent representation of the context,
                typically produced by :meth:`encode_memory`.
            query_shear: Shear-rate features for the query samples.
            query_static: Static formulation features for the query samples.

        Returns:
            The combined prior and context-dependent correction prediction for
            each query sample.
        """
        prior, correction = self._decode_split(query_shear, query_static, memory_vector)
        return prior + correction

    def decode_from_memory_split(
        self,
        memory_vector: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode queries into separate prior and correction components.

        Args:
            memory_vector: Precomputed pooled latent representation of the
                context.
            query_shear: Shear-rate features for the query samples.
            query_static: Static formulation features for the query samples.

        Returns:
            A tuple `(prior, correction)` containing the query-only prior
            and context-dependent correction predictions without combining
            them.
        """
        return self._decode_split(query_shear, query_static, memory_vector)
