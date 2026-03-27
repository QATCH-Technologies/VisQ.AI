"""
models.py
=========
Neural network architectures for the CBM-CNP viscosity prediction system.

Architecture
------------
Both model types are built from three named sub-modules that map cleanly
onto the three stages of the Conditional Neural Process:

    ContextEncoder      : raw context points  ->  pooled latent r
    PhysicsBottleneck   : latent r            ->  gated concept vector c  (CBM only)
    ViscosityDecoder    : (c or r, shear, static) ->  log-viscosity prediction

Classes
-------
AttentionPool
    Multi-head attention pooler with post-LayerNorm stabilisation.
ContextEncoder
    Pre-pool MLP + AttentionPool. Encodes variable-length context into r.
PhysicsBottleneck
    Linear projection + per-concept activations + learned sparsity gates.
ViscosityDecoder
    MLP that combines a memory vector with query shear and static features.
CrossSampleCNP
    Baseline CNP (no bottleneck). Memory vector = r from ContextEncoder.
ConceptBottleneckCNP
    Hard CBM. Memory vector = c from PhysicsBottleneck.

Helpers
-------
_forward(model, ctx, qx, qstat)
    Uniform forward call returning (predictions, concepts | None).
_encode_latent(model, context_tensor)
    Returns raw pre-concept latent r for triplet / diagnostic use.

External API contract (consumed by trainer.py / diagnostics.py)
---------------------------------------------------------------
model.encoder(ctx)                   -> [B, N_pts, latent_dim]  # pre-pool MLP
model.encode_memory(ctx)             -> [B, memory_dim]
model.decode_from_memory(c, qx, qs)  -> [B, Q, 1]
model.encode_latent(ctx)             -> [B, latent_dim]          # CBM only
model.forward(ctx, qx, qs)           -> (pred, c | None)
model.concept_gate_logits            -> Parameter [n_concepts]   # for sparsity loss
model.concept_gates()                -> Tensor [n_concepts]      # detached
model.decov_loss(concepts)           -> scalar Tensor
model._concept_activations           -> list[str]
model.concept_names                  -> list[str]
model.n_concepts                     -> int
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cb_cnp.constants import (
    CONCEPT_ACTIVATIONS,
    CONCEPT_NAMES,
    N_CONCEPTS_SUPERVISED,
)


# ============================================================
# Stage 0 — Attention pooler (shared by both model types)
# ============================================================


class AttentionPool(nn.Module):
    """
    Compresses a variable-length set of encoded context points into a single
    fixed-size vector via learnable query-based attention.

    Post-attention LayerNorm stabilises the scale of the pooled output,
    preventing downstream concept projections from seeing unbounded inputs.

    Parameters
    ----------
    latent_dim : int
    n_heads    : int  (must divide latent_dim evenly)
    """

    def __init__(self, latent_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, N, latent_dim]

        Returns
        -------
        Tensor [B, latent_dim]
        """
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


# ============================================================
# Stage 1 — Context encoder (context points -> latent r)
# ============================================================


class ContextEncoder(nn.Module):
    """
    Maps a set of context (shear, viscosity, static) points to a single
    fixed-size latent vector r via a point-wise MLP followed by attention pooling.

    Sub-modules
    -----------
    mlp    : point-wise encoder  [B, N, 2+static_dim] -> [B, N, latent_dim]
    pooler : attention pooler    [B, N, latent_dim]   -> [B, latent_dim]

    The ``mlp`` attribute is intentionally exposed because ``trainer.py``
    requires pre-pool encodings ([B, N, latent_dim]) for the intra-group
    cosine consistency loss (FIX-4). It is aliased on the parent model as
    ``model.encoder`` so existing call-sites need no changes.

    Parameters
    ----------
    static_dim : int
    hidden_dim : int
    latent_dim : int
    dropout    : float
    """

    def __init__(
        self,
        static_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.pooler = AttentionPool(latent_dim)

    def forward(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        context_tensor : Tensor [B, N, 2 + static_dim]

        Returns
        -------
        r : Tensor [B, latent_dim]
            Pooled context representation.
        """
        encoded = self.mlp(context_tensor)  # [B, N, latent_dim]
        return self.pooler(encoded)  # [B, latent_dim]


# ============================================================
# Stage 2 — Physics bottleneck (latent r -> concept vector c)
# ============================================================


class PhysicsBottleneck(nn.Module):
    """
    Projects the pooled latent r into an interpretable concept vector c.

    Each concept dimension is bounded by a physics-informed activation:
      - ``"tanh"``    -> [-1, 1]  for signed quantities (e.g. charge polarity,
                                 repulsive vs. attractive self-interaction).
      - ``"sigmoid"`` -> [ 0, 1]  for non-negative intensities (e.g. crowding,
                                 hydrophobicity, ionic screening strength).

    Learned sparsity gates (sigmoid of trainable logits) scale each concept.
    Supervised gates are initialised near-open  (logit ≈ +2.2) so physics
    priors are active from the start. Free/latent gates start near-closed
    (logit ≈ -2.2) to force the model to rely on named concepts first.

    Parameters
    ----------
    latent_dim          : int
    n_concepts          : int         (supervised + free)
    concept_activations : list[str]   ("tanh" or "sigmoid" per dimension)
    n_supervised        : int         (number of supervised concept dimensions)
    """

    def __init__(
        self,
        latent_dim: int,
        n_concepts: int,
        concept_activations: list[str],
        n_supervised: int = N_CONCEPTS_SUPERVISED,
    ) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.concept_activations = concept_activations

        # Linear projection from latent space to raw concept scores
        self.concept_proj = nn.Linear(latent_dim, n_concepts)

        # Gate initialisation: supervised open (+2.2), free closed (-2.2)
        init_logits = torch.empty(n_concepts)
        n_sup = min(n_concepts, n_supervised)
        init_logits[:n_sup] = 2.2
        if n_concepts > n_sup:
            init_logits[n_sup:] = -2.2
        self.concept_gate_logits = nn.Parameter(init_logits)

    def _apply_per_concept_activation(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Apply tanh or sigmoid independently per concept dimension.

        Parameters
        ----------
        raw : Tensor [B, n_concepts]  — unbounded projection output

        Returns
        -------
        Tensor [B, n_concepts]  — each dim in [-1, 1] or [0, 1]
        """
        activated = torch.empty_like(raw)
        for i, act_type in enumerate(self.concept_activations):
            if act_type == "sigmoid":
                activated[:, i] = torch.sigmoid(raw[:, i])
            else:
                activated[:, i] = torch.tanh(raw[:, i])
        return activated

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        r : Tensor [B, latent_dim]

        Returns
        -------
        c : Tensor [B, n_concepts]
            Gated concept vector.
        """
        raw = self.concept_proj(r)
        c_activated = self._apply_per_concept_activation(raw)
        gates = torch.sigmoid(self.concept_gate_logits)
        return c_activated * gates


# ============================================================
# Stage 3 — Viscosity decoder (memory vector + query -> log-viscosity)
# ============================================================


class ConceptGatedDecoder(nn.Module):
    def __init__(self, static_dim, memory_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        # Concept vector generates per-feature scale and shift
        self.film_gen = nn.Linear(memory_dim, static_dim * 2)  # γ and β
        self.mlp = nn.Sequential(
            nn.Linear(1 + static_dim, hidden_dim),  # shear + modulated static
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, memory_vector, query_shear, query_static):
        n_q = query_shear.size(1)
        # Generate per-feature scale (γ) and shift (β) from concepts
        film_params = self.film_gen(memory_vector)  # [B, static_dim*2]
        gamma, beta = film_params.chunk(2, dim=-1)  # each [B, static_dim]
        gamma = gamma.unsqueeze(1).expand(-1, n_q, -1)  # [B, Q, static_dim]
        beta = beta.unsqueeze(1).expand(-1, n_q, -1)
        # Modulate: concept controls HOW static features are interpreted
        modulated = query_static * (1 + gamma) + beta
        x = torch.cat([query_shear, modulated], dim=-1)
        return self.mlp(x)


class ViscosityDecoder(nn.Module):
    """
    Predicts log-viscosity at arbitrary query shear rates from a memory vector.

    The memory vector can be either a concept vector c (CBM) or a raw latent
    r (baseline CNP). The decoder is intentionally unaware of this distinction —
    it concatenates whatever vector it receives with query shear and static
    features, then passes the result through an MLP.

    Parameters
    ----------
    static_dim : int
    memory_dim : int    (n_concepts for CBM, latent_dim for baseline CNP)
    hidden_dim : int
    dropout    : float
    """

    def __init__(
        self,
        static_dim: int,
        memory_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + static_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        memory_vector: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        memory_vector : Tensor [B, memory_dim]
        query_shear   : Tensor [B, Q, 1]
        query_static  : Tensor [B, Q, static_dim]

        Returns
        -------
        Tensor [B, Q, 1]  — predicted log-viscosity
        """
        n_q = query_shear.size(1)
        m_exp = memory_vector.unsqueeze(1).repeat(1, n_q, 1)
        x = torch.cat([query_shear, query_static, m_exp], dim=-1)
        return self.mlp(x)


# ============================================================
# Concept Bottleneck CNP (memory vector = c from PhysicsBottleneck)
# ============================================================


class ConceptBottleneckCNP(nn.Module):
    """
    Hard Concept Bottleneck CNP — orchestrates ContextEncoder,
    PhysicsBottleneck, and ViscosityDecoder into a single trainable model.

    Data flow
    ---------
    context_tensor
        -> ContextEncoder  -> r  [B, latent_dim]
        -> PhysicsBottleneck -> c  [B, n_concepts]
        -> ViscosityDecoder  -> log-viscosity  [B, Q, 1]

    The decoder has **no access to r**. Every bit of context must be
    expressed through the named concept dimensions.

    Sub-modules
    -----------
    context_encoder   : ContextEncoder
    bottleneck        : PhysicsBottleneck
    viscosity_decoder : ViscosityDecoder  (memory_dim = n_concepts)

    ``self.encoder`` is aliased to ``context_encoder.mlp`` so that
    ``trainer.py`` can access pre-pool encodings without any API changes.

    Parameters
    ----------
    static_dim          : int
    hidden_dim          : int
    latent_dim          : int
    n_concepts          : int         (total = supervised + free)
    concept_names       : list[str] | None
    concept_activations : list[str] | None
    dropout             : float
    """

    def __init__(
        self,
        static_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        n_concepts: int = N_CONCEPTS_SUPERVISED,
        concept_names: list[str] | None = None,
        concept_activations: list[str] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.n_concepts = n_concepts

        n_free = max(0, n_concepts - N_CONCEPTS_SUPERVISED)
        n_sup = min(n_concepts, N_CONCEPTS_SUPERVISED)

        self.concept_names: list[str] = (
            concept_names
            if concept_names is not None
            else CONCEPT_NAMES[:n_concepts] + [f"latent_{i}" for i in range(n_free)]
        )

        c_acts: list[str] = (
            concept_activations
            if concept_activations is not None
            else CONCEPT_ACTIVATIONS[:n_sup] + ["tanh"] * n_free
        )

        # --- Sub-modules ---
        self.context_encoder = ContextEncoder(
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.bottleneck = PhysicsBottleneck(
            latent_dim=latent_dim,
            n_concepts=n_concepts,
            concept_activations=c_acts,
            n_supervised=N_CONCEPTS_SUPERVISED,
        )
        self.concept_gated_decoder = ConceptGatedDecoder(
            static_dim=static_dim,
            memory_dim=n_concepts,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Alias: trainer.py calls model.encoder(ctx).mean(dim=1) for the
        # pre-pooled intra-group cosine consistency loss (FIX-4).
        self.encoder = self.context_encoder.mlp

    # ----------------------------------------------------------
    # Standard CNP API
    # ----------------------------------------------------------

    def encode_memory(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """Returns the gated concept vector c [B, n_concepts]."""
        r = self.context_encoder(context_tensor)
        return self.bottleneck(r)

    def decode_from_memory(
        self,
        concept_vector: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts log-viscosity from a (possibly intervened) concept vector."""
        return self.concept_gated_decoder(concept_vector, query_shear, query_static)

    def forward(
        self,
        context_tensor: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass for training.

        Returns
        -------
        pred : Tensor [B, Q, 1]
        c    : Tensor [B, n_concepts]  — concept vector for supervision losses
        """
        c = self.encode_memory(context_tensor)
        pred = self.decode_from_memory(c, query_shear, query_static)
        return pred, c

    def encode_latent(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """
        Returns raw pre-concept latent r [B, latent_dim].
        Used by triplet loss, latent variance diagnostic, and norm penalty.
        """
        return self.context_encoder(context_tensor)

    # ----------------------------------------------------------
    # CBM causal intervention API
    # ----------------------------------------------------------

    def intervene(
        self,
        context_tensor: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
        concept_idx: int | list[int],
        concept_value: float,
    ) -> torch.Tensor:
        """
        Causal do-intervention: clamp concept_idx to concept_value, re-decode.

        Implements do(c_i = v) in the Pearl causal sense — not correlation.

        Parameters
        ----------
        concept_idx   : int or list[int]
        concept_value : float  (should be in the activation range of that concept)

        Returns
        -------
        Tensor [B, Q, 1]
        """
        c = self.encode_memory(context_tensor)
        c_mod = c.clone()
        if isinstance(concept_idx, int):
            concept_idx = [concept_idx]
        for idx in concept_idx:
            c_mod[:, idx] = concept_value
        return self.decode_from_memory(c_mod, query_shear, query_static)

    # ----------------------------------------------------------
    # CBM accessors (used by trainer.py and diagnostics.py)
    # ----------------------------------------------------------

    @property
    def concept_gate_logits(self) -> nn.Parameter:
        """Gate logits forwarded from PhysicsBottleneck for the sparsity loss."""
        return self.bottleneck.concept_gate_logits

    @property
    def _concept_activations(self) -> list[str]:
        """Per-concept activation type list forwarded from PhysicsBottleneck."""
        return self.bottleneck.concept_activations

    def concept_gates(self) -> torch.Tensor:
        """Return current gate values [n_concepts] in [0, 1] (detached)."""
        return torch.sigmoid(self.bottleneck.concept_gate_logits).detach()

    def decov_loss(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        DeCov (Decorrelation) loss on a batch of concept activations.

        Penalises off-diagonal elements of the concept covariance matrix to
        encourage each dimension to explain unique variance.

        Parameters
        ----------
        concepts : Tensor [B, n_concepts]

        Returns
        -------
        Scalar Tensor — Frobenius norm of the off-diagonal covariance.
        """
        if concepts.size(0) < 2:
            return torch.tensor(0.0, device=concepts.device)
        c_centered = concepts - concepts.mean(dim=0, keepdim=True)
        cov = (c_centered.T @ c_centered) / (concepts.size(0) - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag**2).sum() / (self.n_concepts * (self.n_concepts - 1))


# ============================================================
# Model-agnostic helpers (consumed by trainer.py)
# ============================================================


def _forward(
    model: CrossSampleCNP | ConceptBottleneckCNP,
    ctx: torch.Tensor,
    qx: torch.Tensor,
    qstat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Uniform forward pass for both model types.

    Returns (predictions [B, Q, 1], concepts [B, n_concepts] or None).
    """
    return model(ctx, qx, qstat)


def _encode_latent(
    model: CrossSampleCNP | ConceptBottleneckCNP,
    context_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Return raw pre-concept latent r [B, latent_dim].

    Uses ``encode_latent()`` for ConceptBottleneckCNP and falls back to
    ``encode_memory()`` for CrossSampleCNP (where r = memory vector).
    """
    if isinstance(model, ConceptBottleneckCNP):
        return model.encode_latent(context_tensor)
    return model.encode_memory(context_tensor)
