"""
models.py
=========
Neural network architectures for the CBM-CNP viscosity prediction system.

Architecture (Hybrid CBM)
-------------------------
The ConceptBottleneckCNP now uses a *hybrid* prediction pathway:

    ContextEncoder      : raw context points  ->  pooled latent r  [B, latent_dim]
    PhysicsBottleneck   : latent r            ->  gated concept c  [B, n_concepts]  (analysis)
    ViscosityDecoder    : (r, shear, static)  ->  log-viscosity    (MAIN prediction)
    ConceptDecoder      : (c, shear, static)  ->  log-viscosity    (intervention only)

The main decoder receives the full 128-dim latent r, giving it 8x the
information capacity of the previous 16-dim concept bottleneck.  The concept
branch is kept alive via supervision losses and provides interpretability
(heatmaps, interventions) through a lightweight secondary decoder.

External API contract
---------------------
model.encode_memory(ctx)             -> [B, latent_dim]    # r for prediction
model.encode_concepts(ctx)           -> [B, n_concepts]    # c for analysis
model.decode_from_memory(r, qx, qs)  -> [B, Q, 1]         # main decoder
model.decode_from_concepts(c, qx, qs) -> [B, Q, 1]        # concept decoder
model.forward(ctx, qx, qs)           -> (pred, c)          # same 2-tuple API
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
        encoded = self.mlp(context_tensor)
        return self.pooler(encoded)


# ============================================================
# Stage 2 — Physics bottleneck (latent r -> concept vector c)
# ============================================================


class PhysicsBottleneck(nn.Module):
    """
    Projects the pooled latent r into an interpretable concept vector c.

    Each concept dimension is bounded by a physics-informed activation:
      - ``"tanh"``    -> [-1, 1]
      - ``"sigmoid"`` -> [ 0, 1]

    Learned sparsity gates scale each concept.
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

        self.concept_proj = nn.Linear(latent_dim, n_concepts)

        init_logits = torch.empty(n_concepts)
        n_sup = min(n_concepts, n_supervised)
        init_logits[:n_sup] = 2.2
        if n_concepts > n_sup:
            init_logits[n_sup:] = -2.2
        self.concept_gate_logits = nn.Parameter(init_logits)

    def _apply_per_concept_activation(self, raw: torch.Tensor) -> torch.Tensor:
        activated = torch.empty_like(raw)
        for i, act_type in enumerate(self.concept_activations):
            if act_type == "sigmoid":
                activated[:, i] = torch.sigmoid(raw[:, i])
            else:
                activated[:, i] = torch.tanh(raw[:, i])
        return activated

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        raw = self.concept_proj(r)
        c_activated = self._apply_per_concept_activation(raw)
        gates = torch.sigmoid(self.concept_gate_logits)
        return c_activated * gates


# ============================================================
# Stage 3 — Decoders
# ============================================================


class LatentFiLMDecoder(nn.Module):
    """
    FiLM-conditioned decoder: the latent r generates per-feature scale (γ)
    and shift (β) that modulate query static features.  The MLP then sees
    only [shear, modulated_static] — there is **no direct r concatenation**.

    This architecture makes it impossible for the decoder to bypass the
    latent.  All protein-specific information from context MUST flow through
    the modulation parameters.  If the encoder doesn't encode useful
    information in r, the decoder degrades to a context-free baseline.

    FiLM generator design
    ---------------------
    A 2-layer MLP (r → hidden → 2×static_dim) produces richer modulations
    than a single linear layer, allowing nonlinear feature interactions.
    γ is passed through tanh and scaled to [-2, +2] so individual features
    can be amplified up to 3× or suppressed to near-zero, but not to
    extreme magnitudes that cause training instability.

    Parameters
    ----------
    static_dim : int
    memory_dim : int  (latent_dim for main decoder)
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
        self.static_dim = static_dim

        # FiLM generator: 2-layer MLP for richer modulations
        self.film_gen = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, static_dim * 2),  # γ and β
        )

        # Prediction MLP: sees ONLY shear + modulated static — no direct r
        self.mlp = nn.Sequential(
            nn.Linear(1 + static_dim, hidden_dim),
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
        n_q = query_shear.size(1)

        # Generate per-feature scale (γ) and shift (β) from latent r
        film_params = self.film_gen(memory_vector)  # [B, 2*static_dim]
        gamma_raw, beta = film_params.chunk(2, dim=-1)  # each [B, static_dim]

        # Constrain γ to [-2, +2] — feature can be amplified up to 3×
        # or suppressed to near-zero, but not to extreme magnitudes
        gamma = 2.0 * torch.tanh(gamma_raw)

        # Expand to query dimension
        gamma = gamma.unsqueeze(1).expand(-1, n_q, -1)  # [B, Q, static_dim]
        beta = beta.unsqueeze(1).expand(-1, n_q, -1)

        # Modulate: protein-specific r controls HOW each static feature
        # is interpreted — this is the ONLY path for context information
        modulated = query_static * (1.0 + gamma) + beta

        x = torch.cat([query_shear, modulated], dim=-1)  # [B, Q, 1+static_dim]
        return self.mlp(x)


class ViscosityDecoder(nn.Module):
    """
    Predicts log-viscosity from a memory vector (latent r or concept c)
    concatenated with query shear and static features.

    Retained for the concept decoder path and the baseline CrossSampleCNP.
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
        n_q = query_shear.size(1)
        m_exp = memory_vector.unsqueeze(1).repeat(1, n_q, 1)
        x = torch.cat([query_shear, query_static, m_exp], dim=-1)
        return self.mlp(x)


# ============================================================
# Baseline CNP (no bottleneck — memory vector = r)
# ============================================================


class CrossSampleCNP(nn.Module):
    """
    Baseline Cross-Sample Conditional Neural Process.
    Memory vector is the raw pooled latent r.
    """

    def __init__(
        self,
        static_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.viscosity_decoder = ViscosityDecoder(
            static_dim=static_dim,
            memory_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.encoder = self.context_encoder.mlp

    def encode_memory(self, context_tensor: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(context_tensor)

    def decode_from_memory(
        self,
        memory_vector: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> torch.Tensor:
        return self.viscosity_decoder(memory_vector, query_shear, query_static)

    def forward(
        self,
        context_tensor: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        r = self.encode_memory(context_tensor)
        pred = self.decode_from_memory(r, query_shear, query_static)
        return pred, None


# ============================================================
# Hybrid Concept Bottleneck CNP
# ============================================================


class ConceptBottleneckCNP(nn.Module):
    """
    Hybrid CBM-CNP: full latent for prediction, concepts for analysis.

    Data flow
    ---------
    context_tensor
        -> ContextEncoder        -> r  [B, latent_dim]   (main memory)
        -> PhysicsBottleneck     -> c  [B, n_concepts]   (analysis branch)
        -> ViscosityDecoder(r)   -> prediction            (main decoder)
        -> ConceptDecoder(c)     -> prediction            (intervention decoder)

    The main decoder receives r (128-dim), giving it 8x more information
    capacity than the previous concept-only pathway (16-dim).  The concept
    branch is supervised for interpretability and provides causal
    intervention through a lightweight secondary decoder.
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
        self.latent_dim = latent_dim
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

        # MAIN decoder: LatentFiLM — r modulates static features with NO
        # direct concatenation.  The decoder CANNOT bypass the latent.
        self.viscosity_decoder = LatentFiLMDecoder(
            static_dim=static_dim,
            memory_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # CONCEPT decoder: receives concept c (16-dim) for intervention analysis
        self.concept_decoder = ViscosityDecoder(
            static_dim=static_dim,
            memory_dim=n_concepts,
            hidden_dim=hidden_dim // 2,
            dropout=dropout,
        )

        # Alias for trainer.py pre-pool consistency loss
        self.encoder = self.context_encoder.mlp

    # ----------------------------------------------------------
    # Encoding
    # ----------------------------------------------------------

    def encode_memory(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """Returns pooled latent r [B, latent_dim] — main memory for prediction."""
        return self.context_encoder(context_tensor)

    def encode_concepts(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """Returns gated concept vector c [B, n_concepts] — for analysis."""
        r = self.context_encoder(context_tensor)
        return self.bottleneck(r)

    def encode_latent(self, context_tensor: torch.Tensor) -> torch.Tensor:
        """Alias for encode_memory — used by triplet loss and diagnostics."""
        return self.context_encoder(context_tensor)

    # ----------------------------------------------------------
    # Decoding
    # ----------------------------------------------------------

    def decode_from_memory(
        self,
        latent_r: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> torch.Tensor:
        """Main prediction: decode from full latent r."""
        return self.viscosity_decoder(latent_r, query_shear, query_static)

    def decode_from_concepts(
        self,
        concept_c: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> torch.Tensor:
        """Concept-pathway prediction: decode from concept vector c."""
        return self.concept_decoder(concept_c, query_shear, query_static)

    # ----------------------------------------------------------
    # Forward (training)
    # ----------------------------------------------------------

    def forward(
        self,
        context_tensor: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass.

        Returns
        -------
        pred : Tensor [B, Q, 1]       — main prediction from latent r
        c    : Tensor [B, n_concepts]  — concept vector for supervision
        """
        r = self.context_encoder(context_tensor)
        c = self.bottleneck(r)
        pred = self.viscosity_decoder(r, query_shear, query_static)
        return pred, c

    # ----------------------------------------------------------
    # Causal intervention
    # ----------------------------------------------------------

    def intervene(
        self,
        context_tensor: torch.Tensor,
        query_shear: torch.Tensor,
        query_static: torch.Tensor,
        concept_idx: int | list[int],
        concept_value: float,
    ) -> torch.Tensor:
        """do(c_i = v): clamp concept and decode via concept decoder."""
        c = self.encode_concepts(context_tensor)
        c_mod = c.clone()
        if isinstance(concept_idx, int):
            concept_idx = [concept_idx]
        for idx in concept_idx:
            c_mod[:, idx] = concept_value
        return self.concept_decoder(c_mod, query_shear, query_static)

    # ----------------------------------------------------------
    # CBM accessors
    # ----------------------------------------------------------

    @property
    def concept_gate_logits(self) -> nn.Parameter:
        return self.bottleneck.concept_gate_logits

    @property
    def _concept_activations(self) -> list[str]:
        return self.bottleneck.concept_activations

    def concept_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.bottleneck.concept_gate_logits).detach()

    def decov_loss(self, concepts: torch.Tensor) -> torch.Tensor:
        if concepts.size(0) < 2:
            return torch.tensor(0.0, device=concepts.device)
        c_centered = concepts - concepts.mean(dim=0, keepdim=True)
        cov = (c_centered.T @ c_centered) / (concepts.size(0) - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag**2).sum() / (self.n_concepts * (self.n_concepts - 1))


# ============================================================
# Model-agnostic helpers
# ============================================================


def _forward(
    model: CrossSampleCNP | ConceptBottleneckCNP,
    ctx: torch.Tensor,
    qx: torch.Tensor,
    qstat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return model(ctx, qx, qstat)


def _encode_latent(
    model: CrossSampleCNP | ConceptBottleneckCNP,
    context_tensor: torch.Tensor,
) -> torch.Tensor:
    if isinstance(model, ConceptBottleneckCNP):
        return model.encode_latent(context_tensor)
    return model.encode_memory(context_tensor)
