import torch
import torch.nn as nn

# ==========================================
# [CBM-2] Concept definitions
# ==========================================
# Each entry: (concept_name, proxy_column, sign, activation)
#   proxy_column: column in df AFTER feature engineering but BEFORE scaling
#   sign: +1 if high proxy value → concept activation should be positive
#         -1 if high proxy value → concept activation should be negative
#   activation: "tanh" → [-1, 1] for concepts that can be pos or neg
#               "sigmoid" → [0, 1] for inherently non-negative concepts
#
# [v3] Changes from v2:
#   - Per-concept activation functions (tanh vs sigmoid) based on physics.
#   - Replaced three r>0.99 pass-through proxies (kP, HCI, C_Class) with
#     interaction-term proxies so the concept learns something the raw
#     feature in query_static doesn't already capture.
#   - Added learned concept gates + DeCov decorrelation loss.
#   - lambda_concept_sup is now annealed (high early, low late).
#
# Physical interpretation of each concept:
#   self_interaction    — concentration-dependent attractive self-interaction
#   hydrophobicity      — concentration-enhanced hydrophobic patch association
#   charge_environment  — charge–ionic strength interplay driving repulsion
#   ionic_screening     — Debye screening of protein-protein charge repulsion
#   crowding            — macromolecular crowding / volume exclusion effects
#   nonlinear_conc      — non-linear concentration effect (approach to jamming)
#   cosolute_interaction— concentration × stabilizer cross-term
#   cosolute_protection — cosolute-mediated steric/entropic viscosity reduction
#
# These are SOFT priors. lambda_concept_sup is annealed from ~0.3 → ~0.01
# so the viscosity MSE signal can override the proxy when the data warrants it.
CONCEPT_DEFS = [
    ("self_interaction", "conc_x_kP", -1, "tanh"),
    ("hydrophobicity", "conc_x_HCI", +1, "sigmoid"),
    ("charge_environment", "charge_x_ionic", +1, "tanh"),
    ("ionic_screening", "Ionic_Strength_Proxy", +1, "sigmoid"),
    ("crowding", "Protein_conc", +1, "sigmoid"),
    ("nonlinear_conc", "conc_sq", +1, "sigmoid"),
    ("cosolute_interaction", "Stabilizer_mg_mL", -1, "tanh"),
    ("cosolute_protection", "Crowding_Index", -1, "sigmoid"),
]

N_CONCEPTS_SUPERVISED = len(CONCEPT_DEFS)
CONCEPT_NAMES = [cd[0] for cd in CONCEPT_DEFS]
CONCEPT_ACTIVATIONS = [cd[3] for cd in CONCEPT_DEFS]


class AttentionPool(nn.Module):
    """Compresses a variable number of context samples into a single fixed vector."""

    def __init__(self, latent_dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class ContextEncoder(nn.Module):
    """Encodes raw formulation data and context points into the memory vector (r)."""

    def __init__(self, static_dim, hidden_dim=128, latent_dim=128, dropout=0.0):
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

    def forward(self, context_tensor):
        # Maps [B, N_c, Features] -> [B, latent_dim]
        encoded = self.mlp(context_tensor)
        r = self.pooler(encoded)
        return r


class PhysicsBottleneck(nn.Module):
    """Maps the memory vector to explicit physical concepts and applies sparsity gates."""

    def __init__(
        self, latent_dim, n_concepts, concept_activations, n_supervised=N_CONCEPTS_SUPERVISED
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.concept_activations = concept_activations

        # Linear projection from latent space to concept space
        self.concept_proj = nn.Linear(latent_dim, n_concepts)

        # Initialize gates: supervised open (2.2), latent closed (-2.2)
        init_logits = torch.empty(n_concepts)
        n_sup = min(n_concepts, n_supervised)
        init_logits[:n_sup] = 2.2
        if n_concepts > n_sup:
            init_logits[n_sup:] = -2.2

        self.concept_gate_logits = nn.Parameter(init_logits)

    def _apply_per_concept_activation(self, raw_logits):
        activated = torch.empty_like(raw_logits)
        for i, act_type in enumerate(self.concept_activations):
            if act_type == "sigmoid":
                activated[:, i] = torch.sigmoid(raw_logits[:, i])
            else:
                activated[:, i] = torch.tanh(raw_logits[:, i])
        return activated

    def forward(self, r):
        # 1. Project to raw concept scores
        raw = self.concept_proj(r)

        # 2. Bound values based on physics (tanh/sigmoid)
        c_activated = self._apply_per_concept_activation(raw)

        # 3. Apply learned L1 sparsity gates
        gates = torch.sigmoid(self.concept_gate_logits)
        c_final = c_activated * gates

        return c_final


class ViscosityDecoder(nn.Module):
    """Expands the concept vector and predicts viscosity across query shear rates."""

    def __init__(self, static_dim, n_concepts, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + static_dim + n_concepts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, concept_vector, query_shear, query_static):
        # Expand concept vector to match number of query points (N_q)
        n_q = query_shear.size(1)
        c_exp = concept_vector.unsqueeze(1).repeat(1, n_q, 1)

        # Concatenate: [Shear | Static Features | Concept State]
        decoder_input = torch.cat([query_shear, query_static, c_exp], dim=-1)

        # Predict Log-Viscosity
        return self.mlp(decoder_input)


# ==========================================
# 3. The Top-Level Model
# ==========================================


class ConceptBottleneckCNP(nn.Module):
    """
    Main Neural Process model orchestrating the Encoder, Bottleneck, and Decoder.
    Maintains the encode_memory and decode_from_memory API for seamless inference.
    """

    def __init__(
        self,
        static_dim,
        hidden_dim=128,
        latent_dim=128,
        n_concepts=N_CONCEPTS_SUPERVISED,
        concept_names=None,
        concept_activations=None,
        dropout=0.0,
    ):
        super().__init__()
        self.static_dim = static_dim
        self.n_concepts = n_concepts

        # Setup Names
        self.concept_names = (
            concept_names
            if concept_names is not None
            else CONCEPT_NAMES[:n_concepts]
            + [f"latent_{i}" for i in range(max(0, n_concepts - N_CONCEPTS_SUPERVISED))]
        )

        # Setup Activations
        if concept_activations is not None:
            c_acts = concept_activations
        else:
            c_acts = CONCEPT_ACTIVATIONS[: min(n_concepts, N_CONCEPTS_SUPERVISED)] + ["tanh"] * max(
                0, n_concepts - N_CONCEPTS_SUPERVISED
            )

        # Instantiate Sub-Modules
        self.encoder = ContextEncoder(
            static_dim=static_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout
        )

        self.bottleneck = PhysicsBottleneck(
            latent_dim=latent_dim,
            n_concepts=n_concepts,
            concept_activations=c_acts,
            n_supervised=N_CONCEPTS_SUPERVISED,
        )

        self.decoder = ViscosityDecoder(
            static_dim=static_dim, n_concepts=n_concepts, hidden_dim=hidden_dim, dropout=dropout
        )

    # --- Standard CNP API Methods --- #

    def encode_memory(self, context_tensor):
        """Standard API: Extracts the final concept vector from context."""
        r = self.encoder(context_tensor)
        c = self.bottleneck(r)
        return c

    def decode_from_memory(self, concept_vector, query_shear, query_static):
        """Standard API: Predicts viscosity from the stored concept vector."""
        return self.decoder(concept_vector, query_shear, query_static)

    def forward(self, context_tensor, query_shear, query_static):
        """End-to-end forward pass for training."""
        c = self.encode_memory(context_tensor)
        return self.decode_from_memory(c, query_shear, query_static)

    # --- CBM Causal Intervention API --- #
    def intervene(self, context_tensor, query_shear, query_static, concept_idx, concept_value):
        """
        [CBM-5] Causal intervention: clamp concept_idx to concept_value, re-decode.
        This is do(c_i = v) in the Pearl causal sense — not merely correlation.

        Args:
            concept_idx:   Index into concept vector (int or list of ints).
            concept_value: Scalar value to clamp to (float, in activation range).

        Returns:
            Predictions under the intervention [B, n_queries, 1].
        """
        # 1. Get the natural physical state
        c = self.encode_memory(context_tensor)
        c_mod = c.clone()

        # 2. Apply the manual intervention
        if isinstance(concept_idx, int):
            concept_idx = [concept_idx]
        for idx in concept_idx:
            c_mod[:, idx] = concept_value

        # 3. Predict the new formulation behavior
        return self.decode_from_memory(c_mod, query_shear, query_static)

    # --- CBM Specific Accessors --- #
    @property
    def concept_gate_logits(self):
        """Expose the gate logits for the L1 sparsity loss during training."""
        return self.bottleneck.concept_gate_logits
