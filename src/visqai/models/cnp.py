"""
cnp.py
======
The Cross-Sample Conditional Neural Process architecture (AttentionPool +
CrossSampleCNP). Previously defined independently in both
ml/cnp_mk2/inference_o_net.py and ml/cnp_mk2/train_o_net_v4_rung1.py — the two
copies were verified byte-identical (diffed, only comments differed) before
being merged here. Both the trainer and the predictor import this single
definition now, so the two copies can no longer drift out of sync.

DECODER SPLIT: prediction = feature_prior(query) + g(query, r)
---------------------------------------------------------------
Previously the decoder was a single MLP over [query_shear, query_static, r]:
r=0 (the zero-shot / uninformative-context case -- see predictor.py's
`memory_vector is None -> torch.zeros`) was just another input value the
network happened to learn *some* mapping for, with no guarantee that mapping
was as good as a context-free prediction. That is what let a bad/OOD context
actively make a good zero-shot prediction worse (e.g. belatacept
0.077->0.125 log MAE under 1-shot -- see logo_scoreboard.csv): nothing in the
architecture stopped the r-dependent path from doing that.

The decoder is now two heads:
  - prior_head(query_shear, query_static): the feature-only prior. It never
    sees r, so a bad context literally cannot reach it.
  - correction_head(query_shear, query_static, r) = g: the only path that
    sees the pooled context. Its FINAL layer is zero-initialized, so at the
    start of training (and for any r the correction head hasn't yet learned a
    reason to react to) g contributes exactly zero and
    prediction = prior_head(query) + 0 = prior_head(query).
Training (visqai.training.loop.train_epoch) reinforces this split by fitting
prior_head directly to y and g to the RESIDUAL y - prior_head(query).detach()
-- see that module's docstring. The hard backstop lives one layer up, in
visqai.eval.cnp_logo's context gate: the LOGO harness now asserts few-shot
never scores worse than zero-shot per held-out group.

torch.save's state_dict is keyed by attribute path, not import path, so the
original AttentionPool/CrossSampleCNP merge did not break loading any
existing checkpoint -- but THIS change does: `decoder.*` is gone, replaced by
`prior_head.*` / `correction_head.*`. Any checkpoint saved before this split
needs retraining.

POST-SPLIT ZERO-SHOT REGRESSION (P0 fix)
-----------------------------------------
The split above initially made zero-shot WORSE on most held-out groups
(polyclonal +0.12, pembrolizumab +0.07, ibalizumab +0.03 log MAE vs.
pre-split -- see logo_scoreboard.csv), which is backwards: prior_head should
have been a strictly easier fit than the old decoder, since it targets y
directly instead of having to extrapolate to an unseen r=0 input. Three
candidate causes were checked, in order:
  1. Capacity: prior_head's hidden width is now hidden_dim + latent_dim (see
     __init__), matching the parameter budget the old combined decoder's
     first layer had (it took r as an extra input at the same hidden width).
  2. Gradient cross-talk: visqai.training.loop.train_epoch clips prior_head's
     gradient separately from the rest of the model's, so a large
     correction/utility/triplet gradient on a given iteration can no longer
     shrink prior_head's own update via the shared clip_grad_norm_ budget.
  (1) and (2) alone did NOT recover pre-split zero-shot numbers on a
  retrain (ibalizumab got WORSE; pembrolizumab and polyclonal barely moved)
  -- ruling out capacity and gradient contention as the cause.
  3. The actual cause: visqai.training.run.train_final_model's early-
     stopping/checkpoint selection used visqai.training.loop.validate()
     alone, which ALWAYS builds a non-empty context split -- it never
     exercises prior_head's r=0 path at all. Once the split made prior_head
     the sole zero-shot path, checkpoint selection was blind to the exact
     metric that governs zero-shot quality: nothing stopped it from picking
     a snapshot that scores well with context and poorly without it.
     visqai.training.loop.validate_zero_shot fixes this by scoring the
     literal r=0 path (matching predictor.py's deployment behavior exactly),
     mixed 50/50 with validate()'s context-informed loss for both the LR
     scheduler and the early-stopping/best-checkpoint decision.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, latent_dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        # LayerNorm constrains the magnitude of r so the contrastive objective
        # uses direction rather than explosive norm scaling.
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


class CrossSampleCNP(nn.Module):
    def __init__(self, static_dim, hidden_dim=128, latent_dim=128, dropout=0.0):
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

        # Feature-only prior: consumes [query_shear, query_static] alone,
        # with no access to r. This is the zero-shot predictor -- a bad
        # context can never reach it because it is architecturally invisible
        # to this head.
        #
        # Widened to hidden_dim + latent_dim: the old combined decoder's
        # first Linear layer took (1 + static_dim + latent_dim) inputs, so it
        # had latent_dim * hidden_dim more first-layer weights than a
        # prior_head sized at plain hidden_dim would -- capacity that used to
        # be available to the query->y mapping even in the r-independent
        # (zero-shot) regime, since the decoder's hidden width was shared
        # across both jobs. Matching that width here rules out "the split
        # starved the zero-shot path of parameters it used to have" as the
        # cause of the post-split zero-shot regression (see module docstring
        # and logo_scoreboard.csv), independent of the residual-loss fix in
        # visqai.training.loop.
        prior_hidden_dim = hidden_dim + latent_dim
        self.prior_head = nn.Sequential(
            nn.Linear(1 + static_dim, prior_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prior_hidden_dim, prior_hidden_dim),
            nn.ReLU(),
            nn.Linear(prior_hidden_dim, 1),
        )

        # Correction head g(query, r): the only path with access to the
        # pooled context. Final layer zero-initialized so g starts (and, for
        # any r it hasn't learned to react to, stays) at exactly zero --
        # prediction = prior_head(query) + g(query, r), g(query, 0) ~= 0.
        self.correction_head = nn.Sequential(
            nn.Linear(1 + static_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def _decode_split(self, query_shear, query_static, r):
        """Returns (prior, correction) separately -- training
        (visqai.training.loop.train_epoch) needs both to fit the correction
        head against the residual rather than the raw target."""
        n_queries = query_shear.size(1)
        r_expanded = r.unsqueeze(1).repeat(1, n_queries, 1)
        prior = self.prior_head(torch.cat([query_shear, query_static], dim=-1))
        correction = self.correction_head(torch.cat([query_shear, query_static, r_expanded], dim=-1))
        return prior, correction

    def forward(self, context_tensor, query_shear, query_static):
        r = self.pooler(self.encoder(context_tensor))
        prior, correction = self._decode_split(query_shear, query_static, r)
        return prior + correction

    def forward_split(self, context_tensor, query_shear, query_static):
        """Same as forward(), but returns (prior, correction) unsummed."""
        r = self.pooler(self.encoder(context_tensor))
        return self._decode_split(query_shear, query_static, r)

    def encode_memory(self, context_tensor):
        return self.pooler(self.encoder(context_tensor))

    def decode_from_memory(self, memory_vector, query_shear, query_static):
        prior, correction = self._decode_split(query_shear, query_static, memory_vector)
        return prior + correction

    def decode_from_memory_split(self, memory_vector, query_shear, query_static):
        return self._decode_split(query_shear, query_static, memory_vector)
