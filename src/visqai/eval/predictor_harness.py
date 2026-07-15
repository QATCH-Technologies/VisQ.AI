"""
predictor_harness.py
=====================
Test/replay-only helpers that reach into ViscosityPredictorCNP internals
(memory_vector/context_t/model weights). Moved from learning_curve_ibal.py.
Deliberately kept out of visqai.inference.predictor's public API -- these
don't belong on the production predictor, only on eval/replay harnesses.
"""

from __future__ import annotations

import torch


def save_state(predictor) -> dict:
    """Snapshot only the latent memory state (weights are immutable)."""
    return {
        "memory_vector": predictor.memory_vector.clone() if predictor.memory_vector is not None else None,
        "context_t": predictor.context_t.clone() if predictor.context_t is not None else None,
    }


def restore_state(predictor, snap: dict):
    """Restore latent memory state from snapshot."""
    predictor.memory_vector = snap["memory_vector"].clone() if snap["memory_vector"] is not None else None
    predictor.context_t = snap["context_t"].clone() if snap["context_t"] is not None else None


def reset_memory(predictor):
    """Clear latent state without touching model weights."""
    predictor.memory_vector = None
    predictor.context_t = None


def has_nan_weights(predictor) -> bool:
    return any(torch.isnan(p).any() for p in predictor.model.parameters())
