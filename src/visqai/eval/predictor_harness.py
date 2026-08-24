"""
predictor_harness.py
=====================
Test/replay-only helpers that reach into ViscosityPredictorCNP internals
(memory_vector/context_t/model weights). Moved from learning_curve_ibal.py.
Deliberately kept out of visqai.inference.predictor's public API -- these
don't belong on the production predictor, only on eval/replay harnesses.
"""

from __future__ import annotations

import pandas as pd
import torch


def save_state(predictor) -> dict:
    """Snapshot the latent memory state (weights are immutable) -- both the
    legacy neural memory_vector and the delta corrector's offset_hat/
    conc_hat/slope_hat (T-R3.1/T-R3.4/Task 1.1), since predict() now depends
    on the latter, not the former."""
    return {
        "memory_vector": predictor.memory_vector.clone() if predictor.memory_vector is not None else None,
        "context_t": predictor.context_t.clone() if predictor.context_t is not None else None,
        "offset_hat": predictor.offset_hat,
        "conc_hat": predictor.conc_hat,
        "conc_center": predictor.conc_center,
        "slope_hat": predictor.slope_hat,
        "slope_center": predictor.slope_center,
        "n_context_points": predictor.n_context_points,
    }


def restore_state(predictor, snap: dict):
    """Restore latent memory state from snapshot."""
    predictor.memory_vector = snap["memory_vector"].clone() if snap["memory_vector"] is not None else None
    predictor.context_t = snap["context_t"].clone() if snap["context_t"] is not None else None
    # Must come AFTER the memory_vector assignment above -- that setter
    # zeroes offset_hat/conc_hat/slope_hat whenever the restored
    # memory_vector is None, which would otherwise clobber real restored
    # values right back to 0.
    predictor.offset_hat = snap.get("offset_hat", 0.0)
    predictor.conc_hat = snap.get("conc_hat", 0.0)
    predictor.conc_center = snap.get("conc_center", 0.0)
    predictor.slope_hat = snap.get("slope_hat", 0.0)
    predictor.slope_center = snap.get("slope_center", 0.0)
    predictor.n_context_points = snap.get("n_context_points", 0)


def reset_memory(predictor):
    """Clear latent state without touching model weights."""
    predictor.memory_vector = None
    predictor.context_t = None


def has_nan_weights(predictor) -> bool:
    return any(torch.isnan(p).any() for p in predictor.model.parameters())


def predict_from_built(predictor, df_built: pd.DataFrame, orig_df: pd.DataFrame) -> pd.DataFrame:
    """Like predictor.predict(orig_df), but skips predict()'s own internal
    build_feature_frame call -- `df_built` must already be the output of
    visqai.preprocessing.pipeline.build_feature_frame (as produced by
    predictor._preprocess/_preprocess_built). This is what lets a caller do
    an ENGINEERED-feature permutation-importance pass: mutate an engineered
    column (e.g. conc_sq, whole_charge) directly, then predict from the
    already-built frame -- calling predict() on that mutated frame would just
    re-derive the column from its raw inputs via another build_feature_frame
    pass and silently overwrite the mutation before the model ever saw it.

    `orig_df` is the RAW (unmutated) df that produced `df_built` -- it
    supplies Protein_conc/row-count for the delta-corrector terms, same role
    `df` plays in predict()'s own signature. Uses predictor._preprocess_built/
    _predict_from_tensors so this shares predict()'s exact corrector logic
    rather than reimplementing it."""
    q_static, q_shear, _ = predictor._preprocess_built(df_built)
    return predictor._predict_from_tensors(q_static, q_shear, orig_df)
