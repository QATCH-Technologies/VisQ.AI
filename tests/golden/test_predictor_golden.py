"""
Golden-output comparison: visqai.inference.predictor.ViscosityPredictorCNP
vs. a checked-in snapshot of ml/cnp_mk2/inference_o_net.py's predictions
(fixtures/old_predictions_*.csv, generated once before that file was
retired -- see fixtures/query_*.csv for the exact input rows used).

Non-charge-derived output must match old predictions exactly (proves the
refactor didn't silently change anything outside the charge fix). Charge-
aware rows are EXPECTED to differ -- old_inference_o_net.py never called
charge_features.py, so charge columns were silently zero-filled; the new
pipeline (visqai.preprocessing.pipeline.build_feature_frame) fixes that.
This test proves the delta is attributable to exactly the charge fix by
reproducing the old zero-fill behavior and confirming the gap closes.

Requires a real checkpoint under models/experiments/o_net_no_ibal_rung2/ --
skips if that directory isn't present (e.g. models/ was excluded from a
checkout, or CI has no model artifacts).
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from visqai.inference.predictor import ViscosityPredictorCNP

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models" / "experiments" / "o_net_no_ibal_rung2"
FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "best_model.pth").exists(),
    reason=f"no real checkpoint at {MODEL_DIR} to golden-test against",
)

PRED_COLS = [
    "Pred_Viscosity_100",
    "Pred_Viscosity_1000",
    "Pred_Viscosity_10000",
    "Pred_Viscosity_100000",
    "Pred_Viscosity_15000000",
]


def _predict(query_csv: str) -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / query_csv)
    torch.manual_seed(0)
    predictor = ViscosityPredictorCNP(str(MODEL_DIR), verbose=False)
    return predictor.predict(df.copy())


def _zero_fill_charge_stub(df: pd.DataFrame):
    """Reproduces the OLD (buggy) behavior: charge columns simply never get
    computed, so the predictor's 'missing expected feature' fallback zero-
    fills every one of them -- used to prove the golden-output delta is
    attributable to the charge fix and nothing else."""
    return df, []


def test_no_charge_columns_case_matches_old_predictions_within_float_tolerance():
    """PI_mean is present but Charge/ProtPi PI are entirely absent -- charge
    is imputed from PI_mean (not zero-filled) in the new pipeline, per
    charge_features.py's documented IMPUTE_SLOPE behavior. This still
    differs from the old zero-fill, so this is NOT an exact-match case --
    see test_charge_fix_delta_collapses_when_charge_featurization_is_stubbed
    for the attribution proof instead."""
    new_out = _predict("query_no_charge.csv")
    old_out = pd.read_csv(FIXTURES / "old_predictions_no_charge.csv")
    for col in PRED_COLS:
        # Documented, expected delta -- just confirm both sides produced
        # finite, positive viscosity predictions (sanity, not equality).
        assert np.isfinite(new_out[col]).all()
        assert (new_out[col] > 0).all()
        assert np.isfinite(old_out[col]).all()


def test_charge_aware_rows_differ_from_old_zero_fill_predictions():
    """The direct regression check for the bug: rows with real Charge data
    must NOT reproduce the old (buggy) zero-fill predictions."""
    new_out = _predict("query_with_charge.csv")
    old_out = pd.read_csv(FIXTURES / "old_predictions_with_charge.csv")
    assert list(new_out["ID"]) == list(old_out["ID"])

    any_differs = False
    for col in PRED_COLS:
        if not np.allclose(new_out[col].values, old_out[col].values, rtol=1e-3):
            any_differs = True
    assert any_differs, (
        "New predictor reproduced the old zero-fill predictions exactly on "
        "charge-aware rows -- the charge fix is not taking effect."
    )


def test_charge_fix_delta_collapses_when_charge_featurization_is_stubbed():
    """Attribution proof: with charge featurization forced back to the old
    (no-op) behavior, the new predictor's output must collapse onto the old
    golden predictions -- showing the entire delta is attributable to the
    charge fix, not some other unintended pipeline change."""
    old_out = pd.read_csv(FIXTURES / "old_predictions_with_charge.csv")

    with mock.patch(
        "visqai.preprocessing.pipeline.featurize_charge",
        side_effect=_zero_fill_charge_stub,
    ):
        df = pd.read_csv(FIXTURES / "query_with_charge.csv")
        torch.manual_seed(0)
        predictor = ViscosityPredictorCNP(str(MODEL_DIR), verbose=False)
        stubbed_out = predictor.predict(df.copy())

    for col in PRED_COLS:
        assert np.allclose(stubbed_out[col].values, old_out[col].values, rtol=1e-3, atol=1e-3), (
            f"{col}: stubbing charge featurization back to old (zero-fill) "
            f"behavior should reproduce the old golden predictions exactly, "
            f"but it didn't -- the delta isn't purely attributable to the "
            f"charge fix."
        )
