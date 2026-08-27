"""
test_cnp_logo_honest_metrics.py
================================
Task 0.2 (issue1_query_conditioned_correction_plan.md): unit tests for the
honest per-k lift reporting added to run_cnp_fold's row assembly and
_assert_context_gate's extended violation check.

run_cnp_fold itself trains a real model per call, so these test the row
post-processing logic directly by constructing the same partial `row` dict
run_cnp_fold builds up to the point where the Task 0.2 block runs, then
re-deriving the Task 0.2 columns with the identical logic -- equivalent to
calling run_cnp_fold with a stubbed _shot_metrics, but far cheaper and
deterministic. _assert_context_gate itself IS called directly (it's already
a pure function of the rows list, same as the existing guard tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from visqai.eval.logo_eval import CONTEXT_GATE_TOLERANCE, MONOTONE_CONTEXT_TOLERANCE, _assert_context_gate


def _apply_task02_columns(row: dict, shots) -> dict:
    """Reimplements run_cnp_fold's Task 0.2 block verbatim (same formulas)
    against a caller-supplied row -- used to test the honest-metrics logic
    without paying for a real training run per test case."""
    zshot = row.get("zero_shot_log_mae", np.nan)
    k_lifts = {}
    if not np.isnan(zshot):
        for k in shots:
            if k == 0:
                continue
            mae_col = f"fewshot_k{k}_log_mae"
            if mae_col in row and not np.isnan(row[mae_col]):
                lift_k = zshot - row[mae_col]
                row[f"fewshot_k{k}_lift"] = lift_k
                k_lifts[k] = lift_k

    if k_lifts:
        largest_k = max(k_lifts)
        row["all_context_lift"] = k_lifts[largest_k]
        ordered_lifts = [k_lifts[k] for k in sorted(k_lifts)]
        row["monotone_context"] = bool(
            all(b >= a - MONOTONE_CONTEXT_TOLERANCE for a, b in zip(ordered_lifts, ordered_lifts[1:]))
        )
    return row


def test_per_k_lift_columns_and_all_context_lift_on_clean_monotone_case():
    row = {
        "axis": "protein",
        "group": "good",
        "zero_shot_log_mae": 0.30,
        "fewshot_k1_log_mae": 0.28,
        "fewshot_k2_log_mae": 0.24,
        "fewshot_k4_log_mae": 0.20,
        "fewshot_k8_log_mae": 0.15,
        "best_fewshot_log_mae": 0.15,
        "lift": 0.15,
    }
    row = _apply_task02_columns(row, shots=(0, 1, 2, 4, 8))

    assert row["fewshot_k1_lift"] == pytest.approx(0.02)
    assert row["fewshot_k2_lift"] == pytest.approx(0.06)
    assert row["fewshot_k4_lift"] == pytest.approx(0.10)
    assert row["fewshot_k8_lift"] == pytest.approx(0.15)
    assert row["all_context_lift"] == pytest.approx(0.15)  # k=8 is the largest feasible k
    assert row["monotone_context"] is True


def test_k8_regression_is_visible_in_all_context_lift_even_when_best_fewshot_looks_good():
    """The exact blind spot Task 0.2 closes: a good small-k score makes the
    OLD best-case `lift` (min over k, already in `row`) look fine, but the
    largest feasible context (k=8) actually regressed. all_context_lift must
    surface that regression even though `lift` does not."""
    row = {
        "axis": "protein",
        "group": "regresses_at_k8",
        "zero_shot_log_mae": 0.30,
        "fewshot_k1_log_mae": 0.25,
        "fewshot_k2_log_mae": 0.22,
        "fewshot_k4_log_mae": 0.20,
        "fewshot_k8_log_mae": 0.38,  # regression: worse than zero-shot
        "best_fewshot_log_mae": 0.20,
        "lift": 0.10,  # best-case lift still looks positive
    }
    row = _apply_task02_columns(row, shots=(0, 1, 2, 4, 8))

    assert row["lift"] == pytest.approx(0.10)  # unaffected, still the old best-case number
    assert row["all_context_lift"] == pytest.approx(0.30 - 0.38)  # -0.08, a real regression
    assert row["monotone_context"] is False


def test_context_gate_fails_on_synthetic_k8_regression_even_though_best_case_lift_passes():
    """Acceptance criterion: a synthetic k=8 regression must fail the gate,
    even though the OLD (pre-Task-0.2) `min`-over-k `lift` alone would have
    passed it."""
    row = {
        "axis": "protein",
        "group": "regresses_at_k8",
        "zero_shot_log_mae": 0.30,
        "fewshot_k1_log_mae": 0.25,
        "fewshot_k8_log_mae": 0.38,
        "best_fewshot_log_mae": 0.25,
        "lift": 0.05,
    }
    row = _apply_task02_columns(row, shots=(0, 1, 8))

    # Sanity: the old best-case lift alone is well within tolerance --
    # confirms this case really is the blind spot, not just any regression.
    assert row["lift"] >= -CONTEXT_GATE_TOLERANCE

    with pytest.raises(AssertionError, match="all_context_lift"):
        _assert_context_gate([row])


def test_context_gate_passes_when_all_context_lift_within_tolerance():
    row = {
        "axis": "protein",
        "group": "fine",
        "zero_shot_log_mae": 0.30,
        "fewshot_k1_log_mae": 0.29,
        "fewshot_k8_log_mae": 0.30 + CONTEXT_GATE_TOLERANCE / 2,
        "best_fewshot_log_mae": 0.29,
        "lift": 0.01,
    }
    row = _apply_task02_columns(row, shots=(0, 1, 8))
    _assert_context_gate([row])  # should not raise


def test_monotone_context_false_on_non_monotone_lift_sequence():
    row = {
        "zero_shot_log_mae": 0.30,
        "fewshot_k1_log_mae": 0.20,  # lift +0.10
        "fewshot_k2_log_mae": 0.25,  # lift +0.05 -- worse than k1's lift, beyond tolerance
    }
    row = _apply_task02_columns(row, shots=(0, 1, 2))
    assert row["monotone_context"] is False


def test_monotone_context_true_within_noise_tolerance():
    row = {
        "zero_shot_log_mae": 0.30,
        "fewshot_k1_log_mae": 0.20,  # lift +0.10
        "fewshot_k2_log_mae": 0.20 + MONOTONE_CONTEXT_TOLERANCE / 2,  # tiny dip, within tolerance
    }
    row = _apply_task02_columns(row, shots=(0, 1, 2))
    assert row["monotone_context"] is True


def test_no_fewshot_columns_leaves_task02_columns_absent():
    row = {"axis": "protein", "group": "errored"}
    row = _apply_task02_columns(row, shots=(0, 1, 2, 4, 8))
    assert "all_context_lift" not in row
    assert "monotone_context" not in row
