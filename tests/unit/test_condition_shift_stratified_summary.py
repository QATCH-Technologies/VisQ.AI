"""
test_condition_shift_stratified_summary.py
============================================
Task A.1 (addendum to issue1_query_conditioned_correction_plan.md): unit
tests for stratified_summary / leave_one_protein_out_sensitivity /
validated_directions -- the re-cut reporting that surfaced the
broken-prior-rescue effect (a single outlier protein and a non-extrapolating
direction were propping up the concentration axis's headline mean delta).
"""

from __future__ import annotations

import pandas as pd
import pytest

from visqai.eval.condition_shift import (
    GOOD_PRIOR_THRESHOLD,
    axis_rollup,
    leave_one_protein_out_sensitivity,
    minimum_detectable_effect,
    stratified_summary,
    validated_directions,
)


def _board():
    return pd.DataFrame(
        [
            # good-prior, fired, small positive delta
            {"protein": "p1", "axis": "concentration", "direction": "conc_low_ctx_high_target",
             "prior_only_log_mae": 0.10, "delta": 0.01},
            # good-prior, abstained (delta==0)
            {"protein": "p2", "axis": "concentration", "direction": "conc_low_ctx_high_target",
             "prior_only_log_mae": 0.12, "delta": 0.0},
            # bad-prior, fired, huge positive delta (the "vudalimab" role)
            {"protein": "p3", "axis": "concentration", "direction": "conc_low_ctx_high_target",
             "prior_only_log_mae": 0.50, "delta": 0.33},
            # bad-prior, fired, negative delta
            {"protein": "p4", "axis": "concentration", "direction": "conc_high_ctx_low_target",
             "prior_only_log_mae": 0.20, "delta": -0.06},
        ]
    )


# ---------------------------------------------------------------------------
# stratified_summary
# ---------------------------------------------------------------------------


def test_prior_band_threshold_splits_correctly():
    summary = stratified_summary(_board())
    conc_low = summary[
        (summary.direction == "conc_low_ctx_high_target") & (summary.stratum == "good_prior")
    ]
    assert conc_low["n"].iloc[0] == 2  # p1 (0.10) and p2 (0.12), both < GOOD_PRIOR_THRESHOLD
    bad = summary[(summary.direction == "conc_low_ctx_high_target") & (summary.stratum == "bad_prior")]
    assert bad["n"].iloc[0] == 1  # p3 (0.50)
    assert GOOD_PRIOR_THRESHOLD == 0.15  # documents the calibrated threshold this test relies on


def test_fired_only_excludes_abstentions():
    summary = stratified_summary(_board())
    fired = summary[(summary.direction == "conc_low_ctx_high_target") & (summary.stratum == "fired_only")]
    assert fired["n"].iloc[0] == 2  # p1 and p3 fired; p2 abstained (delta==0)


def test_median_differs_from_mean_when_outlier_present():
    summary = stratified_summary(_board())
    all_row = summary[(summary.direction == "conc_low_ctx_high_target") & (summary.stratum == "all")].iloc[0]
    # mean is pulled toward p3's 0.33; median (of [0.01, 0.0, 0.33]) is 0.01
    assert all_row["median_delta"] < all_row["mean_delta"]


def test_win_rate_and_non_regress_rate_differ_on_abstention():
    summary = stratified_summary(_board())
    all_row = summary[(summary.direction == "conc_low_ctx_high_target") & (summary.stratum == "all")].iloc[0]
    # win_rate excludes the delta==0 abstention (p2); non_regress_rate counts it as a pass
    assert all_row["win_rate"] == pytest.approx(2 / 3)
    assert all_row["non_regress_rate"] == pytest.approx(3 / 3)


def test_direction_validated_flag_reflects_supplied_set():
    summary = stratified_summary(_board(), validated={"conc_low_ctx_high_target"})
    low = summary[summary.direction == "conc_low_ctx_high_target"]
    high = summary[summary.direction == "conc_high_ctx_low_target"]
    assert (low["direction_validated"] == True).all()  # noqa: E712
    assert (high["direction_validated"] == False).all()  # noqa: E712


def test_direction_validated_is_none_when_not_supplied():
    summary = stratified_summary(_board())
    assert summary["direction_validated"].isna().all()


# ---------------------------------------------------------------------------
# leave_one_protein_out_sensitivity
# ---------------------------------------------------------------------------


def test_loo_sensitivity_flags_the_load_bearing_protein():
    board = _board()
    loo = leave_one_protein_out_sensitivity(board, "concentration", direction="conc_low_ctx_high_target")
    # p3 (delta=0.33) is the outlier -- excluding it must be the biggest single move.
    top = loo.iloc[0]
    assert top["excluded_protein"] == "p3"
    # mean of remaining [0.01, 0.0] = 0.005, clearly lower than the full mean (0.34/3=0.1133...)
    assert top["mean_delta_excluding"] < top["full_mean_delta"]


def test_loo_sensitivity_direction_filter_scopes_correctly():
    board = _board()
    loo = leave_one_protein_out_sensitivity(board, "concentration", direction="conc_high_ctx_low_target")
    assert set(loo["excluded_protein"]) == {"p4"}


def test_loo_sensitivity_without_direction_pools_whole_axis():
    board = _board()
    loo = leave_one_protein_out_sensitivity(board, "concentration")
    assert set(loo["excluded_protein"]) == {"p1", "p2", "p3", "p4"}


# ---------------------------------------------------------------------------
# validated_directions
# ---------------------------------------------------------------------------


def test_validated_directions_excludes_direction_shift_check_fails():
    board = pd.DataFrame(
        [
            {"protein": "p1", "axis": "concentration", "direction": "conc_low_ctx_high_target",
             "prior_only_log_mae": 0.30},
            {"protein": "p1", "axis": "concentration", "direction": "conc_high_ctx_low_target",
             "prior_only_log_mae": 0.05},
        ]
    )
    random_board = pd.DataFrame([{"group": "p1", "zero_shot_log_mae": 0.15}])
    vd = validated_directions(board, random_board)
    assert "conc_low_ctx_high_target" in vd  # 0.30 > 0.15 baseline -> real extrapolation
    assert "conc_high_ctx_low_target" not in vd  # 0.05 < 0.15 baseline -> easier, not extrapolation


def test_validated_directions_includes_non_concentration_axes_unconditionally():
    board = pd.DataFrame(
        [
            {"protein": "p1", "axis": "concentration", "direction": "conc_low_ctx_high_target",
             "prior_only_log_mae": 0.05},
            {"protein": "p1", "axis": "buffer", "direction": "buffer_a_ctx_b_target",
             "prior_only_log_mae": 0.05},
        ]
    )
    random_board = pd.DataFrame([{"group": "p1", "zero_shot_log_mae": 0.15}])
    vd = validated_directions(board, random_board)
    # concentration direction fails its own validity check (0.05 < 0.15)...
    assert "conc_low_ctx_high_target" not in vd
    # ...but buffer has no analogous check, so it's validated by construction.
    assert "buffer_a_ctx_b_target" in vd


# ---------------------------------------------------------------------------
# axis_rollup (fix for the reported A.1 gap: direction_validated was NaN
# whenever a run skipped the validity check, and conc_high_ctx_low_target-
# style invalidated directions were never actually excluded from any
# aggregate -- axis_rollup requires validated and filters on it directly.)
# ---------------------------------------------------------------------------


def test_axis_rollup_raises_without_validated():
    summary = stratified_summary(_board())
    with pytest.raises(ValueError, match="requires `validated`"):
        axis_rollup(summary, validated=None)


def test_axis_rollup_excludes_invalidated_direction():
    summary = stratified_summary(_board(), validated={"conc_low_ctx_high_target"})
    rollup = axis_rollup(summary, validated={"conc_low_ctx_high_target"})
    conc_all = rollup[(rollup.axis == "concentration") & (rollup.stratum == "all")]
    # Only conc_low_ctx_high_target's 3 rows (p1, p2, p3) count -- p4's
    # conc_high_ctx_low_target row must be excluded entirely.
    assert conc_all["n"].iloc[0] == 3
    assert conc_all["mean_delta"].iloc[0] == pytest.approx((0.01 + 0.0 + 0.33) / 3)


def test_axis_rollup_filters_on_passed_validated_not_summary_column():
    """axis_rollup must filter using the `validated` argument directly, not
    trust whatever direction_validated the summary happened to be built
    with -- a caller passing a DIFFERENT validated set than the summary was
    built with must get results consistent with the NEW set."""
    summary = stratified_summary(_board(), validated={"conc_high_ctx_low_target"})
    rollup = axis_rollup(summary, validated={"conc_low_ctx_high_target"})
    conc_all = rollup[(rollup.axis == "concentration") & (rollup.stratum == "all")]
    assert conc_all["n"].iloc[0] == 3  # still conc_low_ctx_high_target's rows, not conc_high's


# ---------------------------------------------------------------------------
# minimum_detectable_effect
# ---------------------------------------------------------------------------


def test_mde_matches_hand_computed_value_for_known_data():
    # 4 proteins with per-protein mean deltas 0.0, 0.02, -0.01, 0.03
    board = pd.DataFrame(
        [
            {"protein": "p1", "delta": 0.0},
            {"protein": "p2", "delta": 0.02},
            {"protein": "p3", "delta": -0.01},
            {"protein": "p4", "delta": 0.03},
        ]
    )
    result = minimum_detectable_effect(board)
    per_protein = board.groupby("protein")["delta"].mean()
    expected_sd = per_protein.std(ddof=1)
    expected_se = expected_sd / (4**0.5)
    assert result["n_proteins"] == 4
    assert result["between_protein_sd"] == pytest.approx(expected_sd)
    assert result["se"] == pytest.approx(expected_se)
    # standard 95%/80% z-sum is ~2.80
    assert result["mde"] == pytest.approx(2.8016 * expected_se, rel=1e-3)


def test_mde_averages_multiple_rows_per_protein_before_clustering():
    # p1 has two rows (should average to one cluster mean, not double-count)
    board = pd.DataFrame(
        [
            {"protein": "p1", "delta": 0.0},
            {"protein": "p1", "delta": 0.02},
            {"protein": "p2", "delta": 0.01},
            {"protein": "p3", "delta": -0.01},
        ]
    )
    result = minimum_detectable_effect(board)
    assert result["n_proteins"] == 3  # 3 distinct proteins, not 4 rows


def test_mde_handles_fewer_than_two_proteins():
    board = pd.DataFrame([{"protein": "p1", "delta": 0.05}])
    result = minimum_detectable_effect(board)
    assert result["n_proteins"] == 1
    import math

    assert math.isnan(result["mde"])
