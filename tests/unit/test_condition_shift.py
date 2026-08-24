"""
test_condition_shift.py
========================
Task 0.1: unit tests for the pure split-construction and sanity-check logic
in visqai.eval.condition_shift -- concentration_split/ingredient_splits/
buffer_splits/shift_validity_check. These never train a model (that's an
expensive, environment-dependent integration path exercised manually via
cli/condition_shift_eval.py --quick); they test the deterministic DataFrame
partitioning and the shift-validity aggregation directly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from visqai.eval.condition_shift import (
    MIN_ROWS_FOR_CONCENTRATION_SPLIT,
    buffer_splits,
    concentration_split,
    ingredient_splits,
    shift_validity_check,
)


def _make_row(**overrides):
    row = {
        "ID": "s1",
        "Protein_type": "trastuzumab",
        "Protein_class_type": "mab_igg1",
        "Protein_conc": 100.0,
        "Buffer_type": "histidine",
        "Buffer_pH": 6.0,
        "PI_mean": 8.5,
        "kP": 1.0,
        "MW": 148000.0,
        "PI_range": 0.5,
        "Temperature": 25.0,
        "Buffer_conc": 20.0,
        "Salt_conc": 0.0,
        "Salt_type": "none",
        "Stabilizer_conc": 60.0,
        "Stabilizer_type": "sucrose",
        "Surfactant_conc": 0.02,
        "Surfactant_type": "tween-80",
        "Excipient_conc": 0.0,
        "Excipient_type": "none",
    }
    row.update(overrides)
    return row


def _held_df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# concentration_split
# ---------------------------------------------------------------------------


def test_concentration_split_produces_both_directions_with_correct_membership():
    rows = [_make_row(ID=f"s{i}", Protein_conc=c) for i, c in enumerate([10, 20, 80, 90])]
    held = _held_df(rows)
    splits = concentration_split(held)
    assert {d for d, _, _ in splits} == {"conc_low_ctx_high_target", "conc_high_ctx_low_target"}

    by_direction = {d: (ctx, tgt) for d, ctx, tgt in splits}
    ctx, tgt = by_direction["conc_low_ctx_high_target"]
    assert set(ctx["Protein_conc"]) == {10, 20}
    assert set(tgt["Protein_conc"]) == {80, 90}

    ctx2, tgt2 = by_direction["conc_high_ctx_low_target"]
    assert set(ctx2["Protein_conc"]) == {80, 90}
    assert set(tgt2["Protein_conc"]) == {10, 20}


def test_concentration_split_too_few_rows_returns_empty():
    rows = [_make_row(ID=f"s{i}", Protein_conc=c) for i, c in enumerate([10, 90])]
    assert len(rows) < MIN_ROWS_FOR_CONCENTRATION_SPLIT
    held = _held_df(rows)
    assert concentration_split(held) == []


def test_concentration_split_ignores_rows_with_missing_conc():
    rows = [_make_row(ID=f"s{i}", Protein_conc=c) for i, c in enumerate([10, 20, 80, 90])]
    rows.append(_make_row(ID="bad", Protein_conc=float("nan")))
    held = _held_df(rows)
    splits = concentration_split(held)
    total_rows = sum(len(ctx) + len(tgt) for _, ctx, tgt in splits) // 2
    assert total_rows == 4  # the NaN row never appears on either side


# ---------------------------------------------------------------------------
# ingredient_splits
# ---------------------------------------------------------------------------


def test_ingredient_splits_only_for_columns_with_both_present_and_absent():
    rows = [
        _make_row(ID="a", Salt_type="none"),
        _make_row(ID="b", Salt_type="nacl"),
        _make_row(ID="c", Stabilizer_type="sucrose"),
        _make_row(ID="d", Stabilizer_type="sucrose"),  # never absent -> no split for Stabilizer_type
    ]
    held = _held_df(rows)
    splits = ingredient_splits(held)
    directions = {d for d, _, _ in splits}
    assert "Salt_type_absent_ctx_present_target" in directions
    assert not any(d.startswith("Stabilizer_type") for d in directions)


def test_ingredient_split_membership_is_correct():
    rows = [
        _make_row(ID="a", Salt_type="none"),
        _make_row(ID="b", Salt_type="nacl"),
        _make_row(ID="c", Salt_type="nacl"),
    ]
    held = _held_df(rows)
    splits = ingredient_splits(held)
    salt = [(d, ctx, tgt) for d, ctx, tgt in splits if d.startswith("Salt_type")][0]
    _, ctx, tgt = salt
    assert list(ctx["ID"]) == ["a"]
    assert set(tgt["ID"]) == {"b", "c"}


def test_ingredient_splits_empty_when_no_ingredient_columns_present():
    held = _held_df([_make_row(ID="a"), _make_row(ID="b", Protein_conc=200.0)])
    # every row has the exact same ingredient categories -> no absent/present split anywhere
    assert ingredient_splits(held) == []


# ---------------------------------------------------------------------------
# buffer_splits
# ---------------------------------------------------------------------------


def test_buffer_splits_produces_ordered_pairs_for_all_distinct_types():
    rows = [
        _make_row(ID="a", Buffer_type="histidine"),
        _make_row(ID="b", Buffer_type="citrate"),
        _make_row(ID="c", Buffer_type="acetate"),
    ]
    held = _held_df(rows)
    splits = buffer_splits(held)
    directions = {d for d, _, _ in splits}
    # 3 types -> 3*2 = 6 ordered pairs
    assert len(directions) == 6
    assert "buffer_histidine_ctx_citrate_target" in directions
    assert "buffer_citrate_ctx_histidine_target" in directions


def test_buffer_splits_empty_when_single_buffer_type():
    rows = [_make_row(ID="a", Buffer_type="histidine"), _make_row(ID="b", Buffer_type="histidine")]
    held = _held_df(rows)
    assert buffer_splits(held) == []


# ---------------------------------------------------------------------------
# shift_validity_check
# ---------------------------------------------------------------------------


def test_shift_validity_check_ok_when_any_direction_shift_mae_higher():
    """Per-direction check (not pooled): a real acceptance run showed
    extrapolating UP in concentration is much harder than the random-split
    baseline while extrapolating DOWN is easier -- pooling them cancels the
    signal (see shift_validity_check's docstring), so `ok` must key off
    whether ANY direction clears the bar, not a blended average."""
    condition_shift_df = pd.DataFrame(
        [
            {
                "protein": "p1",
                "axis": "concentration",
                "direction": "conc_low_ctx_high_target",
                "prior_only_log_mae": 0.30,
            },
            {
                "protein": "p2",
                "axis": "concentration",
                "direction": "conc_low_ctx_high_target",
                "prior_only_log_mae": 0.34,
            },
            {
                "protein": "p1",
                "axis": "concentration",
                "direction": "conc_high_ctx_low_target",
                "prior_only_log_mae": 0.05,
            },
            {
                "protein": "p2",
                "axis": "concentration",
                "direction": "conc_high_ctx_low_target",
                "prior_only_log_mae": 0.05,
            },
        ]
    )
    random_split = pd.DataFrame(
        [{"group": "p1", "zero_shot_log_mae": 0.15}, {"group": "p2", "zero_shot_log_mae": 0.10}]
    )
    result = shift_validity_check(condition_shift_df, random_split)
    assert result["ok"] is True
    assert result["per_direction"]["conc_low_ctx_high_target"]["ok"] is True
    assert result["per_direction"]["conc_high_ctx_low_target"]["ok"] is False


def test_shift_validity_check_fails_when_no_direction_shift_mae_higher():
    condition_shift_df = pd.DataFrame(
        [
            {
                "protein": "p1",
                "axis": "concentration",
                "direction": "conc_low_ctx_high_target",
                "prior_only_log_mae": 0.10,
            },
            {
                "protein": "p2",
                "axis": "concentration",
                "direction": "conc_low_ctx_high_target",
                "prior_only_log_mae": 0.10,
            },
        ]
    )
    random_split = pd.DataFrame(
        [{"group": "p1", "zero_shot_log_mae": 0.20}, {"group": "p2", "zero_shot_log_mae": 0.20}]
    )
    result = shift_validity_check(condition_shift_df, random_split)
    assert result["ok"] is False
    assert result["per_direction"]["conc_low_ctx_high_target"]["ok"] is False


def test_shift_validity_check_handles_no_concentration_rows():
    condition_shift_df = pd.DataFrame(
        [{"protein": "p1", "axis": "buffer", "prior_only_log_mae": 0.10}]
    )
    random_split = pd.DataFrame([{"group": "p1", "zero_shot_log_mae": 0.20}])
    result = shift_validity_check(condition_shift_df, random_split)
    assert result["ok"] is False
    assert result["n_proteins"] == 0


def test_shift_validity_check_handles_no_overlapping_proteins():
    condition_shift_df = pd.DataFrame(
        [
            {
                "protein": "unknown_protein",
                "axis": "concentration",
                "direction": "conc_low_ctx_high_target",
                "prior_only_log_mae": 0.10,
            }
        ]
    )
    random_split = pd.DataFrame([{"group": "p1", "zero_shot_log_mae": 0.20}])
    result = shift_validity_check(condition_shift_df, random_split)
    assert result["ok"] is False
    assert result["n_proteins"] == 0
