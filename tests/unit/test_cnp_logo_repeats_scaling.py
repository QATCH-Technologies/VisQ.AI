"""
test_cnp_logo_repeats_scaling.py
=================================
Unit tests for cnp_logo._effective_n_repeats -- the fix for a real
belatacept context-gate failure traced to a too-small n_repeats for a small
held-out group's few-shot estimate (see the constant's docstring in
cnp_logo.py for the full empirical story: 5 repeats gave all_context_lift
-0.0117, a 30-repeat rerun of the identical fold settled at +0.0074).
"""

from __future__ import annotations

from visqai.eval.cnp_logo import (
    MAX_N_REPEATS_MULTIPLIER,
    REFERENCE_N_HELD_FOR_REPEATS,
    _effective_n_repeats,
)


def test_belatacept_sized_group_reaches_the_empirically_verified_30_repeats():
    assert _effective_n_repeats(n_held=12, n_repeats=5) == 30


def test_reference_sized_group_is_unchanged():
    assert _effective_n_repeats(n_held=REFERENCE_N_HELD_FOR_REPEATS, n_repeats=5) == 5


def test_large_group_is_unchanged_not_reduced():
    assert _effective_n_repeats(n_held=1000, n_repeats=5) == 5


def test_scaling_is_capped_at_max_multiplier():
    # A tiny group (below the cap threshold) must not scale past the cap.
    assert _effective_n_repeats(n_held=1, n_repeats=5) == 5 * MAX_N_REPEATS_MULTIPLIER


def test_scaling_decreases_monotonically_with_group_size():
    sizes = [12, 16, 20, 30, 50, 80, 150]
    repeats = [_effective_n_repeats(n_held=n, n_repeats=5) for n in sizes]
    assert all(a >= b for a, b in zip(repeats, repeats[1:]))


def test_zero_n_held_does_not_divide_by_zero():
    assert _effective_n_repeats(n_held=0, n_repeats=5) == 5
