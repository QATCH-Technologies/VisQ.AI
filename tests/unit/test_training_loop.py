"""
Behavioral tests for training/loop.py's train/validation functions. These
are hot-path deep-learning internals (see visqai.validation's module
docstring for why they're deliberately NOT input-validated), but that's a
separate question from whether they're correct -- they compute the actual
loss/gradient-clipping/checkpoint-selection logic the module's own docstring
describes several past bugs in, so real behavioral coverage matters here.

Assertions are shape/finiteness/gradient-flow checks rather than exact
values -- these functions are genuinely stochastic (random context/target
splits, dropout, AdamW), so bit-exact expected numbers would be flaky.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from visqai.models.cnp import CrossSampleCNP
from visqai.training.loop import (
    train_epoch,
    validate,
    validate_zero_shot,
    validate_fewshot,
    log_latent_variance,
    log_flatness,
)

DEVICE = torch.device("cpu")
STATIC_DIM = 4


def _make_samples(groups_and_counts, n_points=4, static_dim=STATIC_DIM, seed=0):
    """Synthetic samples in the exact dict shape load_and_preprocess
    produces: {"group": str, "points": [n_points, 2], "static": [static_dim]}.
    `points` col 0 is a monotonically increasing "shear" axis, col 1 a
    smooth-ish "log-visc" curve -- shape doesn't need to be physically
    meaningful, just finite and consistent."""
    rng = np.random.RandomState(seed)
    samples = []
    for group, count in groups_and_counts.items():
        for _ in range(count):
            shear = np.sort(rng.uniform(-2, 2, size=n_points))
            visc = -0.3 * shear + rng.normal(0, 0.1, size=n_points)
            points = np.stack([shear, visc], axis=1)
            samples.append(
                {
                    "group": group,
                    "points": torch.tensor(points, dtype=torch.float32),
                    "static": torch.tensor(rng.normal(size=static_dim), dtype=torch.float32),
                }
            )
    return samples


def _make_model(hidden_dim=16, latent_dim=8, dropout=0.0, seed=0):
    torch.manual_seed(seed)
    return CrossSampleCNP(static_dim=STATIC_DIM, hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout)


# --------------------------------------------------------------------------
# train_epoch
# --------------------------------------------------------------------------

def test_train_epoch_returns_finite_loss_and_per_group_mse():
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    samples = _make_samples({"proteinA": 6, "proteinB": 6, "proteinC": 6})

    avg_loss, per_group_mse = train_epoch(model, samples, optimizer, DEVICE, iterations=15)

    assert isinstance(avg_loss, float)
    assert np.isfinite(avg_loss)
    assert avg_loss >= 0
    assert isinstance(per_group_mse, dict)
    assert set(per_group_mse) <= {"proteinA", "proteinB", "proteinC"}
    for v in per_group_mse.values():
        assert np.isfinite(v)


def test_train_epoch_updates_model_parameters():
    """A real gradient must actually flow and update weights -- not just
    return a plausible-looking loss number."""
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    samples = _make_samples({"proteinA": 6, "proteinB": 6})

    before = [p.detach().clone() for p in model.parameters()]
    train_epoch(model, samples, optimizer, DEVICE, iterations=10)
    after = list(model.parameters())

    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))


def test_train_epoch_with_fewer_than_two_eligible_groups_is_a_safe_noop():
    """protein_list needs >=2 groups with >=4 members each; with only one
    qualifying group every iteration hits `continue` and train_epoch must
    still return cleanly (0 loss, no crash) rather than divide by zero."""
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    samples = _make_samples({"proteinA": 5})

    avg_loss, per_group_mse = train_epoch(model, samples, optimizer, DEVICE, iterations=5)

    assert avg_loss == 0.0
    assert per_group_mse == {}


@pytest.mark.parametrize("mask_prob", [0.0, 1.0])
def test_train_epoch_runs_at_either_masking_extreme(mask_prob):
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    samples = _make_samples({"proteinA": 6, "proteinB": 6})

    avg_loss, _ = train_epoch(model, samples, optimizer, DEVICE, iterations=8, mask_prob=mask_prob)
    assert np.isfinite(avg_loss)


def test_train_epoch_respects_protected_indices_under_full_masking():
    """With mask_prob=1.0 every iteration masks -- protected_indices must
    still survive (mask[..., prot_idx] = 1.0 in the module's own masking
    logic). Verified indirectly: a model that only ever sees protected
    features intact should differ from one where nothing is protected, for
    the same seed/data (the unprotected run zeroes strictly more inputs)."""
    samples = _make_samples({"proteinA": 6, "proteinB": 6})

    model_a = _make_model(seed=1)
    torch.manual_seed(42)
    train_epoch(
        model_a, samples, torch.optim.AdamW(model_a.parameters(), lr=1e-2), DEVICE,
        iterations=10, mask_prob=1.0, protected_indices=[0, 1],
    )

    model_b = _make_model(seed=1)
    torch.manual_seed(42)
    train_epoch(
        model_b, samples, torch.optim.AdamW(model_b.parameters(), lr=1e-2), DEVICE,
        iterations=10, mask_prob=1.0, protected_indices=None,
    )

    # Different masking behavior over 10 stochastic iterations should not
    # converge to bit-identical weights.
    params_a = torch.cat([p.detach().flatten() for p in model_a.parameters()])
    params_b = torch.cat([p.detach().flatten() for p in model_b.parameters()])
    assert not torch.allclose(params_a, params_b)


def test_train_epoch_with_group_weights_does_not_crash_and_stays_finite():
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    samples = _make_samples({"proteinA": 6, "proteinB": 6})
    weights = {"proteinA": 2.0, "proteinB": 0.5}

    avg_loss, per_group_mse = train_epoch(
        model, samples, optimizer, DEVICE, iterations=8, group_weights=weights
    )
    assert np.isfinite(avg_loss)


def test_train_epoch_with_physics_scaler_uses_weighted_loss_path():
    """physics_scaler not None exercises compute_viscosity_weights inside
    train_epoch instead of the plain F.mse_loss branch."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.array([[0.0, 0.0], [1.0, 1.0]]))  # [log_shear, log_visc] pairs

    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    samples = _make_samples({"proteinA": 6, "proteinB": 6})

    avg_loss, _ = train_epoch(model, samples, optimizer, DEVICE, iterations=8, physics_scaler=scaler)
    assert np.isfinite(avg_loss)


# --------------------------------------------------------------------------
# validate / validate_zero_shot / validate_fewshot
# --------------------------------------------------------------------------

def test_validate_returns_finite_nonnegative_float():
    model = _make_model()
    samples = _make_samples({"proteinA": 5, "proteinB": 5})
    result = validate(model, samples, DEVICE, n_repeats=2)
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert result >= 0


def test_validate_skips_groups_with_fewer_than_two_samples():
    model = _make_model()
    samples = _make_samples({"proteinA": 1})  # every group too small
    result = validate(model, samples, DEVICE)
    assert result == 0.0


def test_validate_zero_shot_matches_manual_r_zero_decode():
    """validate_zero_shot must use the exact r=0 path, not the encoder --
    cross-check it against decode_from_memory(zeros(...)) directly."""
    latent_dim = 8
    model = _make_model(latent_dim=latent_dim)
    model.eval()
    samples = _make_samples({"proteinA": 3})

    result = validate_zero_shot(model, samples, DEVICE, latent_dim=latent_dim)
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert result >= 0

    # Manual cross-check for the same single group.
    import torch.nn.functional as F

    s0 = samples[0]
    q_x = s0["points"][:, [0]].unsqueeze(0)
    q_stat = s0["static"].unsqueeze(0).repeat(s0["points"].shape[0], 1).unsqueeze(0)
    true_y = s0["points"][:, [1]].unsqueeze(0)
    for s in samples[1:]:
        q_x = torch.cat([q_x, s["points"][:, [0]].unsqueeze(0)], dim=1)
        q_stat = torch.cat([q_stat, s["static"].unsqueeze(0).repeat(s["points"].shape[0], 1).unsqueeze(0)], dim=1)
        true_y = torch.cat([true_y, s["points"][:, [1]].unsqueeze(0)], dim=1)
    with torch.no_grad():
        pred = model.decode_from_memory(torch.zeros((1, latent_dim)), q_x, q_stat)
        expected = F.mse_loss(pred, true_y).item()
    assert result == pytest.approx(expected, rel=1e-4)


def test_validate_fewshot_returns_inf_when_too_few_samples():
    model = _make_model()
    assert validate_fewshot(model, [_make_samples({"proteinA": 1})[0]], DEVICE) == float("inf")


def test_validate_fewshot_returns_finite_float_with_enough_samples():
    model = _make_model()
    samples = _make_samples({"proteinA": 6})
    result = validate_fewshot(model, samples, DEVICE, shots=(1, 2), n_repeats=2)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_validate_fewshot_skips_shots_not_smaller_than_pool():
    """shots >= len(val_samples) must be skipped, not raise."""
    model = _make_model()
    samples = _make_samples({"proteinA": 3})
    result = validate_fewshot(model, samples, DEVICE, shots=(1, 10), n_repeats=2)
    assert np.isfinite(result)


# --------------------------------------------------------------------------
# log_latent_variance / log_flatness
# --------------------------------------------------------------------------

def test_log_latent_variance_returns_nonnegative_float_with_enough_groups():
    model = _make_model()
    samples = _make_samples({"proteinA": 3, "proteinB": 3})
    result = log_latent_variance(model, samples, DEVICE)
    assert isinstance(result, float)
    assert result >= 0


def test_log_latent_variance_zero_with_fewer_than_two_qualifying_groups():
    model = _make_model()
    samples = _make_samples({"proteinA": 3})
    assert log_latent_variance(model, samples, DEVICE) == 0.0


def test_log_latent_variance_excludes_non_protein_groups():
    """"none" is in NON_PROTEIN_GROUPS -- with only one real protein group
    plus a "none" group, there still aren't 2 qualifying groups."""
    model = _make_model()
    samples = _make_samples({"proteinA": 3, "none": 3})
    assert log_latent_variance(model, samples, DEVICE) == 0.0


def test_log_flatness_returns_two_finite_floats():
    model = _make_model()
    samples = _make_samples({"proteinA": 2, "proteinB": 2, "proteinC": 2})
    shear_shape_std, cross_sample_std = log_flatness(model, samples, DEVICE, n_groups=3)
    assert np.isfinite(shear_shape_std)
    assert np.isfinite(cross_sample_std)
    assert shear_shape_std >= 0


def test_log_flatness_zero_when_no_protein_groups():
    model = _make_model()
    samples = _make_samples({"none": 3})
    assert log_flatness(model, samples, DEVICE) == (0.0, 0.0)


def test_log_flatness_cross_sample_std_zero_with_single_group():
    """cross_sample_std needs >=2 per-sample means -- with one group it must
    stay 0.0 rather than raise on a single-element std."""
    model = _make_model()
    samples = _make_samples({"proteinA": 2})
    _, cross_sample_std = log_flatness(model, samples, DEVICE)
    assert cross_sample_std == 0.0
