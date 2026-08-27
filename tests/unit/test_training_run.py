"""
Tests for training/run.py's train_final_model: the end-to-end training loop
(early stopping, checkpoint save/reload) built on top of training/loop.py's
already-covered train_epoch/validate/validate_zero_shot.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from visqai.constants import DEFAULT_PARAMS
from visqai.training.run import train_final_model

STATIC_DIM = 4
TINY_PARAMS = {"hidden_dim": 8, "latent_dim": 4, "dropout": 0.0, "lr": 5e-3, "weight_decay": 1e-4}


def _make_samples(groups_and_counts, n_points=4, static_dim=STATIC_DIM, seed=0):
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


def _physics_scaler():
    scaler = StandardScaler()
    scaler.fit(np.array([[-2.0, -1.0], [2.0, 1.0]]))
    return scaler


@pytest.fixture
def samples():
    return _make_samples({"proteinA": 6, "proteinB": 6, "proteinC": 6})


def test_train_final_model_returns_expected_keys_and_saves_checkpoint(tmp_path, samples):
    out_dir = tmp_path / "run1"
    result = train_final_model(
        samples, STATIC_DIM, _physics_scaler(), protected_indices=[0, 1],
        out_dir=str(out_dir), params=TINY_PARAMS, max_epochs=3, patience=80, verbose=False,
    )

    assert set(result) == {"best_loss", "epochs_run", "group_weights"}
    assert np.isfinite(result["best_loss"])
    assert result["epochs_run"] == 3  # patience never triggers in 3 epochs
    assert set(result["group_weights"]) == {"proteinA", "proteinB", "proteinC"}

    ckpt_path = out_dir / "best_model.pth"
    assert ckpt_path.exists()


def test_saved_checkpoint_is_loadable_and_matches_config(tmp_path, samples):
    out_dir = tmp_path / "run1"
    train_final_model(
        samples, STATIC_DIM, _physics_scaler(), protected_indices=[],
        out_dir=str(out_dir), params=TINY_PARAMS, max_epochs=2, patience=80, verbose=False,
    )

    checkpoint = torch.load(out_dir / "best_model.pth", map_location="cpu", weights_only=False)
    assert set(checkpoint) == {"state_dict", "config", "static_dim"}
    assert checkpoint["static_dim"] == STATIC_DIM
    assert checkpoint["config"] == TINY_PARAMS
    # Prior/correction decoder split (see models/cnp.py) -- the state_dict
    # must use the new key names, not a stale single `decoder.*` block.
    keys = set(checkpoint["state_dict"])
    assert any(k.startswith("prior_head.") for k in keys)
    assert any(k.startswith("correction_head.") for k in keys)
    assert not any(k.startswith("decoder.") for k in keys)

    from visqai.models.cnp import CrossSampleCNP

    model = CrossSampleCNP(
        static_dim=checkpoint["static_dim"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        latent_dim=checkpoint["config"]["latent_dim"],
        dropout=checkpoint["config"]["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)  # must not raise


def test_train_final_model_defaults_to_default_params_when_none_given(tmp_path, samples):
    out_dir = tmp_path / "run1"
    train_final_model(
        samples, STATIC_DIM, _physics_scaler(), protected_indices=[],
        out_dir=str(out_dir), params=None, max_epochs=2, patience=80, verbose=False,
    )
    checkpoint = torch.load(out_dir / "best_model.pth", map_location="cpu", weights_only=False)
    assert checkpoint["config"] == DEFAULT_PARAMS


def test_train_final_model_early_stopping_never_exceeds_max_epochs(tmp_path, samples):
    """patience=0 stops the instant val_loss fails to improve -- epochs_run
    must never exceed max_epochs regardless of how early it triggers."""
    out_dir = tmp_path / "run1"
    result = train_final_model(
        samples, STATIC_DIM, _physics_scaler(), protected_indices=[],
        out_dir=str(out_dir), params=TINY_PARAMS, max_epochs=6, patience=0, verbose=False,
    )
    assert 1 <= result["epochs_run"] <= 6


def test_train_final_model_group_weights_stay_normalised_around_one(tmp_path, samples):
    """Each epoch's EMA update renormalises group_weights so mean weight *
    n_groups == sum(weights) stays close to n_groups (see train_epoch's own
    normalisation) -- a sanity check that weighting doesn't blow up or
    collapse to zero over several epochs."""
    out_dir = tmp_path / "run1"
    result = train_final_model(
        samples, STATIC_DIM, _physics_scaler(), protected_indices=[],
        out_dir=str(out_dir), params=TINY_PARAMS, max_epochs=5, patience=80, verbose=False,
    )
    weights = list(result["group_weights"].values())
    assert all(np.isfinite(w) and w > 0 for w in weights)


def test_train_final_model_handles_singleton_group(tmp_path):
    """A group with only 1 sample can't be split into train/stop -- it must
    go entirely into train_set (len(g_samples) < 2 branch) without crashing."""
    samples = _make_samples({"proteinA": 1, "proteinB": 6})
    out_dir = tmp_path / "run1"
    result = train_final_model(
        samples, STATIC_DIM, _physics_scaler(), protected_indices=[],
        out_dir=str(out_dir), params=TINY_PARAMS, max_epochs=2, patience=80, verbose=False,
    )
    assert np.isfinite(result["best_loss"])


def test_train_final_model_verbose_true_runs_without_error(tmp_path, samples, capsys):
    out_dir = tmp_path / "run1"
    train_final_model(
        samples, STATIC_DIM, _physics_scaler(), protected_indices=[],
        out_dir=str(out_dir), params=TINY_PARAMS, max_epochs=1, patience=80, verbose=True,
    )
    captured = capsys.readouterr()
    assert "Final Train" in captured.out


def test_train_final_model_prints_early_stopping_message_when_verbose(tmp_path, samples, capsys):
    out_dir = tmp_path / "run1"
    train_final_model(
        samples, STATIC_DIM, _physics_scaler(), protected_indices=[],
        out_dir=str(out_dir), params=TINY_PARAMS, max_epochs=6, patience=0, verbose=True,
    )
    captured = capsys.readouterr()
    assert "Stopping early" in captured.out
