import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from visqai.eval.logo_eval import (
    CONTEXT_GATE_TOLERANCE,
    _assert_context_gate,
    _check_fold_feature_range,
    run_cnp_logo,
)
from visqai.eval.logo_eval import LogoGroup


def _make_row(**overrides):
    row = {
        "ID": "s1",
        "Protein_type": "trastuzumab",
        "Protein_class_type": "mab_igg1",
        "Protein_conc": 100.0,
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
        "C_Class": 1.0,
        "HCI": 0.0,
    }
    row.update(overrides)
    return row


def _fit_fold_preprocessor(work_dir, train_df):
    """Minimal stand-in for training.data.load_and_preprocess -- builds and
    dumps a real preprocessor.pkl from train_df's engineered features,
    without the torch-training machinery the guard doesn't need."""
    from visqai.features.dataprocessor import build_feature_frame

    built, num_cols, cat_cols = build_feature_frame(train_df.copy())
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    preprocessor.fit(built)
    os.makedirs(work_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(work_dir, "preprocessor.pkl"))


def test_guard_fires_on_out_of_range_held_out_salt_concentration(tmp_path):
    rng = np.random.RandomState(0)
    train_rows = [
        _make_row(ID=f"train{i}", Protein_conc=rng.uniform(20, 150), Salt_type="none", Salt_conc=0.0)
        for i in range(20)
    ]
    train_df = pd.DataFrame(train_rows)
    _fit_fold_preprocessor(str(tmp_path), train_df)

    # Held-out fold: nacl at a concentration the salt-free training fold
    # never had any reason to represent.
    held_df = pd.DataFrame([_make_row(ID="held1", Salt_type="nacl", Salt_conc=175.0)])

    violations = _check_fold_feature_range(str(tmp_path), held_df)
    assert "Salt_conc" in violations["out_of_range"]
    assert violations["out_of_range"]["Salt_conc"]["n_bad"] == 1


def test_guard_is_silent_when_held_out_values_are_in_range(tmp_path):
    rng = np.random.RandomState(1)
    train_rows = [
        _make_row(
            ID=f"train{i}",
            Protein_conc=rng.uniform(20, 150),
            Salt_type="nacl",
            Salt_conc=rng.uniform(50, 150),
        )
        for i in range(20)
    ]
    train_df = pd.DataFrame(train_rows)
    _fit_fold_preprocessor(str(tmp_path), train_df)

    held_df = pd.DataFrame([_make_row(ID="held1", Salt_type="nacl", Salt_conc=100.0)])

    violations = _check_fold_feature_range(str(tmp_path), held_df)
    assert violations["out_of_range"] == {}


def test_context_gate_raises_when_fewshot_regresses_past_tolerance():
    rows = [
        {"axis": "protein", "group": "good", "lift": 0.02},
        {"axis": "protein", "group": "belatacept", "lift": -0.048},  # 0.077 -> 0.125
    ]
    with pytest.raises(AssertionError, match="belatacept"):
        _assert_context_gate(rows)


def test_context_gate_allows_small_negative_lift_within_tolerance():
    # A single mildly-negative group must not trip the PER-GROUP tolerance --
    # paired here with a positive group so the scoreboard-wide MEAN also
    # clears its own floor, isolating the per-group check from the mean-lift
    # check (see test_context_gate_enforces_mean_lift_floor below for that).
    rows = [
        {"axis": "protein", "group": "flat", "lift": -CONTEXT_GATE_TOLERANCE / 2},
        {"axis": "protein", "group": "good", "lift": 0.05},
    ]
    _assert_context_gate(rows)  # should not raise


def test_context_gate_enforces_mean_lift_floor():
    # Every group individually clears the per-group tolerance, but the mean
    # across the board is negative -- a systemic regression the per-group
    # check alone would wave through.
    rows = [
        {"axis": "protein", "group": "a", "lift": -CONTEXT_GATE_TOLERANCE / 2},
        {"axis": "protein", "group": "b", "lift": -CONTEXT_GATE_TOLERANCE / 2},
        {"axis": "protein", "group": "c", "lift": CONTEXT_GATE_TOLERANCE / 4},
    ]
    with pytest.raises(AssertionError, match="mean lift"):
        _assert_context_gate(rows)


def test_context_gate_ignores_rows_without_a_lift_column():
    rows = [
        {"axis": "protein", "group": "errored", "error": "too few held-out rows"},
        {"axis": "protein", "group": "nan_lift", "lift": float("nan")},
    ]
    _assert_context_gate(rows)  # should not raise


def test_run_cnp_logo_propagates_context_gate_failure_uncaught(tmp_path, monkeypatch):
    """The gate must run OUTSIDE the per-fold try/except in run_cnp_logo --
    a violation should abort the run, not get swallowed into an 'error' row
    like a normal per-fold exception would."""

    def _fake_run_cnp_fold(train_df, held_df, group, work_dir, **kwargs):
        return {"axis": group.axis, "group": group.key, "lift": -1.0}

    monkeypatch.setattr("visqai.eval.logo_eval.run_cnp_fold", _fake_run_cnp_fold)

    df = pd.DataFrame({"Protein_type": ["mab", "mab", "other", "other"], "dummy": [1, 2, 3, 4]})
    group = LogoGroup(axis="protein", key="mab", column="Protein_type", value="mab")

    with pytest.raises(AssertionError, match="Context gate failed"):
        run_cnp_logo(df, "protein", str(tmp_path), groups=[group])
