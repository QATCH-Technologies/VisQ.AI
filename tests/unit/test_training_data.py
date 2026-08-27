import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from visqai.training.data import (
    ZERO_VARIANCE_FALLBACK_SCALE,
    _drop_blank_rows,
    _fix_zero_variance_scale,
    _build_ctx_tensor,
    _build_tgt_tensors,
    compute_viscosity_weights,
    load_and_preprocess,
)


def test_zero_variance_column_gets_fallback_scale_not_degenerate_one():
    """P0 fix: sklearn's StandardScaler sets scale_=1 for a zero-variance
    training column (see _handle_zeros_in_scale), which means an OOD held-out
    value passes through the fitted transform almost raw. This is the
    at-fit-time guard that replaces that degenerate 1.0 with a fixed,
    generous a-priori scale instead."""
    num_cols = ["constant_col", "varying_col"]
    df = pd.DataFrame(
        {
            "constant_col": [5.0, 5.0, 5.0, 5.0],
            "varying_col": [1.0, 2.0, 3.0, 4.0],
        }
    )
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), num_cols)])
    X = preprocessor.fit_transform(df)

    scaler = preprocessor.named_transformers_["num"]
    assert scaler.scale_[0] == 1.0  # sklearn's degenerate default, pre-fix
    assert scaler.var_[0] == 0.0
    assert scaler.var_[1] > 0.0  # sanity: the varying column is untouched

    X_fixed = _fix_zero_variance_scale(preprocessor, X, num_cols)

    assert scaler.scale_[0] == ZERO_VARIANCE_FALLBACK_SCALE
    assert scaler.scale_[1] != ZERO_VARIANCE_FALLBACK_SCALE  # varying col left alone

    # An OOD held-out value for the constant column now maps to a bounded
    # z-score under the fixed scale, not the raw (value - mean) it would
    # have gotten under sklearn's scale_=1.
    ood_value = np.array([[500.0, 2.5]])
    z = preprocessor.transform(pd.DataFrame(ood_value, columns=num_cols))
    raw_passthrough_z = (500.0 - scaler.mean_[0]) / 1.0
    assert abs(z[0, 0]) < abs(raw_passthrough_z)
    assert z[0, 0] == pytest.approx((500.0 - scaler.mean_[0]) / ZERO_VARIANCE_FALLBACK_SCALE)


def test_no_zero_variance_columns_leaves_scaler_untouched():
    num_cols = ["a", "b"]
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), num_cols)])
    X = preprocessor.fit_transform(df)
    scaler = preprocessor.named_transformers_["num"]
    original_scale = scaler.scale_.copy()

    X_fixed = _fix_zero_variance_scale(preprocessor, X, num_cols)

    assert np.array_equal(scaler.scale_, original_scale)
    assert np.array_equal(X_fixed, X)


# ---------------------------------------------------------------------------
# Pre-2.4 data-hygiene fix (issue1_query_conditioned_correction_plan.md
# Finding C): fully-blank rows (trailing/malformed CSV rows -- no
# Protein_type AND no Protein_conc) must be dropped before load_and_preprocess
# fits the StandardScaler, or they contaminate it as all-zero(ish) rows.
# ---------------------------------------------------------------------------


def test_drop_blank_rows_removes_fully_blank_rows():
    df = pd.DataFrame(
        {
            "Protein_type": ["mab", None, "mab2"],
            "Protein_conc": [100.0, None, 150.0],
            "kP": [3.0, None, 3.5],
        }
    )
    out = _drop_blank_rows(df)
    assert len(out) == 2
    assert list(out["Protein_type"]) == ["mab", "mab2"]


def test_drop_blank_rows_keeps_row_with_only_protein_conc():
    """A row missing Protein_type but with a real Protein_conc is NOT the
    blank/malformed case this targets -- keep it (matches the 86 legitimate
    Protein_type=='none' buffer-only rows, which have a real Protein_conc of
    0.0, i.e. not NaN, so this filter never touches them)."""
    df = pd.DataFrame(
        {
            "Protein_type": [None, "mab"],
            "Protein_conc": [50.0, 100.0],
        }
    )
    out = _drop_blank_rows(df)
    assert len(out) == 2


def test_drop_blank_rows_leftover_stray_value_still_dropped():
    """The real-data case: a row with Protein_type AND Protein_conc both NaN,
    but a stray leftover Viscosity_100 value -- still fully blank by this
    filter's definition and must be dropped."""
    df = pd.DataFrame(
        {
            "Protein_type": ["mab", None],
            "Protein_conc": [100.0, None],
            "Viscosity_100": [12.0, 19.0],
        }
    )
    out = _drop_blank_rows(df)
    assert len(out) == 1
    assert out["Protein_type"].iloc[0] == "mab"


def test_drop_blank_rows_no_blanks_is_a_noop():
    df = pd.DataFrame({"Protein_type": ["mab", "mab2"], "Protein_conc": [100.0, 150.0]})
    out = _drop_blank_rows(df)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# load_and_preprocess: end-to-end against a small synthetic CSV. Rows need
# >=3 of the 5 SHEAR_MAP viscosity columns populated to survive
# load_and_preprocess's own per-sample resampling filter (`len(raw_x) < 3`).
# ---------------------------------------------------------------------------

def _raw_row(**overrides):
    row = {
        "ID": "s1",
        "Protein_type": "trastuzumab",
        "Protein_class_type": "mab_igg1",
        "Protein_conc": 100.0,
        "Buffer_pH": 6.0,
        "PI_mean": 8.5,
        "Whole_Antibody_Charge_at_Buffer_pH": 12.0,
        "kP": 1.0,
        "MW": 148000.0,
        "PI_range": 0.5,
        "Temperature": 25.0,
        "Buffer_conc": 20.0,
        "Salt_conc": 50.0,
        "Salt_type": "nacl",
        "Stabilizer_conc": 0.1,
        "Stabilizer_type": "sucrose",
        "Surfactant_conc": 0.02,
        "Surfactant_type": "tween-80",
        "Excipient_conc": 100.0,
        "Excipient_type": "arginine",
        "C_Class": 1.0,
        "HCI": 0.0,
        "Viscosity_100": 20.0,
        "Viscosity_1000": 15.0,
        "Viscosity_10000": 11.0,
        "Viscosity_100000": 8.0,
        "Viscosity_15000000": 3.0,
    }
    row.update(overrides)
    return row


def _write_csv(tmp_path, rows, name="formulations.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_and_preprocess_returns_one_sample_per_valid_row(tmp_path):
    rows = [
        _raw_row(ID="s1", Protein_type="trastuzumab"),
        _raw_row(ID="s2", Protein_type="bevacizumab", Viscosity_1000=25.0),
    ]
    csv_path = _write_csv(tmp_path, rows)

    samples, static_dim, physics_scaler, protected_indices = load_and_preprocess(csv_path)

    assert len(samples) == 2
    assert static_dim > 0
    assert isinstance(protected_indices, list)
    for s in samples:
        assert set(s.keys()) >= {"static", "points", "group", "id"}
        assert s["static"].shape == (static_dim,)
        assert s["points"].ndim == 2 and s["points"].shape[1] == 2
        assert s["static"].dtype == torch.float32
        assert s["points"].dtype == torch.float32
    assert {s["group"] for s in samples} == {"trastuzumab", "bevacizumab"}

    # physics_scaler was fit on (log10_shear, log10_visc) pairs.
    assert physics_scaler.mean_.shape == (2,)


def test_load_and_preprocess_drops_rows_with_fewer_than_three_shear_points(tmp_path):
    """A row with only 2 of the 5 shear columns populated can't be
    PCHIP-interpolated (needs >=3 points) and must be silently excluded from
    `samples`, even though it still contributed to fitting the ColumnTransformer."""
    good = _raw_row(ID="good")
    sparse = _raw_row(
        ID="sparse",
        Protein_type="bevacizumab",
        Viscosity_100=None,
        Viscosity_10000=None,
        Viscosity_100000=None,
    )
    csv_path = _write_csv(tmp_path, [good, sparse])

    samples, _, _, _ = load_and_preprocess(csv_path)

    assert len(samples) == 1
    assert samples[0]["id"] == "good"


def test_load_and_preprocess_drops_fully_blank_rows(tmp_path):
    good = _raw_row(ID="good")
    blank = {"ID": "blank"}  # no Protein_type, no Protein_conc
    csv_path = _write_csv(tmp_path, [good, blank])

    samples, _, _, _ = load_and_preprocess(csv_path)

    assert len(samples) == 1
    assert samples[0]["id"] == "good"


def test_load_and_preprocess_saves_artifacts_when_save_dir_given(tmp_path):
    import joblib

    rows = [_raw_row(ID="s1")]
    csv_path = _write_csv(tmp_path, rows)
    save_dir = tmp_path / "out"

    samples, static_dim, physics_scaler, protected_indices = load_and_preprocess(csv_path, save_dir=save_dir)

    preprocessor = joblib.load(save_dir / "preprocessor.pkl")
    loaded_scaler = joblib.load(save_dir / "physics_scaler.pkl")
    loaded_indices = joblib.load(save_dir / "protected_indices.pkl")

    assert hasattr(preprocessor, "transform")
    assert np.allclose(loaded_scaler.mean_, physics_scaler.mean_)
    assert loaded_indices == protected_indices


def test_load_and_preprocess_clamps_non_positive_viscosity_before_log10(tmp_path):
    """A zero/negative viscosity reading would make log10() blow up
    (-inf/NaN); load_and_preprocess clamps it to a tiny positive epsilon
    first so PCHIP interpolation and the physics scaler never see that."""
    rows = [_raw_row(ID="s1", Viscosity_100=0.0, Viscosity_1000=-5.0)]
    csv_path = _write_csv(tmp_path, rows)

    samples, _, physics_scaler, _ = load_and_preprocess(csv_path)

    assert len(samples) == 1
    assert torch.isfinite(samples[0]["points"]).all()
    assert np.isfinite(physics_scaler.mean_).all()


def test_load_and_preprocess_accepts_xlsx(tmp_path):
    rows = [_raw_row(ID="s1")]
    xlsx_path = tmp_path / "formulations.xlsx"
    pd.DataFrame(rows).to_excel(xlsx_path, index=False)

    samples, *_ = load_and_preprocess(xlsx_path)
    assert len(samples) == 1


# ---------------------------------------------------------------------------
# _build_ctx_tensor / _build_tgt_tensors
# ---------------------------------------------------------------------------

def _toy_samples(n=3, n_points=2, static_dim=4):
    return [
        {
            "static": torch.arange(static_dim, dtype=torch.float32) + i * 100,
            "points": torch.arange(n_points * 2, dtype=torch.float32).reshape(n_points, 2) + i * 1000,
        }
        for i in range(n)
    ]


def test_build_ctx_tensor_shape_and_content():
    samples = _toy_samples(n=3, n_points=2, static_dim=4)
    device = torch.device("cpu")
    ctx = _build_ctx_tensor(samples, [0, 2], device)

    # [1, sum(n_points over selected indices), 2 + static_dim]
    assert ctx.shape == (1, 4, 6)
    # First row's context is sample 0's first point concatenated with its static features.
    expected_first_row = torch.cat([samples[0]["points"][0], samples[0]["static"]])
    assert torch.allclose(ctx[0, 0], expected_first_row)


def test_build_tgt_tensors_shapes():
    samples = _toy_samples(n=3, n_points=2, static_dim=4)
    device = torch.device("cpu")
    q_x, q_stat, q_y = _build_tgt_tensors(samples, [1], device)

    assert q_x.shape == (1, 2, 1)
    assert q_y.shape == (1, 2, 1)
    assert q_stat.shape == (1, 2, 4)
    assert torch.allclose(q_x[0, :, 0], samples[1]["points"][:, 0])
    assert torch.allclose(q_y[0, :, 0], samples[1]["points"][:, 1])


def test_build_tgt_tensors_empty_indices_returns_all_none():
    samples = _toy_samples(n=2)
    q_x, q_stat, q_y = _build_tgt_tensors(samples, [], torch.device("cpu"))
    assert q_x is None
    assert q_stat is None
    assert q_y is None


# ---------------------------------------------------------------------------
# compute_viscosity_weights
# ---------------------------------------------------------------------------

def test_compute_viscosity_weights_are_mean_normalised_to_one():
    qy_scaled = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
    weights = compute_viscosity_weights(qy_scaled, visc_mean=0.0, visc_scale=1.0)
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-5)


def test_compute_viscosity_weights_upweight_high_viscosity_points():
    """Points well above `threshold` (in real log10-cP space) must get a
    strictly higher weight than points well below it."""
    qy_scaled = torch.tensor([[-5.0], [5.0]])  # -> log_visc = -5, 5 with mean=0/scale=1
    weights = compute_viscosity_weights(qy_scaled, visc_mean=0.0, visc_scale=1.0, threshold=2.0, max_weight=4.0)
    assert weights[1] > weights[0]


def test_compute_viscosity_weights_detached_from_graph():
    qy_scaled = torch.tensor([[1.0]], requires_grad=True)
    weights = compute_viscosity_weights(qy_scaled, visc_mean=0.0, visc_scale=1.0)
    assert weights.requires_grad is False
