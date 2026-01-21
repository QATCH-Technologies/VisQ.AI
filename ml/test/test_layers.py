import pytest
import torch
from visq_ml.layers import (
    EmbeddingDropout,
    LearnablePhysicsPrior,
    LearnableSoftThresholdPrior,
    ResidualAdapter,
    ResidualBlock,
)

# --- Constants for Testing ---
BATCH_SIZE = 8
N_CLASSES = 3  # e.g., IgG1, IgG4, Other
N_REGIMES = 4  # e.g., Near, Mixed, Far, NoProtein
N_EXCIPIENTS = 5  # e.g., NaCl, Arg, Sucrose, Tween, None
EMBED_DIM = 16
NUMERIC_DIM = 10


@pytest.fixture
def indices():
    """Generates random indices for batch testing."""
    p_idx = torch.randint(0, N_CLASSES, (BATCH_SIZE,))
    r_idx = torch.randint(0, N_REGIMES, (BATCH_SIZE,))
    e_idx = torch.randint(0, N_EXCIPIENTS, (BATCH_SIZE,))
    return p_idx, r_idx, e_idx


@pytest.fixture
def physics_layer():
    return LearnablePhysicsPrior(N_CLASSES, N_REGIMES, N_EXCIPIENTS)


@pytest.fixture
def soft_threshold_layer():
    # Initialize with dummy thresholds
    thresholds = torch.tensor([10.0, 50.0, 100.0, 5.0, 1.0])
    return LearnableSoftThresholdPrior(
        N_CLASSES, N_REGIMES, N_EXCIPIENTS, initial_thresholds=thresholds
    )


# --- Tests for LearnablePhysicsPrior ---


def test_physics_prior_forward_shape(physics_layer, indices):
    """Verifies output shape and existence of detail dictionary."""
    p_idx, r_idx, e_idx = indices

    # Simulate normalized low/high split concentrations
    e_low = torch.rand(BATCH_SIZE, 1)
    e_high = torch.rand(BATCH_SIZE, 1)

    output, details = physics_layer(p_idx, r_idx, e_idx, e_low, e_high)

    # Expect [Batch, 1] output (correction scalar per sample)
    assert output.shape == (BATCH_SIZE, 1)
    assert "base_term" in details
    assert "conc_term" in details


def test_physics_prior_gradients(physics_layer, indices):
    """Ensures learnable parameters (delta, weights) receive gradients."""
    p_idx, r_idx, e_idx = indices
    e_low = torch.rand(BATCH_SIZE, 1, requires_grad=True)
    e_high = torch.rand(BATCH_SIZE, 1, requires_grad=True)

    output, _ = physics_layer(p_idx, r_idx, e_idx, e_low, e_high)
    loss = output.sum()
    loss.backward()

    # Check if parameters have gradients
    assert physics_layer.delta.grad is not None
    assert physics_layer.w_L.grad is not None
    assert physics_layer.w_H.grad is not None

    # Static scores should NOT have gradients (buffer)
    assert physics_layer.static_scores.grad is None


def test_physics_prior_logic(physics_layer):
    """
    Manually verifies the math:
    Result = (Score + Delta) * (wL*Low + wH*tanh(High))
    """
    # Set known values for a single sample (Batch=1)
    p, r, e = 0, 0, 0
    physics_layer.static_scores[p, r, e] = 1.0  # Base Score
    physics_layer.delta.data[p, r, e] = 0.5  # Delta
    physics_layer.w_L.data[p, r, e] = 2.0  # wL
    physics_layer.w_H.data[p, r, e] = 0.0  # wH (ignore high for now)

    p_idx = torch.tensor([p])
    r_idx = torch.tensor([r])
    e_idx = torch.tensor([e])
    e_low = torch.tensor([[10.0]])  # Normalized value
    e_high = torch.tensor([[0.0]])

    output, _ = physics_layer(p_idx, r_idx, e_idx, e_low, e_high)

    # Expected: (1.0 + 0.5) * (2.0 * 10.0 + 0) = 1.5 * 20 = 30.0
    expected = 30.0
    assert torch.isclose(output, torch.tensor([[expected]]), atol=1e-5)


# --- Tests for LearnableSoftThresholdPrior (New Logic) ---


def test_soft_threshold_gating_logic(soft_threshold_layer):
    """
    Verifies that the soft gate switches behavior based on concentration vs threshold.
    """
    # Pick index 0: Threshold is 10.0
    p, r, e = 0, 0, 0
    p_idx = torch.tensor([p])
    r_idx = torch.tensor([r])
    e_idx = torch.tensor([e])

    # 1. Test Low Concentration (1.0 vs Threshold 10.0) -> Ratio 0.1
    # Gate should be close to 0 (Low Regime)
    conc_low = torch.tensor([[1.0]])
    _, details_low = soft_threshold_layer(p_idx, r_idx, e_idx, conc_low)
    gate_val_low = details_low["gate"].item()
    assert (
        gate_val_low < 0.1
    ), f"Gate should be low for conc < thresh, got {gate_val_low}"

    # 2. Test High Concentration (100.0 vs Threshold 10.0) -> Ratio 10.0
    # Gate should be close to 1 (High Regime)
    conc_high = torch.tensor([[100.0]])
    _, details_high = soft_threshold_layer(p_idx, r_idx, e_idx, conc_high)
    gate_val_high = details_high["gate"].item()
    assert (
        gate_val_high > 0.9
    ), f"Gate should be high for conc > thresh, got {gate_val_high}"


def test_soft_threshold_broadcast_shape(soft_threshold_layer, indices):
    """
    Crucial: Verifies that the internal unsqueeze/broadcasting logic works
    for a full batch without dimension mismatch errors.
    """
    p_idx, r_idx, e_idx = indices
    raw_conc = torch.rand(BATCH_SIZE, 1) * 100.0

    # This often fails if internal dimensions aren't handled right (e.g. [Batch] vs [Batch, 1])
    try:
        output, _ = soft_threshold_layer(p_idx, r_idx, e_idx, raw_conc)
    except RuntimeError as e:
        pytest.fail(f"Broadcasting error in SoftThresholdPrior: {e}")

    assert output.shape == (BATCH_SIZE, 1)


# --- Tests for ResidualAdapter ---


def test_residual_adapter_dimensions():
    """Verifies adapter accepts numeric + multiple categorical inputs."""
    cat_dims = [10, 5, 2]  # 3 Categorical features
    adapter = ResidualAdapter(NUMERIC_DIM, cat_dims, embed_dim=4)

    x_num = torch.randn(BATCH_SIZE, NUMERIC_DIM)
    x_cat = torch.stack([torch.randint(0, d, (BATCH_SIZE,)) for d in cat_dims], dim=1)

    # Output should correspond to TARGETS length (defined in config, usually 5)
    # The adapter hardcodes 'len(TARGETS)' via import, but if we can't control import easily
    # we check output shape is [Batch, Any]
    output = adapter(x_num, x_cat)
    assert output.ndim == 2
    assert output.shape[0] == BATCH_SIZE


# --- Tests for Helper Layers ---


def test_embedding_dropout():
    """Verifies entire vectors are zeroed out, not just elements."""
    emb_drop = EmbeddingDropout(p=0.5)  # High probability for testing

    # Shape: [Batch, Sequence/Features, Embed_Dim]
    # e.g. 2 samples, 4 features, 5 dim embedding
    x = torch.ones(2, 4, 5)

    # Force dropout (training mode)
    emb_drop.train()
    out = emb_drop(x)

    # Check that for any dropped vector, ALL elements are 0
    # Reshape to [Total_Vectors, Dim]
    flat = out.view(-1, 5)
    norms = flat.norm(dim=1)

    for n in norms:
        # Norm is either full (sqrt(5) approx 2.23) or 0
        assert torch.isclose(n, torch.tensor(0.0)) or n > 1.0


def test_residual_block_gradient_flow():
    """Verifies skip connection allows gradient flow."""
    block = ResidualBlock(dim=NUMERIC_DIM, dropout=0.0)
    x = torch.randn(BATCH_SIZE, NUMERIC_DIM, requires_grad=True)

    out = block(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    # In a residual block (y = x + f(x)), grad at x should be roughly 1 + f'(x)
    # It shouldn't vanish completely even if f(x) weights are tiny.
    assert torch.all(torch.abs(x.grad) > 0.0)
