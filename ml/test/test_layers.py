import sys
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from src.layers import (
    EmbeddingDropout,
    LearnablePhysicsPrior,
    ResidualAdapter,
    ResidualBlock,
)

# --- MOCK CONFIGURATION ---
# We must mock config.TARGETS before importing layers because it is used
# in the ResidualAdapter __init__ method.
mock_config = MagicMock()
mock_config.TARGETS = ["Viscosity_1", "Viscosity_2", "Viscosity_3"]
sys.modules["inference.config"] = mock_config


class TestLayers(unittest.TestCase):

    def setUp(self):
        # Standardize random seed for reproducible tests
        torch.manual_seed(42)

    def test_learnable_physics_prior_math(self):
        """
        Verify the custom equation:
        Result = (Score + Delta) * (w_L * E_low + w_H * tanh(E_high))
        """
        # Dimensions: 1 Protein Class, 1 Regime, 1 Excipient
        layer = LearnablePhysicsPrior(n_classes=1, n_regimes=1, n_excipients=1)

        # Manually set parameters to known values
        # Static Score = 1.0
        layer.register_buffer("static_scores", torch.tensor([[[1.0]]]))

        # Delta = 0.5 (within clamp range)
        with torch.no_grad():
            layer.delta.fill_(0.5)
            # w_L = 2.0
            layer.w_L.fill_(2.0)
            # w_H = 0.5
            layer.w_H.fill_(0.5)

        # Inputs
        # Indices are all 0
        p_idx = torch.tensor([0])
        r_idx = torch.tensor([0])
        e_idx = torch.tensor([0])

        # Input Values
        e_low = torch.tensor([[1.0]])
        e_high = torch.tensor([[0.0]])  # tanh(0) = 0

        # Expected Calculation:
        # Term 1: (Score + Delta) = 1.0 + 0.5 = 1.5
        # Term 2: (w_L * e_low) + (w_H * tanh(e_high))
        #         = (2.0 * 1.0) + (0.5 * 0.0) = 2.0
        # Result: 1.5 * 2.0 = 3.0

        output, details = layer(p_idx, r_idx, e_idx, e_low, e_high)

        self.assertTrue(
            torch.isclose(output, torch.tensor([[3.0]])),
            f"Expected 3.0, got {output.item()}",
        )

    def test_physics_prior_clamping(self):
        """Test that the learned delta is clamped between -2.0 and 2.0."""
        layer = LearnablePhysicsPrior(1, 1, 1)

        # Force Delta to be large (e.g., 10.0)
        with torch.no_grad():
            layer.delta.fill_(10.0)

        p_idx = torch.tensor([0])
        r_idx = torch.tensor([0])
        e_idx = torch.tensor([0])
        e_low = torch.tensor([[1.0]])
        e_high = torch.tensor([[0.0]])

        _, details = layer(p_idx, r_idx, e_idx, e_low, e_high)

        # The 'd' used in calculation should be clamped to 2.0
        self.assertEqual(details["delta"].item(), 2.0)

    def test_physics_prior_tanh_saturation(self):
        """Test that the high concentration input is passed through tanh."""
        layer = LearnablePhysicsPrior(1, 1, 1)

        # Set w_H to 1, everything else to 0/neutral to isolate Term 2
        layer.register_buffer("static_scores", torch.zeros(1, 1, 1))
        with torch.no_grad():
            layer.delta.fill_(1.0)  # Base term = 1
            layer.w_L.fill_(0.0)
            layer.w_H.fill_(1.0)

        p_idx = torch.tensor([0])
        r_idx = torch.tensor([0])
        e_idx = torch.tensor([0])
        e_low = torch.tensor([[0.0]])

        # Input e_high is very large (e.g., 100).
        # If linear, result is huge. If tanh, result approaches 1.0.
        e_high = torch.tensor([[100.0]])

        output, _ = layer(p_idx, r_idx, e_idx, e_low, e_high)

        # Expected: (0+1) * (0 + 1 * tanh(100)) ~= 1 * 1 = 1
        self.assertTrue(torch.isclose(output, torch.tensor([[1.0]]), atol=1e-4))

    def test_embedding_dropout_logic(self):
        """
        Verify EmbeddingDropout drops entire rows (vectors), not individual elements.
        """
        # Probability 0.5
        dropout = EmbeddingDropout(p=0.5)
        dropout.train()

        # Batch of 100 samples, embedding dim 10
        x = torch.ones(100, 10)
        out = dropout(x)

        # 1. Check shapes match
        self.assertEqual(x.shape, out.shape)

        # 2. Check "All or Nothing" property
        # For every row, the sum should be either 0 (dropped) or 10 (kept)
        # It should NOT be some random number like 4.3 or 7.0
        row_sums = out.sum(dim=1)

        # Check that every row sum is essentially 0 or 10
        is_zero = torch.isclose(row_sums, torch.tensor(0.0))
        is_ten = torch.isclose(row_sums, torch.tensor(10.0))

        # Assert that ALL rows are either zero or ten
        self.assertTrue(
            torch.all(is_zero | is_ten),
            "Dropout should zero out entire rows, not partial features.",
        )

    def test_embedding_dropout_eval_mode(self):
        """Verify dropout is disabled in eval mode."""
        dropout = EmbeddingDropout(p=1.0)  # 100% dropout
        dropout.eval()

        x = torch.ones(5, 5)
        out = dropout(x)

        # Should be identical
        self.assertTrue(torch.equal(x, out))

    def test_residual_block_shapes(self):
        """Verify ResidualBlock maintains shapes."""
        dim = 16
        block = ResidualBlock(dim, dropout=0.1)

        x = torch.randn(10, dim)
        out = block(x)

        self.assertEqual(out.shape, (10, dim))

    def test_residual_adapter_integration(self):
        """
        Test the ResidualAdapter structure.
        It combines numeric inputs + categorical embeddings -> MLP -> Targets
        """
        # Mock setup: 2 Numeric inputs, 2 Categorical inputs (vocab sizes 5 and 3)
        # Mock TARGETS has 3 elements
        adapter = ResidualAdapter(numeric_dim=2, cat_dims=[5, 3], embed_dim=4)

        batch_size = 4
        x_num = torch.randn(batch_size, 2)
        x_cat = torch.zeros(batch_size, 2, dtype=torch.long)  # Indices

        out = adapter(x_num, x_cat)

        # Output dim should equal len(TARGETS) (3)
        self.assertEqual(out.shape, (batch_size, 3))


if __name__ == "__main__":
    unittest.main()
