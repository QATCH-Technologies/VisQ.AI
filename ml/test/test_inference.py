import sys
import unittest
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --- MOCK DEPENDENCIES ---
# We mock the neighbor modules so we can test inference.py in isolation.

mock_config = MagicMock()
mock_config.TARGETS = ["Viscosity"]

mock_layers = MagicMock()
# Mock ResidualAdapter as a simple Linear layer so we can actually run forward passes
mock_layers.ResidualAdapter = lambda in_dim, cat_dims, embed_dim: nn.Linear(in_dim, 1)
# Mock LearnablePhysicsPrior class type
mock_layers.LearnablePhysicsPrior = type("LearnablePhysicsPrior", (), {})

mock_management = MagicMock()
mock_models = MagicMock()
mock_utils = MagicMock()

# Apply mocks to sys.modules
sys.modules["inference.config"] = mock_config
sys.modules["inference.layers"] = mock_layers
sys.modules["inference.management"] = mock_management
sys.modules["inference.models"] = mock_models
sys.modules["inference.utils"] = mock_utils

# Now safe to import
from inference import ViscosityPredictor


class TestViscosityPredictor(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.dummy_path = "dummy_ckpt.pt"
        self.predictor = ViscosityPredictor(self.dummy_path, device=self.device)

        # Create a mock processor and model to be returned by load_model_checkpoint
        self.mock_processor = MagicMock()
        self.mock_processor.cat_maps = {
            "Protein_type": ["A", "B"],
            "Buffer": ["1", "2"],
        }
        self.mock_processor.transform.return_value = (
            np.zeros((5, 10)),
            np.zeros((5, 2)),
        )
        self.mock_processor.categorical_features = ["Protein_type", "Buffer"]
        self.mock_processor.detect_new_categories.return_value = (
            {}
        )  # Default no new cats

        self.mock_model = MagicMock(spec=nn.Module)
        self.mock_model.cat_feature_names = ["Protein_type", "Buffer"]
        # Mock forward to return a tensor of ones
        self.mock_model.return_value = torch.ones((5, 1))

        # Mock embeddings list for expansion tests
        self.mock_embedding = nn.Embedding(2, 4)
        self.mock_model.embeddings = nn.ModuleList([self.mock_embedding])
        self.mock_model.cat_maps = {"Protein_type": ["A", "B"]}

        self.mock_params = {"lr": 0.01}

    @patch("inference.load_model_checkpoint")
    def test_hydrate_single_model(self, mock_load):
        """Test lazy loading of a single model."""
        mock_load.return_value = (
            self.mock_model,
            self.mock_processor,
            self.mock_params,
        )

        self.predictor.hydrate()

        self.assertTrue(self.predictor._hydrated)
        self.assertEqual(self.predictor.model, self.mock_model)
        self.assertEqual(self.predictor.processor, self.mock_processor)
        # Verify it captured base vocab sizes
        self.assertEqual(self.predictor.base_vocab_sizes["Protein_type"], 2)

    @patch("inference.load_model_checkpoint")
    @patch("inference.glob.glob")
    def test_hydrate_ensemble(self, mock_glob, mock_load):
        """Test lazy loading of an ensemble of models."""
        # Setup ensemble predictor
        ensemble_pred = ViscosityPredictor("ensemble_dir", is_ensemble=True)

        # Mock file finding
        mock_glob.return_value = ["model1.pt", "model2.pt"]

        # Mock loading (returns same mock model for simplicity)
        mock_load.return_value = (
            self.mock_model,
            self.mock_processor,
            self.mock_params,
        )

        # We need to mock EnsembleModel class to accept the list
        with patch("inference.EnsembleModel") as MockEnsemble:
            ensemble_pred.hydrate()

            # Should load 2 checkpoints
            self.assertEqual(mock_load.call_count, 2)
            # Should instantiate EnsembleModel
            self.assertTrue(MockEnsemble.called)
            self.assertTrue(ensemble_pred._hydrated)

    @patch("inference.to_tensors")
    @patch("inference.inverse_log_transform")
    @patch("inference.load_model_checkpoint")
    def test_predict_flow(self, mock_load, mock_inv_log, mock_to_tensor):
        """Test the standard prediction pipeline."""
        mock_load.return_value = (
            self.mock_model,
            self.mock_processor,
            self.mock_params,
        )

        # Setup data transforms
        mock_to_tensor.return_value = (torch.randn(5, 10), torch.randn(5, 2))
        # Inverse log transform just returns input for verification
        mock_inv_log.side_effect = lambda x: x

        self.predictor.hydrate()

        df = pd.DataFrame({"Data": [1, 2, 3]})
        preds = self.predictor.predict(df)

        # Check flow
        self.mock_processor.transform.assert_called_once()
        self.mock_model.eval.assert_called()
        self.mock_model.assert_called()  # Forward pass
        self.assertIsInstance(preds, np.ndarray)

    @patch("inference.to_tensors")
    @patch("inference.inverse_log_transform")
    @patch("inference.load_model_checkpoint")
    def test_predict_uncertainty(self, mock_load, mock_inv_log, mock_to_tensor):
        """Test Monte Carlo Dropout prediction."""
        mock_load.return_value = (
            self.mock_model,
            self.mock_processor,
            self.mock_params,
        )
        mock_to_tensor.return_value = (torch.randn(5, 10), torch.randn(5, 2))
        mock_inv_log.side_effect = lambda x: x

        self.predictor.hydrate()

        results = self.predictor.predict_with_uncertainty(pd.DataFrame(), n_samples=10)

        # Model should have been called 10 times
        self.assertEqual(self.mock_model.call_count, 10)
        # Should return stats
        self.assertIn("mean", results)
        self.assertIn("lower_ci", results)
        self.assertIn("upper_ci", results)

    @patch("inference.load_model_checkpoint")
    @patch("inference.log_transform_targets")
    @patch("inference.to_tensors")
    def test_learn_adapter_training(self, mock_to_tensor, mock_log_trans, mock_load):
        """Test that calling learn() triggers adapter training and attachment."""
        mock_load.return_value = (
            self.mock_model,
            self.mock_processor,
            self.mock_params,
        )
        mock_to_tensor.return_value = (
            torch.randn(5, 10),
            torch.randn(5, 2),
            torch.randn(5, 1),
        )
        mock_log_trans.return_value = np.zeros((5, 1))

        self.predictor.hydrate()

        # Mock the optimizer to avoid actual step logic issues in test
        with patch("torch.optim.Adam") as mock_opt:
            self.predictor.learn(pd.DataFrame(), np.array([1, 2]), epochs=1)

            # Check adapter created
            self.assertIsNotNone(self.predictor.adapter)
            # Check optimizer stepped
            mock_opt.return_value.step.assert_called()
            # Check model forward was patched (adapter attached)
            self.assertNotEqual(
                self.predictor.model.forward, self.mock_model.return_value
            )

    @patch("inference.load_model_checkpoint")
    def test_gated_adapter_logic(self, mock_load):
        """
        CRITICAL TEST: Verify the monkey-patched forward method correctly
        gates the adapter based on whether categories are new or old.
        """
        # 1. Setup hydrated model
        mock_load.return_value = (
            self.mock_model,
            self.mock_processor,
            self.mock_params,
        )
        self.predictor.hydrate()

        # 2. Define Vocab Sizes (from hydrate)
        # Protein_type has size 2. Indices 0, 1 are OLD. Index 2+ is NEW.
        self.predictor.base_vocab_sizes = {"Protein_type": 2}

        # 3. Create a dummy adapter that adds +10 to everything
        # We manually attach it to test the logic without running full training
        self.predictor.adapter = MagicMock(spec=nn.Module)
        self.predictor.adapter.return_value = torch.full((2, 1), 10.0)  # Adds 10
        self.predictor.adapter.eval = MagicMock()

        # 4. Mock the base model to return 0s
        self.mock_model.forward = MagicMock(return_value=torch.zeros((2, 1)))

        # 5. Attach the logic
        self.predictor._attach_gated_adapter()

        # 6. Create Input Tensor
        # Row 0: Protein_type index 1 (OLD). Should NOT get adapter.
        # Row 1: Protein_type index 2 (NEW). SHOULD get adapter.
        x_num = torch.zeros(2, 10)
        x_cat = torch.tensor([[1, 0], [2, 0]])  # Col 0 is Protein_type

        # 7. Run the patched forward
        output = self.predictor.model.forward(x_num, x_cat)

        # 8. Assertions
        # Row 0: Base(0) + Adapter(10)*Mask(0) = 0
        self.assertEqual(output[0].item(), 0.0, "Old category should block adapter")
        # Row 1: Base(0) + Adapter(10)*Mask(1) = 10
        self.assertEqual(output[1].item(), 10.0, "New category should trigger adapter")

    @patch("inference.load_model_checkpoint")
    def test_smart_expand_category(self, mock_load):
        """Test dynamic expansion of embedding layers."""
        mock_load.return_value = (
            self.mock_model,
            self.mock_processor,
            self.mock_params,
        )
        self.predictor.hydrate()

        # Initial state: 2 embeddings
        self.assertEqual(self.mock_embedding.num_embeddings, 2)

        # Expand
        self.predictor._smart_expand_category("Protein_type", "NewProtein")

        # Check embeddings layer was replaced with larger one
        new_layer = self.mock_model.embeddings[0]
        self.assertEqual(new_layer.num_embeddings, 3)
        # Check map updated
        self.assertIn("NewProtein", self.mock_model.cat_maps["Protein_type"])
        # Check weights copied (index 0 should be same)
        self.assertTrue(torch.equal(new_layer.weight[0], self.mock_embedding.weight[0]))


if __name__ == "__main__":
    unittest.main()
