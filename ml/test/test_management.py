import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
from src.management import (
    attach_adapter,
    expand_processor_and_model,
    load_model_checkpoint,
    save_model_checkpoint,
)

# --- MOCK CONFIGURATION ---
mock_config = MagicMock()
mock_config.TARGETS = ["Viscosity_1"]
sys.modules["inference.config"] = mock_config


class TestManagement(unittest.TestCase):
    def setUp(self):
        # Common setup for mocks
        self.mock_processor = MagicMock()
        self.mock_processor.cat_maps = {"Protein_type": ["A", "B"]}
        self.mock_processor.numeric_features = ["Conc"]
        self.mock_processor.categorical_features = ["Protein_type"]
        self.mock_processor.scaler = MagicMock()
        self.mock_processor.scaler.mean_ = np.array([0.0])
        self.mock_processor.scaler.scale_ = np.array([1.0])
        self.mock_processor.split_indices = {}

        self.mock_model = MagicMock(spec=nn.Module)
        self.mock_model.cat_maps = {"Protein_type": ["A", "B"]}
        self.mock_model.cat_feature_names = ["Protein_type"]
        self.mock_model.state_dict.return_value = {"layer.weight": torch.tensor([1.0])}

        self.best_params = {"hidden_size": 10, "n_layers": 2, "dropout": 0.1}

    def test_expand_processor_and_model(self):
        """Test that expansion calls the correct methods on both objects."""
        expand_processor_and_model(
            self.mock_processor,
            self.mock_model,
            "Protein_type",
            ["NewProt"],
            initialization="mean",
        )

        # Check Processor updated
        self.mock_processor.add_categories.assert_called_with(
            "Protein_type", ["NewProt"]
        )
        # Check Model updated
        self.mock_model.expand_categorical_embedding.assert_called_with(
            "Protein_type", ["NewProt"], "mean"
        )

    @patch("inference.management.ResidualAdapter")
    def test_attach_adapter_wiring(self, MockAdapter):
        """
        Test that attach_adapter correctly initializes the adapter
        and monkey-patches the forward method.
        """
        # Setup Model Structure to calculate dimensions
        # 1 Linear Layer (Base)
        mock_linear = MagicMock(spec=nn.Linear)
        mock_linear.in_features = 20
        self.mock_model.base = [mock_linear]

        # 1 Embedding Layer
        mock_emb = MagicMock(spec=nn.Embedding)
        mock_emb.embedding_dim = 5
        self.mock_model.embeddings = [mock_emb]

        # Setup Adapter
        mock_adapter_instance = MockAdapter.return_value
        # Mock forward of adapter to return a tensor
        mock_adapter_instance.return_value = torch.ones(1, 1)

        # Dummy State Dict
        adapter_state = {"weight": torch.tensor([1.0])}

        # Call function
        # We assume Protein_type (size 2) is the gate. Threshold is index 1.
        attached_adapter = attach_adapter(
            self.mock_model, adapter_state, gating_thresholds={"Protein_type": 1}
        )

        # Verify Adapter Init
        # Numeric Dim = 20 (total) - 5 (emb) = 15
        MockAdapter.assert_called_with(15, [2], embed_dim=16)
        mock_adapter_instance.load_state_dict.assert_called_with(adapter_state)

        # Verify Monkey Patch
        # Create inputs
        x_num = torch.zeros(2, 15)
        x_cat = torch.tensor([[0], [2]])  # Row 0: Index 0 (Old). Row 1: Index 2 (New).

        # Mock original forward to return 0
        self.mock_model.forward.return_value = torch.zeros(
            2, 1
        )  # This is the "original_forward"

        # Run NEW forward (the patched one)
        # Note: self.mock_model.forward IS the new function now
        output = self.mock_model.forward(x_num, x_cat)

        # Logic Check:
        # Row 0 (Index 0 < Threshold 1): Base(0) + Adapter(1)*0 = 0
        # Row 1 (Index 2 >= Threshold 1): Base(0) + Adapter(1)*1 = 1
        self.assertEqual(output[0].item(), 0.0)
        self.assertEqual(output[1].item(), 1.0)

    @patch("torch.save")
    def test_save_model_checkpoint(self, mock_save):
        """Verify dictionary structure passed to torch.save."""
        save_model_checkpoint(
            self.mock_model, self.mock_processor, self.best_params, "test.pt"
        )

        # Extract the args passed to save
        save_args = mock_save.call_args[0]
        checkpoint = save_args[0]
        filepath = save_args[1]

        self.assertEqual(filepath, "test.pt")
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("scaler_state", checkpoint)
        self.assertEqual(checkpoint["best_params"], self.best_params)
        self.assertFalse(checkpoint["has_adapter"])  # No adapter passed

    @patch("inference.management.Model")
    @patch("inference.management.DataProcessor")
    @patch("torch.load")
    def test_load_model_checkpoint_structure(self, mock_load, MockProcessor, MockModel):
        """Test reconstruction of objects from checkpoint dict."""
        # Setup Checkpoint Dict
        checkpoint = {
            "model_state_dict": {"layer.weight": 1},
            "best_params": self.best_params,
            "cat_maps": {"A": [1]},
            "scaler_state": {"mean_": [0.0], "scale_": [1.0], "n_features_in_": 1},
        }
        mock_load.return_value = checkpoint

        # Call load
        model, processor, params = load_model_checkpoint("test.pt")

        # Verify Processor Reconstruction
        MockProcessor.assert_called()
        self.assertTrue(processor.is_fitted)
        self.assertEqual(processor.cat_maps, {"A": [1]})

        # Verify Model Reconstruction
        MockModel.assert_called()
        # Verify params matched
        call_kwargs = MockModel.call_args[1]
        self.assertEqual(
            call_kwargs["hidden_sizes"], [10, 10]
        )  # n_layers=2 * hidden=10

        # Verify State Loaded
        model.load_state_dict.assert_called_with(checkpoint["model_state_dict"])

    @patch("inference.management.Model")
    @patch("inference.management.DataProcessor")
    @patch("torch.load")
    def test_load_checkpoint_ensemble_stripping(
        self, mock_load, MockProcessor, MockModel
    ):
        """Test handling of 'models.0.' prefix in checkpoint keys (legacy ensemble)."""
        checkpoint = {
            "model_state_dict": {"models.0.layer.weight": 1, "models.0.bias": 0},
            "best_params": self.best_params,
            "scaler_state": {"mean_": [0], "scale_": [1], "n_features_in_": 1},
        }
        mock_load.return_value = checkpoint

        model, _, _ = load_model_checkpoint("test.pt")

        # Extract the state dict passed to the model
        loaded_dict = model.load_state_dict.call_args[0][0]

        # Keys should be stripped
        self.assertIn("layer.weight", loaded_dict)
        self.assertNotIn("models.0.layer.weight", loaded_dict)


if __name__ == "__main__":
    unittest.main()
