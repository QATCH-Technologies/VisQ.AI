import sys
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

# --- MOCK CONFIGURATION & LAYERS ---
# We mock these BEFORE importing models.py to control the environment.
from visq_ml.models import EnsembleModel, Model

# 1. Mock Config
mock_config = MagicMock()
mock_config.CONC_TYPE_PAIRS = {"Salt_conc": "Salt_type"}
mock_config.EXCIPIENT_TYPE_MAPPING = {"Salt_type": ["nacl"]}
# Define a specific rule we can test for:
# If Protein=mAb1 and Regime=near, NaCl has effect -1.0
mock_config.EXCIPIENT_PRIORS = {("mab1", "near"): {"nacl": -1.0}}
sys.modules["inference.config"] = mock_config


# 2. Mock Layers
# We need a functional LearnablePhysicsPrior to test _init_static_priors
class MockPhysicsLayer(nn.Module):
    def __init__(self, n_classes, n_regimes, n_excipients):
        super().__init__()
        self.register_buffer(
            "static_scores", torch.zeros(n_classes, n_regimes, n_excipients)
        )

    def forward(self, p_idx, r_idx, e_idx, low, high):
        # Return dummy correction and details
        return torch.zeros(p_idx.shape[0], 1), {"static_score": 0}


mock_layers = MagicMock()
mock_layers.LearnablePhysicsPrior = MockPhysicsLayer
mock_layers.EmbeddingDropout = nn.Dropout  # Swap for standard dropout
mock_layers.ResidualBlock = nn.Linear  # Swap for simple linear
sys.modules["inference.layers"] = mock_layers

# Now safe to import


class TestModel(unittest.TestCase):
    def setUp(self):
        # Define a standard vocab map matching our mock config
        self.cat_maps = {
            "Protein_class_type": ["mAb1", "mAb2"],
            "Regime": ["near", "far"],
            "Salt_type": ["NaCl", "Water"],
            "Buffer_type": ["Histidine"],  # Irrelevant for physics
        }

        # Split indices for Salt_conc (indices 0 and 1 in numeric input)
        self.split_indices = {"Salt_conc": (0, 1)}

        self.model_params = {
            "cat_maps": self.cat_maps,
            "numeric_dim": 2,  # Salt_low, Salt_high
            "out_dim": 1,
            "hidden_sizes": [10],
            "dropout": 0.1,
            "split_indices": self.split_indices,
        }

    def test_layer_initialization(self):
        """Test that layers are created correctly based on config."""
        model = Model(**self.model_params)

        # 1. Check Embeddings created for all 4 features
        self.assertEqual(len(model.embeddings), 4)

        # 2. Check Physics Layers
        # Indices: 0=P_class, 1=Regime, 2=Salt_type, 3=Buffer_type

        # Salt_type (idx 2) is in CONC_TYPE_PAIRS and has split_indices.
        # Should be MockPhysicsLayer.
        self.assertIsInstance(model.physics_layers[2], MockPhysicsLayer)

        # Buffer_type (idx 3) is NOT in pairs. Should be Identity.
        self.assertIsInstance(model.physics_layers[3], nn.Identity)

        # Protein/Regime columns themselves get Identity physics layers
        self.assertIsInstance(model.physics_layers[0], nn.Identity)

    def test_static_prior_population(self):
        """
        Verify _init_static_priors correctly maps config dicts to the tensor.
        Config Rule: ('mab1', 'near') -> {'nacl': -1.0}
        """
        model = Model(**self.model_params)

        # Get the physics layer for Salt_type (Index 2)
        phys_layer = model.physics_layers[2]
        tensor = phys_layer.static_scores

        # Vocabulary Indices:
        # Protein 'mAb1' -> index 0
        # Regime 'near' -> index 0
        # Salt 'NaCl' -> index 0

        val = tensor[0, 0, 0].item()
        self.assertEqual(val, -1.0, "Did not map -1.0 effect from config to tensor")

        # Check defaults (mAb2, far, etc should be 0.0)
        self.assertEqual(tensor[1, 1, 0].item(), 0.0)

    def test_forward_shape(self):
        """Verify forward pass returns correct shape."""
        model = Model(**self.model_params)
        batch_size = 5

        # Inputs
        x_num = torch.randn(batch_size, 2)  # 2 numeric features
        x_cat = torch.zeros(batch_size, 4, dtype=torch.long)  # 4 cat features

        output = model(x_num, x_cat)
        self.assertEqual(output.shape, (batch_size, 1))

    def test_forward_returns_extras(self):
        """Verify return_features and return_physics_details."""
        model = Model(**self.model_params)
        x_num = torch.randn(2, 2)
        x_cat = torch.zeros(2, 4, dtype=torch.long)

        # Return tuple of 3
        ret = model(x_num, x_cat, return_features=True, return_physics_details=True)
        self.assertEqual(len(ret), 3)

        # Check features
        features = ret[1]
        self.assertIsInstance(features, torch.Tensor)

        # Check details dict
        details = ret[2]
        self.assertIsInstance(details, dict)
        # Should contain 'Salt_type' keys because that layer is active
        self.assertIn("Salt_type", details)

    def test_expand_embedding(self):
        """Test dynamic vocabulary expansion."""
        model = Model(**self.model_params)

        # Buffer_type has 1 item ("Histidine"). Expand with "Citrate".
        # Index 3
        old_emb = model.embeddings[3]
        self.assertEqual(old_emb.num_embeddings, 1)

        model.expand_categorical_embedding("Buffer_type", ["Citrate"])

        new_emb = model.embeddings[3]
        self.assertEqual(new_emb.num_embeddings, 2)
        self.assertIn("Citrate", model.cat_maps["Buffer_type"])

        # Verify old weight preservation (Index 0 should be identical)
        self.assertTrue(torch.equal(old_emb.weight[0], new_emb.weight[0]))

    def test_expand_physics_warning(self):
        """Expanding a column attached to a physics layer should warn."""
        model = Model(**self.model_params)

        # Salt_type (Index 2) has a physics layer
        with self.assertRaises(RuntimeWarning):
            model.expand_categorical_embedding("Salt_type", ["NewSalt"])


class TestEnsembleModel(unittest.TestCase):
    def setUp(self):
        # Create 2 dummy models
        self.m1 = MagicMock(spec=Model)
        self.m1.cat_maps = {"A": [1]}
        self.m1.cat_feature_names = ["A"]
        self.m1.embeddings = [nn.Embedding(1, 1)]
        # Make them output constant tensors
        self.m1.return_value = torch.tensor([[1.0]])

        self.m2 = MagicMock(spec=Model)
        self.m2.return_value = torch.tensor([[3.0]])

        self.ensemble = EnsembleModel([self.m1, self.m2])

    def test_forward_averaging(self):
        """Ensemble forward should average the outputs."""
        x_num = torch.zeros(1, 1)
        x_cat = torch.zeros(1, 1)

        # Model 1 -> 1.0, Model 2 -> 3.0. Average -> 2.0.
        out = self.ensemble(x_num, x_cat)
        self.assertEqual(out.item(), 2.0)

    def test_get_individual_predictions(self):
        """Should return stacked predictions."""
        x_num = torch.zeros(1, 1)
        x_cat = torch.zeros(1, 1)

        out = self.ensemble.get_individual_predictions(x_num, x_cat)
        # Shape: (n_models, batch, targets) -> (2, 1, 1)
        self.assertEqual(out.shape, (2, 1, 1))
        self.assertEqual(out[0].item(), 1.0)
        self.assertEqual(out[1].item(), 3.0)

    def test_proxy_attributes(self):
        """Ensemble should proxy attributes to the first model."""
        self.assertEqual(self.ensemble.cat_maps, self.m1.cat_maps)
        self.assertEqual(self.ensemble.cat_feature_names, self.m1.cat_feature_names)

    def test_ensemble_expansion(self):
        """Expanding ensemble should call expand on all models."""
        self.ensemble.expand_categorical_embedding("A", ["B"])

        self.m1.expand_categorical_embedding.assert_called_with("A", ["B"], "mean")
        self.m2.expand_categorical_embedding.assert_called_with("A", ["B"], "mean")


if __name__ == "__main__":
    unittest.main()
