# Ensure local imports work if running as script
import os
import sys
import unittest
from unittest.mock import MagicMock

import torch
from src.loss import PhysicsInformedLoss, get_physics_masks

sys.path.append(os.getcwd())


class TestPhysicsMasks(unittest.TestCase):
    """Tests for the helper function get_physics_masks."""

    def setUp(self):
        self.device = "cpu"
        self.mock_processor = MagicMock()

        # Setup standard mock attributes
        self.mock_processor.categorical_features = [
            "Protein_type",
            "Regime",
            "Excipient_type",
        ]
        self.mock_processor.numeric_features = ["Concentration", "HCI"]
        self.mock_processor.scalable_features = ["Concentration", "HCI"]

        # Setup maps
        self.mock_processor.cat_maps = {
            "Regime": ["none", "near", "mixed", "far"],
            "Excipient_type": ["none", "l-arginine", "sucrose", "tween-20"],
            "Stabilizer_type": ["sucrose"],  # Extra for robust checks
            "Surfactant_type": ["tween-20"],
        }

        # Setup scaler mock
        self.mock_processor.scaler = MagicMock()
        self.mock_processor.scaler.mean_ = [0.0, 0.0]  # No shift
        self.mock_processor.scaler.scale_ = [1.0, 1.0]  # No scaling

    def test_regime_mask_creation(self):
        """Test that 'near' and 'mixed' regimes generate a positive mask."""
        # Indices: 0=none, 1=near, 2=mixed, 3=far
        X_cat = torch.tensor(
            [
                [1, 1, 0],  # Row 0: Regime=1 (near) -> Should be 1
                [0, 3, 0],  # Row 1: Regime=3 (far)  -> Should be 0
                [0, 2, 0],  # Row 2: Regime=2 (mixed)-> Should be 1
            ],
            device=self.device,
        )

        # Dummy numeric
        X_num = torch.zeros(3, 2)

        masks = get_physics_masks(X_cat, X_num, self.mock_processor)

        expected = torch.tensor([[1.0], [0.0], [1.0]])
        self.assertTrue(torch.equal(masks["is_near_mixed"], expected))

    def test_hci_threshold_logic(self):
        """Test detection of High HCI values."""
        # HCI is at index 1 in numeric_features
        # Threshold is defined in module (usually 1.3)
        X_cat = torch.zeros(2, 3, dtype=torch.long)

        # Row 0: HCI = 0.5 (Low)
        # Row 1: HCI = 1.5 (High)
        X_num = torch.tensor([[0.0, 0.5], [0.0, 1.5]])

        masks = get_physics_masks(X_cat, X_num, self.mock_processor)

        self.assertEqual(masks["is_high_hci"][0].item(), 0.0)
        self.assertEqual(masks["is_high_hci"][1].item(), 1.0)

    def test_substring_matching_excipients(self):
        """Test that substring matching works (e.g. 'l-arginine' matches 'arginine')."""
        # Excipient_type map: ["none", "l-arginine", "sucrose", "tween-20"]
        # Indices: 0, 1, 2, 3

        # Row 0: Index 1 (l-arginine) -> Should match 'arginine'
        # Row 1: Index 2 (sucrose)    -> Should NOT match 'arginine'
        X_cat = torch.tensor([[0, 0, 1], [0, 0, 2]])
        X_num = torch.zeros(2, 2)

        masks = get_physics_masks(X_cat, X_num, self.mock_processor)

        self.assertEqual(masks["has_arginine"][0].item(), 1.0)
        self.assertEqual(masks["has_arginine"][1].item(), 0.0)

    def test_processor_none_check(self):
        """Ensure robust failure if processor is missing."""
        with self.assertRaises(ValueError):
            get_physics_masks(torch.zeros(1, 1), torch.zeros(1, 1), None)


class TestPhysicsInformedLoss(unittest.TestCase):
    """Tests for the PhysicsInformedLoss class logic."""

    def setUp(self):
        # We need to map feature names to indices for the loss to find them
        self.numeric_cols = ["Salt_conc", "Stabilizer_conc", "Surfactant_conc"]
        self.loss_fn = PhysicsInformedLoss(
            lambda_shear=1.0, lambda_input=1.0, numeric_cols=self.numeric_cols
        )
        self.device = "cpu"

    def test_shear_thinning_constraint(self):
        """
        Test Shear Thinning: Viscosity should decrease or stay constant as shear rate increases.
        Columns of 'pred' represent increasing shear rates.
        Pred[i+1] > Pred[i] is a VIOLATION.
        """
        # Batch 1, 3 targets (Shear rates)
        # Prediction: [1.0, 2.0, 1.0].
        # Step 1->2: 2.0 > 1.0 (Violation, diff=1.0)
        # Step 2->3: 1.0 < 2.0 (OK)
        pred = torch.tensor([[1.0, 2.0, 1.0]], requires_grad=True)
        target = torch.tensor([[1.0, 2.0, 1.0]])  # Perfect match so MSE is 0

        # Dummy inputs/masks required by forward signature
        inputs_num = torch.zeros(1, 3, requires_grad=True)
        masks = {}

        loss = self.loss_fn(pred, target, inputs_num, masks)

        # Expected: MSE(0) + Shear(1.0) + Input(0) = 1.0
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_gradient_salt_violation(self):
        """
        Test Salt Constraint: In 'near/mixed' regimes, Salt should REDUCE viscosity.
        Therefore, Gradient(Viscosity w.r.t Salt) should be NEGATIVE.
        Positive Gradient = Penalty.
        """
        # 1. Inputs: Salt is index 0. Value=1.0
        inputs_num = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)

        # 2. Fake Model: Viscosity = 2 * Salt. (Positive Slope/Gradient = +2)
        # This is a VIOLATION.
        pred = inputs_num * 2.0
        target = pred.detach()  # MSE = 0

        # 3. Mask: Active Regime
        masks = {"is_near_mixed": torch.tensor([[1.0]])}

        loss = self.loss_fn(pred, target, inputs_num, masks)

        # Expected:
        # MSE = 0
        # Shear = 0 (only 1 output col here effectively, code loops range(cols-1))
        # Input: Grad is +2. Violation is ReLU(+2) = 2.
        # Total = 2.0 * lambda_input(1.0) = 2.0
        # Note: Shear loop range(pred.shape[1]-1). If shape is (1,3) loop runs.
        # Here pred shape is (1,3) because inputs_num is (1,3) * 2.
        # pred = [2, 0, 0].
        # Shear check:
        # 0 - 2 = -2 (OK)
        # 0 - 0 = 0 (OK)
        # So Shear loss is 0.

        self.assertAlmostEqual(loss.item(), 2.0, places=5)

    def test_gradient_stabilizer_violation(self):
        """
        Test Stabilizer Constraint: Stabilizers should INCREASE viscosity.
        Gradient should be POSITIVE.
        Negative Gradient = Penalty.
        """
        # Stabilizer is index 1
        inputs_num = torch.tensor([[0.0, 1.0, 0.0]], requires_grad=True)

        # Fake Model: Viscosity = -5 * Stabilizer (Gradient = -5)
        # VIOLATION.
        pred = inputs_num * -5.0
        target = pred.detach()

        # Mask: Has Stabilizer
        masks = {"has_stabilizer": torch.tensor([[1.0]])}

        loss = self.loss_fn(pred, target, inputs_num, masks)

        # Input Logic: ReLU(-grad). Grad is -5. -(-5) = +5.
        # Loss = 5.0
        self.assertAlmostEqual(loss.item(), 5.0, places=5)

    def test_gradient_surfactant_violation(self):
        """
        Test Surfactant: In High HCI (hydrophobic), Surfactants REDUCE viscosity.
        Gradient should be NEGATIVE.
        Positive Gradient = Penalty.
        """
        # Surfactant is index 2
        inputs_num = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)

        # Fake Model: Viscosity = 3 * Surfactant (Gradient = +3)
        # VIOLATION.
        pred = inputs_num * 3.0
        target = pred.detach()

        # Mask: High HCI AND Has Tween
        masks = {
            "is_high_hci": torch.tensor([[1.0]]),
            "has_tween20": torch.tensor([[1.0]]),
            "has_tween80": torch.tensor([[0.0]]),
        }

        loss = self.loss_fn(pred, target, inputs_num, masks)

        # Input Logic: Grad(+3). Relu(3) = 3.
        self.assertAlmostEqual(loss.item(), 3.0, places=5)

    def test_no_masks_active(self):
        """If masks are 0, gradients shouldn't matter."""
        inputs_num = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
        pred = inputs_num * 10.0  # Huge positive gradient
        target = pred.detach()

        masks = {
            "is_near_mixed": torch.tensor([[0.0]]),  # Inactive
            "is_high_hci": torch.tensor([[0.0]]),
            "has_stabilizer": torch.tensor([[0.0]]),
        }

        loss = self.loss_fn(pred, target, inputs_num, masks)
        self.assertAlmostEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
