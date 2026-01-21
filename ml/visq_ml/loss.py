"""Loss function module for the Viscosity library.

This module contains the PhysicsInformedLoss class and helper functions for
generating physics-based boolean masks used in loss calculation. It enforces
domain-specific constraints such as shear-thinning behavior and specific
chemical component interactions.

Attributes:
    HCI_THRESHOLD (float): Threshold value (1.3) used to determine high
        Hydrophobic Interaction Chromatography (HCI) regimes.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-01-15

Version:
    1.0
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.nn as nn

try:
    if TYPE_CHECKING:
        from .data import DataProcessor
except ImportError:
    if TYPE_CHECKING:
        from visq_ml.data import DataProcessor

HCI_THRESHOLD = 1.3


def get_physics_masks(
    X_cat: torch.Tensor, X_num: torch.Tensor, processor: "DataProcessor"
) -> Dict[str, torch.Tensor]:
    """Generates boolean masks for physics constraints based on batch data.

    Identifies specific regimes (e.g., "near", "mixed") and chemical components
    (e.g., arginine, sucrose) within the input batch to apply conditional
    physics-informed loss constraints.

    Args:
        X_cat: Tensor containing categorical features indices of shape
            (batch_size, n_cat_features).
        X_num: Tensor containing numeric features of shape
            (batch_size, n_num_features).
        processor: DataProcessor instance used to decode categorical maps
            and unscale numeric values for logic checks.

    Returns:
        A dictionary mapping mask names (str) to binary tensors (torch.Tensor)
        of shape (batch_size, 1), indicating the presence of specific conditions.

    Raises:
        ValueError: If the processor argument is None.
    """
    if processor is None:
        raise ValueError("Processor cannot be None when generating physics masks.")
    masks = {}
    device = X_cat.device
    try:
        if "Regime" in processor.categorical_features:
            idx_regime = processor.categorical_features.index("Regime")
            regime_map = processor.cat_maps["Regime"]
            near_idx = [i for i, x in enumerate(regime_map) if "near" in x.lower()]
            mixed_idx = [i for i, x in enumerate(regime_map) if "mixed" in x.lower()]
            target_regimes = torch.tensor(near_idx + mixed_idx, device=device)
            masks["is_near_mixed"] = (
                torch.isin(X_cat[:, idx_regime], target_regimes).float().unsqueeze(1)
            )
        else:
            masks["is_near_mixed"] = torch.zeros(X_cat.shape[0], 1, device=device)
        if "HCI" in processor.numeric_features:
            hci_idx = processor.numeric_features.index("HCI")
            hci_vals = X_num[:, hci_idx]

            if (
                "HCI" in processor.scalable_features
                and hasattr(processor.scaler, "mean_")
                and processor.scaler.mean_ is not None
                and hasattr(processor.scaler, "scale_")
                and processor.scaler.scale_ is not None
            ):
                scaler_idx = processor.scalable_features.index("HCI")
                mean_val = processor.scaler.mean_[scaler_idx]
                scale_val = processor.scaler.scale_[scaler_idx]
                mean_t = torch.tensor(mean_val, device=device, dtype=torch.float32)
                scale_t = torch.tensor(scale_val, device=device, dtype=torch.float32)
                hci_vals = hci_vals * scale_t + mean_t

            masks["is_high_hci"] = (hci_vals >= HCI_THRESHOLD).float().unsqueeze(1)
        else:
            masks["is_high_hci"] = torch.zeros(X_cat.shape[0], 1, device=device)

        def get_mask(col_name: str, search_terms: List[str]) -> torch.Tensor:
            """Creates a mask if a categorical column contains any search terms.

            Args:
                col_name: The name of the categorical column to check.
                search_terms: A list of substrings to search for in the column's
                    vocabulary.

            Returns:
                A binary tensor of shape (batch_size, 1).
            """
            if col_name not in processor.categorical_features:
                return torch.zeros(X_cat.shape[0], 1, device=device)

            col_idx = processor.categorical_features.index(col_name)
            vocab = processor.cat_maps.get(col_name, [])

            # Find all vocab indices that match ANY search term
            target_indices = []
            for i, v in enumerate(vocab):
                v_str = str(v).lower()
                if any(term in v_str for term in search_terms):
                    target_indices.append(i)

            if not target_indices:
                return torch.zeros(X_cat.shape[0], 1, device=device)

            target_t = torch.tensor(target_indices, device=device)
            return torch.isin(X_cat[:, col_idx], target_t).float().unsqueeze(1)

        masks["has_arginine"] = get_mask("Excipient_type", ["arginine"])
        masks["has_lysine"] = get_mask("Excipient_type", ["lysine"])
        masks["has_proline"] = get_mask("Excipient_type", ["proline"])
        masks["has_stabilizer"] = get_mask("Stabilizer_type", ["sucrose", "trehalose"])
        masks["has_tween20"] = get_mask("Surfactant_type", ["20"])
        masks["has_tween80"] = get_mask("Surfactant_type", ["80"])

    except Exception as e:
        print(f"Warning: Could not build physics masks: {e}")
        return {
            k: torch.zeros(X_cat.shape[0], 1, device=device)
            for k in [
                "is_near_mixed",
                "is_high_hci",
                "has_arginine",
                "has_lysine",
                "has_proline",
                "has_stabilizer",
                "has_tween20",
                "has_tween80",
            ]
        }

    return masks


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss function incorporating domain knowledge constraints.

    This loss function combines standard MSE loss with regularization terms
    enforcing physical plausibility, specifically shear-thinning behavior
    and gradient constraints based on chemical composition.

    Attributes:
        lambda_shear (float): Weighting factor for the shear-thinning constraint.
        lambda_input (float): Weighting factor for input gradient constraints.
        numeric_cols (List[str]): Names of numeric columns used to map features
            to their indices.
        idx_map (Dict[str, int]): Mapping from feature name to index.
    """

    def __init__(
        self,
        lambda_shear: float = 1.0,
        lambda_input: float = 0.1,
        numeric_cols: Optional[List[str]] = None,
    ):
        """Initializes the PhysicsInformedLoss.

        Args:
            lambda_shear: Weight for shear-thinning constraint (output consistency).
                Defaults to 1.0.
            lambda_input: Weight for input-gradient constraints (Table D rules).
                Defaults to 0.1.
            numeric_cols: List of numeric column names to identify indices for
                gradient calculations. Defaults to None.
        """
        super().__init__()
        self.lambda_shear = lambda_shear
        self.lambda_input = lambda_input
        self.numeric_cols = numeric_cols or []

        # Pre-calculate indices for numeric features if possible
        self.idx_map = {name: i for i, name in enumerate(self.numeric_cols)}

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        inputs_num: torch.Tensor,
        masks: Dict[str, torch.Tensor],
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates the combined physics-informed loss.

        Computes MSE loss and adds penalties for violations of physical constraints:
        1. Shear Thinning: Viscosity should decrease or stay constant as shear rate increases.
        2. Input Gradients: Specific components (salts, excipients) should have
           directional impacts on viscosity (increase/decrease) in specific regimes.

        Args:
            pred: Model predictions of shape (batch_size, n_targets).
            target: Ground truth values of shape (batch_size, n_targets).
            inputs_num: Numeric inputs of shape (batch_size, n_features).
                Must have requires_grad=True for gradient constraint calculation.
            masks: Dictionary of boolean masks identifying regimes and components
                for the current batch.
            weights: Optional sample weights of shape (batch_size,) or
                (batch_size, 1). Defaults to None.

        Returns:
            A scalar Tensor representing the total calculated loss.
        """
        # Standard MSE Loss
        squared_diff = (pred - target) ** 2
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(1)
            mse_loss = (squared_diff * weights).mean()
        else:
            mse_loss = squared_diff.mean()

        # Shear Thinning Constraint (Output Monotonicity)
        # Viscosity should decrease (or stay same) as shear rate increases
        shear_loss = 0.0
        for i in range(pred.shape[1] - 1):
            # diff > 0 means Viscosity(High Shear) > Viscosity(Low Shear) -> Violation
            diff = pred[:, i + 1] - pred[:, i]
            violation = torch.relu(diff)
            shear_loss += violation.mean()

        # Input Gradient Constraints
        input_loss = 0.0

        if self.lambda_input > 0.001 and inputs_num.requires_grad:
            grads = torch.autograd.grad(
                outputs=pred.sum(),
                inputs=inputs_num,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True,
            )[0]

            if grads is not None:
                # Ionic/Excipient Viscosity Reduction
                # Logic: In "Near/Mixed" regimes, Salts and Amino Acids (Arg, Lys, Pro)
                # should reduce viscosity. Positive gradient = Violation.

                # Create composite "Viscosity Reducer" mask from specific amino acids
                # Using clamp(max=1) effectively performs a logical OR on the float tensors
                zero_t = torch.tensor(0.0, device=pred.device)

                # Ionic/Excipient Viscosity Reduction ---
                mask_amino = (
                    masks.get("has_arginine", zero_t)
                    + masks.get("has_lysine", zero_t)
                    + masks.get("has_proline", zero_t)
                ).clamp(max=1.0)

                # Combine with Regime mask
                mask_reduction_regime = masks.get("is_near_mixed", 0)

                # Check Salt Gradients
                if "Salt_conc" in self.idx_map:
                    idx_s = self.idx_map["Salt_conc"]
                    grad_salt = grads[:, idx_s]
                    # Salts generally reduce viscosity in these regimes regardless of excipient
                    input_loss += (torch.relu(grad_salt) * mask_reduction_regime).mean()

                # Check Excipient Gradients (masked by specific amino acid presence)
                if "Excipient_conc" in self.idx_map:
                    idx_e = self.idx_map["Excipient_conc"]
                    grad_excipient = grads[:, idx_e]

                    # Only enforce if it is one of our target excipients AND in the right regime
                    mask_a = mask_reduction_regime * mask_amino
                    input_loss += (torch.relu(grad_excipient) * mask_a).mean()

                # Stabilizers (Sucrose/Trehalose)
                # Logic: Stabilizers INCREASE viscosity. Negative gradient = Violation.
                if "Stabilizer_conc" in self.idx_map:
                    idx_suc = self.idx_map["Stabilizer_conc"]
                    grad_suc = grads[:, idx_suc]

                    # Use the new generalized stabilizer mask
                    mask_b = masks.get("has_stabilizer", 0)
                    input_loss += (torch.relu(-grad_suc) * mask_b).mean()

                # Constraint C: Surfactants (Tween 20/80)
                # Logic: In High HCI (hydrophobic) regimes, Surfactants REDUCE viscosity.
                # Positive gradient = Violation.
                if "Surfactant_conc" in self.idx_map:
                    idx_tw = self.idx_map["Surfactant_conc"]
                    grad_tw = grads[:, idx_tw]
                    mask_has_tween = (
                        masks.get("has_tween20", zero_t)
                        + masks.get("has_tween80", zero_t)
                    ).clamp(max=1.0)

                    mask_c = masks.get("is_high_hci", zero_t) * mask_has_tween
                    input_loss += (torch.relu(grad_tw) * mask_c).mean()

        total_loss = (
            mse_loss
            + (self.lambda_shear * shear_loss)
            + (self.lambda_input * input_loss)
        )
        return total_loss
