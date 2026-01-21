"""
Model management module for VisQ.AI.

This module handles the lifecycle management of Viscosity models, including:
    - Checkpointing (saving and loading models, processors, and hyperparameters).
    - Model expansion (adding new categories to embeddings dynamically).
    - Adapter attachment (injecting residual adapters for fine-tuning).

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-01-15

Version:
    1.0
"""

from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.nn import Embedding, Linear

try:
    from .config import TARGETS
    from .data import DataProcessor
    from .layers import ResidualAdapter
    from .models import Model
except ImportError:
    from config import TARGETS
    from layers import ResidualAdapter

    from data import DataProcessor
    from models import Model


def expand_processor_and_model(
    processor: DataProcessor,
    model: Model,
    feature_name: str,
    new_categories: List[str],
    initialization: str = "mean",
) -> None:
    """
    Expand both the data processor and the model embeddings for new categories.

    Updates the processor's internal vocabulary to recognize the new categories and
    resizes the corresponding embedding layer in the model. The new embedding weights
    are initialized according to the specified strategy.

    Args:
        processor (DataProcessor): The data processor containing the categorical mappings.
        model (Model): The neural network model to update.
        feature_name (str): The name of the categorical feature to expand.
        new_categories (List[str]): A list of new category labels to add.
        initialization (str, optional): The initialization strategy for new embedding
            weights. Options are usually "mean" (average of existing weights) or "random".
            Defaults to "mean".

    Returns:
        None
    """
    processor.add_categories(feature_name, new_categories)
    model.expand_categorical_embedding(feature_name, new_categories, initialization)


def attach_adapter(
    model: Model,
    adapter_state_dict: Dict,
    gating_thresholds: Optional[Dict[str, int]] = None,
) -> nn.Module:
    """
    Reconstructs a residual adapter and attaches it to the model with dynamic gating.

    This function calculates the necessary dimensions from the base model, instantiates
    a `ResidualAdapter`, loads its weights, and monkeys-patches the `model.forward`
    method. The new forward pass applies the adapter output only when specific
    categorical features exceed a defined threshold (e.g., new protein types).

    Args:
        model (Model): The base model to which the adapter will be attached.
        adapter_state_dict (Dict): A state dictionary containing the trained weights
            for the adapter.
        gating_thresholds (Optional[Dict[str, int]]): A dictionary mapping feature
            names to their "base" vocabulary size (integer index). Input indices
            greater than or equal to this threshold will trigger the adapter.
            If None, attempts to default to the last category of "Protein_type".

    Returns:
        nn.Module: The initialized and attached adapter module.
    """
    # Inspect model dimensions
    cat_dims = [len(m) for m in model.cat_maps.values()]
    total_in = cast(Linear, model.base[0]).in_features
    total_emb = sum((cast(Embedding, e).embedding_dim for e in model.embeddings), 0)

    numeric_dim = total_in - total_emb

    # Initialize Adapter
    adapter = ResidualAdapter(numeric_dim, cat_dims, embed_dim=16)
    adapter.load_state_dict(adapter_state_dict)
    adapter.eval()
    original_forward = model.forward

    # If no thresholds provided, fallback to "always on" or legacy "last protein" logic
    checks = []
    if gating_thresholds:
        for feat, threshold in gating_thresholds.items():
            if feat in model.cat_feature_names:
                idx = model.cat_feature_names.index(feat)
                checks.append((idx, threshold))
    else:
        try:
            p_idx = model.cat_feature_names.index("Protein_type")
            p_thresh = len(model.cat_maps["Protein_type"]) - 1  # Only last one
            checks.append((p_idx, p_thresh))
        except ValueError:
            pass

    def new_forward(
        x_num,
        x_cat,
        return_features: bool = False,
        return_physics_details: bool = False,
    ):
        """
        Modified forward pass that conditionally applies the adapter.
        """
        if return_features or return_physics_details:
            ret = original_forward(
                x_num,
                x_cat,
                return_features=return_features,
                return_physics_details=return_physics_details,
            )
            if isinstance(ret, tuple):
                pred = ret[0]
                others = ret[1:]
            else:
                pred = ret
                others = ()
        else:
            pred = original_forward(x_num, x_cat)
            others = ()
        adapt = adapter(x_num, x_cat)

        # Dynamic Masking
        mask = torch.zeros(x_cat.size(0), 1, dtype=torch.bool, device=x_cat.device)
        for col_idx, cutoff in checks:
            is_new = x_cat[:, col_idx] >= cutoff
            mask = mask | is_new.unsqueeze(1)

        res = pred + (adapt * mask.float())

        if others:
            return (res,) + others
        return res

    model.forward = new_forward
    return adapter


def save_model_checkpoint(
    model: nn.Module,
    processor: DataProcessor,
    best_params: Dict,
    filepath: str,
    adapter: Optional[nn.Module] = None,
) -> None:
    """
    Save a comprehensive model checkpoint to disk.

    The checkpoint includes the model state, data processor state (mappings,
    scalers, split indices), hyperparameters, and optionally an adapter's state.

    Args:
        model (nn.Module): The trained model to save.
        processor (DataProcessor): The fitted data processor associated with the model.
        best_params (Dict): A dictionary of the best hyperparameters found during tuning.
        filepath (str): The file path where the checkpoint will be saved.
        adapter (Optional[nn.Module]): An optional adapter module to save alongside
            the base model. Defaults to None.

    Returns:
        None
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "best_params": best_params,
        "cat_maps": processor.cat_maps,
        "numeric_features": processor.numeric_features,
        "categorical_features": processor.categorical_features,
        "constant_values": processor.constant_values,
        "scalable_features": processor.scalable_features,
        "split_indices": processor.split_indices,
        "generated_feature_names": processor.generated_feature_names,
        "scaler_state": {
            "mean_": processor.scaler.mean_,
            "scale_": processor.scaler.scale_,
            "n_features_in_": getattr(processor.scaler, "n_features_in_", 0),
        },
    }

    if adapter is not None:
        checkpoint["adapter_state_dict"] = adapter.state_dict()
        checkpoint["has_adapter"] = True
    else:
        # Check if the model itself has a detached state dict stored
        if (
            hasattr(model, "adapter_state_dict")
            and model.adapter_state_dict is not None
        ):
            checkpoint["adapter_state_dict"] = model.adapter_state_dict
            checkpoint["has_adapter"] = True
        else:
            checkpoint["has_adapter"] = False

    torch.save(checkpoint, filepath)


def load_model_checkpoint(
    filepath: str, device: str = "cpu"
) -> Tuple[Model, DataProcessor, Dict]:
    """
    Load a model checkpoint and reconstruct the associated components.

    This function restores the `DataProcessor` (including scalers and split metadata),
    rebuilds the `Model` architecture using the saved hyperparameters, and loads
    the model weights. It handles legacy ensemble checkpoints by stripping prefixes
    if necessary.

    Args:
        filepath (str): The path to the checkpoint file.
        device (str, optional): The device to map the model weights to ('cpu' or 'cuda').
            Defaults to "cpu".

    Returns:
        Tuple[Model, DataProcessor, Dict]: A tuple containing:
            - The reconstructed and loaded `Model`.
            - The restored `DataProcessor`.
            - The dictionary of best parameters (`best_params`).
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Reconstruct processor
    processor = DataProcessor()
    processor.cat_maps = checkpoint.get("cat_maps", {})
    processor.numeric_features = checkpoint.get("numeric_features", [])
    processor.categorical_features = checkpoint.get("categorical_features", [])
    processor.constant_values = checkpoint.get("constant_values", {})
    processor.scalable_features = checkpoint.get(
        "scalable_features", processor.numeric_features.copy()
    )
    processor.constant_features = list(processor.constant_values.keys())

    # Restore Split Metadata
    processor.split_indices = checkpoint.get("split_indices", {})
    processor.generated_feature_names = checkpoint.get("generated_feature_names", [])

    processor.allow_new_categories = False
    processor.is_fitted = True

    # Rebuild scaler
    scaler_state = checkpoint.get("scaler_state", None)
    if scaler_state is not None:
        processor.scaler = StandardScaler()
        processor.scaler.mean_ = np.asarray(scaler_state["mean_"], dtype=np.float64)
        processor.scaler.scale_ = np.asarray(scaler_state["scale_"], dtype=np.float64)
        processor.scaler.n_features_in_ = int(scaler_state["n_features_in_"])
        if "feature_names_in_" in scaler_state:
            processor.scaler.feature_names_in_ = np.array(
                scaler_state["feature_names_in_"]
            )
        actual_numeric_dim = len(processor.scaler.mean_)
    else:
        processor.scaler = None
        actual_numeric_dim = len(processor.numeric_features)  # Fallback

    # Reconstruct model
    best_params = checkpoint["best_params"]
    hidden_size = best_params["hidden_size"]
    n_layers = best_params["n_layers"]
    hidden_sizes = [hidden_size] * n_layers

    model = Model(
        cat_maps=processor.cat_maps,
        numeric_dim=actual_numeric_dim,
        out_dim=len(TARGETS),
        hidden_sizes=hidden_sizes,
        dropout=best_params["dropout"],
        split_indices=processor.split_indices,
    )

    # Handle Ensemble Keys
    state_dict = checkpoint["model_state_dict"]
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("models."):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("models.0."):
                new_key = k.replace("models.0.", "")
                new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.to(device)

    # Attach adapter state dict to model for downstream use
    if "adapter_state_dict" in checkpoint:
        print("FOUND ADAPTER, RELOADING...")
        model.adapter_state_dict = checkpoint["adapter_state_dict"]
    else:
        model.adapter_state_dict = None  # type: ignore
    return model, processor, best_params
