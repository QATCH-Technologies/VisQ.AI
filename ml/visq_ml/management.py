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
    1.1 (Includes embedding dimension fix)
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
except (ImportError, ModuleNotFoundError):
    try:
        from config import TARGETS
        from layers import ResidualAdapter

        from data import DataProcessor
        from models import Model
    except (ImportError, ModuleNotFoundError):
        from visq_ml.config import TARGETS
        from visq_ml.data import DataProcessor
        from visq_ml.layers import ResidualAdapter
        from visq_ml.models import Model


def expand_processor_and_model(
    processor: DataProcessor,
    model: Model,
    feature_name: str,
    new_categories: List[str],
    initialization: str = "mean",
) -> None:
    """
    Expand both the data processor and the model embeddings for new categories.
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

    Handles dimension mismatches that occur if the model vocabulary was expanded
    without increasing embedding width (preserving transfer learning weights).
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
        actual_numeric_dim = len(processor.numeric_features)

    # Reconstruct model
    best_params = checkpoint["best_params"]
    hidden_size = best_params["hidden_size"]
    n_layers = best_params["n_layers"]
    hidden_sizes = [hidden_size] * n_layers
    dropout = best_params["dropout"]

    model = Model(
        cat_maps=processor.cat_maps,
        numeric_dim=actual_numeric_dim,
        out_dim=len(TARGETS),
        hidden_sizes=hidden_sizes,
        dropout=dropout,
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

    # --- FIX: Reconcile Embedding Dimensions ---
    # The Model constructor may initialize larger embedding dimensions due to
    # larger vocab sizes (heuristic). We must force the model to match the
    # dimensions stored in the checkpoint.

    total_emb_dim = 0
    dims_changed = False

    # 1. Fix Embedding Layers
    for i, emb_layer in enumerate(model.embeddings):
        key = f"embeddings.{i}.weight"
        if key in state_dict:
            saved_weight = state_dict[key]
            saved_vocab, saved_dim = saved_weight.shape

            # If width differs, replace the layer
            if emb_layer.embedding_dim != saved_dim:
                # We trust the saved_vocab size (rows) matches processor.cat_maps
                model.embeddings[i] = nn.Embedding(saved_vocab, saved_dim)
                dims_changed = True

            total_emb_dim += saved_dim
        else:
            total_emb_dim += emb_layer.embedding_dim

    # 2. Fix Base Network (MLP) Input Layer
    # If embedding dimensions changed, the concatenated input size to the MLP changed.
    if dims_changed:
        input_dim = total_emb_dim + actual_numeric_dim

        # Verify against checkpoint to be sure
        base_key = "base.0.weight"
        if base_key in state_dict:
            saved_in_features = state_dict[base_key].shape[1]

            # Reconstruct the MLP if input features mismatch
            if saved_in_features != cast(Linear, model.base[0]).in_features:
                layers = []
                prev = saved_in_features  # Use the checkpoint's truth
                for h in hidden_sizes:
                    layers.append(nn.Linear(prev, h))
                    layers.append(nn.LayerNorm(h))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    prev = h
                model.base = nn.Sequential(*layers)

    model.load_state_dict(state_dict)
    model.to(device)

    # Attach adapter state dict to model for downstream use
    if "adapter_state_dict" in checkpoint:
        # print("FOUND ADAPTER, RELOADING...")
        model.adapter_state_dict = checkpoint["adapter_state_dict"]
    else:
        model.adapter_state_dict = None  # type: ignore
    return model, processor, best_params
