"""
Models module for the VisQ.AI.

This module defines the primary neural network architecture (`Model`) and its
ensemble wrapper (`EnsembleModel`). The architecture is designed for tabular
data with a specific focus on incorporating physics-informed priors for
excipient effects in protein formulation.

Key Components:
    - **Model**: A deep learning model combining categorical embeddings,
      residual blocks, and a custom learnable physics prior layer.
    - **EnsembleModel**: A container for managing multiple `Model` instances
      to provide uncertainty quantification and robust predictions.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-01-15

Version:
    1.0
"""

from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding

try:
    from .config import (
        CONC_THRESHOLDS,
        CONC_TYPE_PAIRS,
        EXCIPIENT_PRIORS,
        EXCIPIENT_TYPE_MAPPING,
    )
    from .layers import (
        EmbeddingDropout,
        LearnablePhysicsPrior,
        LearnableSoftThresholdPrior,
        ResidualBlock,
    )
except ImportError:
    from config import (
        CONC_THRESHOLDS,
        CONC_TYPE_PAIRS,
        EXCIPIENT_PRIORS,
        EXCIPIENT_TYPE_MAPPING,
    )
    from layers import (
        EmbeddingDropout,
        LearnablePhysicsPrior,
        LearnableSoftThresholdPrior,
        ResidualBlock,
    )


class Model(nn.Module):
    """
    Main Neural Network Architecture for Viscosity Prediction.

    This model integrates standard deep learning components (embeddings, MLPs)
    with domain-specific physics priors. It features a mechanism to "correct"
    the baseline neural network prediction using known physical relationships
    between excipients and protein viscosity.

    Attributes:
        adapter_state_dict (Optional[Dict[str, Any]]): Placeholder for storing
            state dictionaries of adapters (e.g., for fine-tuning or transfer learning).
        cat_feature_names (List[str]): List of categorical feature names.
        cat_maps (Dict[str, List[str]]): Dictionary mapping feature names to their
            list of categories (vocabulary).
        split_indices (Dict[str, Tuple[int, int]]): Dictionary mapping concentration
            feature names to their start/end indices in the numerical input tensor.
        none_indices (Dict[str, int]): Map of column names to the integer index
            representing "missing" or "none" values (e.g., "nan", "n/a").
        p_class_idx (int): Column index for the 'Protein_class_type' feature.
        regime_idx (int): Column index for the 'Regime' feature.
        emb_drop (EmbeddingDropout): Custom dropout layer for embeddings.
        embeddings (nn.ModuleList): List of embedding layers for categorical features.
        physics_layers (nn.ModuleList): List of physics prior layers (or Identity
            layers where not applicable).
        physics_col_indices (List[int]): Mapping to track which columns have
            active physics layers.
        regime_gate (Optional[nn.Embedding]): An optional gating mechanism that
            scales the final output based on the 'Regime'.
        residual_blocks (nn.ModuleList): A sequence of residual blocks for deep
            feature extraction.
        base (nn.Sequential): The initial fully connected network (MLP) processing
            concatenated embeddings and numerical inputs.
        heads (nn.ModuleList): Parallel output heads for multi-target prediction.
    """

    NO_ENTRIES = ["none", "nan", "null", "n/a"]
    adapter_state_dict: Optional[Dict[str, Any]]

    def __init__(
        self,
        cat_maps: Dict,
        numeric_dim: int,
        out_dim: int,
        hidden_sizes: List[int],
        dropout: float,
        split_indices: Dict[str, Tuple[int, int]],
    ):
        """
        Initialize the Model.

        Args:
            cat_maps (Dict): Dictionary where keys are feature names and values are
                lists of categories (vocabularies).
            numeric_dim (int): The number of numerical input features.
            out_dim (int): The number of output targets.
            hidden_sizes (List[int]): A list of integers defining the size of each
                hidden layer in the base network and residual blocks.
            dropout (float): The dropout probability.
            split_indices (Dict[str, Tuple[int, int]]): Mapping for numerical feature
                slicing, specifically for concentration values needed by physics layers.
        """
        super().__init__()
        self.adapter_state_dict = None
        self.cat_feature_names = list(cat_maps.keys())
        self.cat_maps = {k: list(v) for k, v in cat_maps.items()}
        self.split_indices = split_indices or {}

        # Store indices of "none" for robust masking in physics layers
        self.none_indices = {}
        for col, categories in self.cat_maps.items():
            none_idx = -1
            for candidate in self.NO_ENTRIES:
                if candidate in categories:
                    none_idx = categories.index(candidate)
                    break
            self.none_indices[col] = none_idx

        # Indices for specific logic
        self.p_class_idx = (
            self.cat_feature_names.index("Protein_class_type")
            if "Protein_class_type" in self.cat_feature_names
            else -1
        )
        self.regime_idx = (
            self.cat_feature_names.index("Regime")
            if "Regime" in self.cat_feature_names
            else -1
        )

        self.emb_drop = EmbeddingDropout(0.05)

        # Embeddings & Physics Layers
        self.embeddings = nn.ModuleList()
        self.physics_layers = nn.ModuleList()
        self.physics_col_indices = []

        for i, col in enumerate(self.cat_feature_names):
            vocab = len(self.cat_maps[col])
            # Heuristic for embedding dimension: roughly 2 * sqrt(vocab_size), clamped between 4 and 32
            emb_dim = min(32, max(4, int(vocab**0.5 * 2)))
            self.embeddings.append(nn.Embedding(vocab, emb_dim))

            # Initialize Physics Layers if applicable
            # Conditions: Feature is a Protein Class, matches an Excipient Type, and has a paired concentration
            if (
                self.p_class_idx != -1
                and col in EXCIPIENT_TYPE_MAPPING
                and col in CONC_TYPE_PAIRS.values()
            ):
                conc_col_name = [k for k, v in CONC_TYPE_PAIRS.items() if v == col][0]

                # We now assume split_indices points to the RAW concentration column
                if conc_col_name in self.split_indices:
                    n_classes = len(self.cat_maps["Protein_class_type"])
                    n_regimes = len(self.cat_maps["Regime"])
                    n_excipients = len(self.cat_maps[col])

                    # --- NEW: Build Threshold Tensor ---
                    # default to 1.0 to avoid division by zero if missing
                    init_thresh = torch.ones(n_excipients)
                    vocab = self.cat_maps[col]

                    for idx, name in enumerate(vocab):
                        # Simple fuzzy matching to find threshold in config
                        # e.g. "sucrose" in "sucrose_experimental" -> 200.0
                        name_lower = name.lower()
                        for key, val in CONC_THRESHOLDS.items():
                            if key in name_lower:
                                init_thresh[idx] = float(val)
                                break
                    # -----------------------------------

                    # Instantiate the NEW layer class (Update class name if needed)
                    phys_layer = LearnableSoftThresholdPrior(
                        n_classes,
                        n_regimes,
                        n_excipients,
                        initial_thresholds=init_thresh,  # <--- Pass it here
                    )

                    self._init_static_priors(phys_layer, col)
                    self.physics_layers.append(phys_layer)
                    self.physics_col_indices.append(i)
                else:
                    self.physics_layers.append(nn.Identity())
                    self.physics_col_indices.append(-1)
            else:
                self.physics_layers.append(nn.Identity())
                self.physics_col_indices.append(-1)

        # Calculate total embedding dimension for input layer sizing
        # Fixes type ambiguity for static analysis tools
        total_emb = sum((cast(Embedding, e).embedding_dim for e in self.embeddings), 0)

        # Regime Gate
        self.regime_gate = None
        if self.regime_idx != -1:
            num_regimes = len(self.cat_maps["Regime"])
            self.regime_gate = nn.Embedding(num_regimes, 1)
            nn.init.constant_(self.regime_gate.weight, 1.0)

        # Residual Blocks
        self.residual_blocks = nn.ModuleList()
        for h in hidden_sizes:
            self.residual_blocks.append(ResidualBlock(h, dropout))

        # Base Network (MLP)
        layers = []
        prev = total_emb + numeric_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.base = nn.Sequential(*layers)

        # Output Heads (Multi-task support)
        self.heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(out_dim)])

    def _init_static_priors(self, layer: LearnablePhysicsPrior, col_name: str) -> None:
        """
        Initializes the static prior tensor in a physics layer.

        Populates the learnable physics layer with initial values derived from
        `EXCIPIENT_PRIORS`. It maps specific vocabulary items (e.g., 'tween-20')
        to generic roles defined in the configuration (e.g., 'tween') to set
        plausible initial biases for excipient effects.

        Args:
            layer (LearnablePhysicsPrior): The physics layer to initialize.
            col_name (str): The name of the categorical column (excipient type)
                associated with this layer.
        """
        p_classes = self.cat_maps["Protein_class_type"]
        regimes = self.cat_maps["Regime"]
        excipients = self.cat_maps[col_name]

        tensor = layer.static_scores
        tensor = cast(Tensor, layer.static_scores)
        with torch.no_grad():
            for p_i, p_val in enumerate(p_classes):
                for r_i, r_val in enumerate(regimes):
                    for e_i, e_val in enumerate(excipients):
                        p_key = p_val.lower().strip()
                        r_key = r_val.lower().strip()
                        e_key = e_val.lower().strip()
                        prior_dict = None

                        # Helper to find tuple key (class, regime) case-insensitively
                        def find_prior_dict(pk, rk):
                            if (pk, rk) in EXCIPIENT_PRIORS:
                                return EXCIPIENT_PRIORS[(pk, rk)]
                            for (dict_p, dict_r), val in EXCIPIENT_PRIORS.items():
                                if (
                                    str(dict_p).lower().strip() == pk
                                    and str(dict_r).lower().strip() == rk
                                ):
                                    return val
                            return None

                        prior_dict = find_prior_dict(p_key, r_key)
                        if prior_dict is None and (
                            p_key in ["noprotein", "none"] or r_key == "noprotein"
                        ):
                            prior_dict = find_prior_dict("noprotein", "noprotein")
                        if prior_dict is None:
                            prior_dict = find_prior_dict("other", r_key)

                        if prior_dict is None:
                            continue
                        effect = 0.0
                        target_key = None

                        # Map specific excipients to general categories
                        if "nacl" in e_key:
                            target_key = "nacl"
                        elif "arginine" in e_key:
                            target_key = "arginine"
                        elif "lysine" in e_key:
                            target_key = "lysine"
                        elif "proline" in e_key:
                            target_key = "proline"
                        elif "sucrose" in e_key or "trehalose" in e_key:
                            target_key = "stabilizer"
                        elif "tween-20" in e_key:
                            target_key = "tween20"
                        elif "tween-80" in e_key:
                            target_key = "tween80"

                        if target_key:
                            if target_key in prior_dict:
                                effect = float(prior_dict[target_key])
                            else:
                                for k, v in prior_dict.items():
                                    if str(k).lower().strip() == target_key:
                                        effect = float(v)
                                        break

                        tensor[p_i, r_i, e_i] = effect

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        return_features: bool = False,
        return_physics_details: bool = False,
    ):
        """
        Forward pass of the model.

        The flow includes:
        1.  Embedding categorical features.
        2.  Concatenating embeddings with numerical features.
        3.  Passing through the Base MLP and Residual Blocks.
        4.  Generating initial predictions via output heads.
        5.  Calculating and adding physics-based corrections (if applicable).
        6.  Applying regime-based gating (if applicable).

        Args:
            x_num (torch.Tensor): Numerical input features.
            x_cat (torch.Tensor): Categorical input features (integer encoded).
            return_features (bool, optional): If True, returns the penultimate
                feature vector. Defaults to False.
            return_physics_details (bool, optional): If True, returns a dictionary
                containing breakdown of physics corrections. Defaults to False.

        Returns:
            torch.Tensor or Tuple: The prediction tensor. If `return_features` or
            `return_physics_details` is True, returns a tuple containing the
            prediction and the requested extra data.
        """
        # Embeddings & Base Network
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        emb_cat = torch.cat(embs, dim=1)
        emb_cat = self.emb_drop(emb_cat)
        x = torch.cat([emb_cat, x_num], dim=1)
        x = self.base(x)

        if self.residual_blocks:
            for block in self.residual_blocks:
                x = block(x)

        if return_features:
            features = x.clone()

        outputs = [head(x) for head in self.heads]
        pred = torch.cat(outputs, dim=1)

        # Learnable Physics Correction
        physics_details = {}

        if self.p_class_idx != -1:
            p_idx = x_cat[:, self.p_class_idx]
            r_idx = x_cat[:, self.regime_idx]
            total_correction = 0.0

            for i, phys_layer in enumerate(self.physics_layers):
                if not isinstance(phys_layer, nn.Identity):
                    e_idx = x_cat[:, i]
                    col_name = self.cat_feature_names[i]
                    conc_name = [
                        k for k, v in CONC_TYPE_PAIRS.items() if v == col_name
                    ][0]

                    # Get indices for the raw concentration
                    # NOTE: Ensure split_indices now points to the single raw column!
                    idx_start, idx_end = self.split_indices[conc_name]

                    # --- NEW: Fetch single raw value ---
                    val_raw = x_num[:, idx_start:idx_end]
                    # -----------------------------------

                    # Ignore "none" categories
                    none_idx = self.none_indices.get(col_name, -1)
                    if none_idx != -1:
                        mask = (e_idx != none_idx).float().unsqueeze(1)
                    else:
                        mask = torch.ones_like(e_idx).float().unsqueeze(1)

                    # --- NEW: Updated Call Signature ---
                    # We pass 'val_raw' instead of 'val_low, val_high'
                    correction, layer_details = phys_layer(p_idx, r_idx, e_idx, val_raw)

                    # Ensure correction is [Batch, 1]
                    if correction.dim() == 1:
                        correction = correction.unsqueeze(1)

                    total_correction = total_correction + (correction * mask)

                    if return_physics_details:
                        physics_details[col_name] = {
                            k: v * mask.squeeze() for k, v in layer_details.items()
                        }
            pred = pred + total_correction

        # Regime Gate
        if self.regime_gate is not None and self.regime_idx != -1:
            gate = self.regime_gate(x_cat[:, self.regime_idx])
            gate = torch.sigmoid(gate)
            pred = pred * gate

        ret = [pred]
        if return_features:
            ret.append(features)
        if return_physics_details:
            ret.append(physics_details)

        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

    def expand_categorical_embedding(
        self,
        feature_name: str,
        new_categories: List[str],
        initialization: str = "mean",
    ) -> None:
        """
        Expand an embedding layer to accommodate new categories.

        Dynamically resizes a specific embedding layer to include new vocabulary
        items. Weights for new items can be initialized via the mean of existing
        embeddings (with noise) or zeros.

        Args:
            feature_name (str): The name of the feature to expand.
            new_categories (List[str]): List of new category labels.
            initialization (str, optional): Initialization method ("mean" or "zero").
                Defaults to "mean".

        Raises:
            ValueError: If `feature_name` is not in the model.
            RuntimeWarning: If the feature is attached to a Physics Prior layer,
                which cannot currently be resized dynamically.
        """
        if feature_name not in self.cat_feature_names:
            raise ValueError(f"Feature {feature_name} not in categorical features")

        idx = self.cat_feature_names.index(feature_name)
        # Cast to Embedding to resolve type ambiguity
        old_emb = cast(Embedding, self.embeddings[idx])
        old_vocab_size = len(self.cat_maps[feature_name])
        new_vocab_size = old_vocab_size + len(new_categories)
        new_emb = nn.Embedding(new_vocab_size, old_emb.embedding_dim)
        with torch.no_grad():
            # Copy existing knowledge
            new_emb.weight[:old_vocab_size] = old_emb.weight
            if initialization == "mean":
                mean_embedding = old_emb.weight.mean(dim=0)
                for i in range(len(new_categories)):
                    noise = torch.randn_like(mean_embedding) * 0.01
                    new_emb.weight[old_vocab_size + i] = mean_embedding + noise
            elif initialization == "zero":
                new_emb.weight[old_vocab_size:] = 0.0

        # Replace the layer
        self.embeddings[idx] = new_emb

        # Update internal map
        self.cat_maps[feature_name].extend(new_categories)

        # Check Physics Layer Compatibility
        # Using isinstance check instead of 'is not None' for compatibility with nn.Identity
        if not isinstance(self.physics_layers[idx], nn.Identity):
            raise RuntimeWarning(
                f"Expanded '{feature_name}' which has a Physics Prior attached. "
                f"The new categories {new_categories} will have 0 physics effect "
                "until the physics tensor is manually resized."
            )


class EnsembleModel(nn.Module):
    """
    Ensemble Wrapper for managing multiple Model instances.

    This class wraps a list of `Model` instances and provides a unified interface
    for training and inference. It averages predictions during standard forward
    passes and exposes individual predictions for uncertainty estimation.

    Attributes:
        models (nn.ModuleList): The list of individual `Model` instances.
    """

    def __init__(self, models: List[nn.Module]):
        """
        Initialize the EnsembleModel.

        Args:
            models (List[nn.Module]): A list of instantiated `Model` objects.
        """
        super().__init__()
        self.models = nn.ModuleList(models)

    @property
    def cat_maps(self) -> Dict:
        """
        Proxy to the first model's `cat_maps` for consistent interface.

        Returns:
            Dict: The category mapping dictionary.
        """
        return cast(Model, self.models[0]).cat_maps

    @property
    def cat_feature_names(self) -> List[str]:
        """
        Proxy to the first model's `cat_feature_names` for consistent interface.

        Returns:
            List[str]: The list of categorical feature names.
        """
        return cast(Model, self.models[0]).cat_feature_names

    @property
    def embeddings(self) -> nn.ModuleList:
        """
        Proxy to the first model's `embeddings` for consistent interface.

        Returns:
            nn.ModuleList: The embedding layers of the first model.
        """
        return cast(Model, self.models[0]).embeddings

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning the mean prediction of the ensemble.

        Args:
            x_num (torch.Tensor): Numerical input features.
            x_cat (torch.Tensor): Categorical input features.

        Returns:
            torch.Tensor: The averaged prediction tensor across all models.
        """
        outputs = [model(x_num, x_cat) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

    def get_individual_predictions(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> torch.Tensor:
        """
        Get predictions from each model individually.

        Useful for calculating variance or confidence intervals (uncertainty quantification).

        Args:
            x_num (torch.Tensor): Numerical input features.
            x_cat (torch.Tensor): Categorical input features.

        Returns:
            torch.Tensor: A tensor of shape (n_models, batch_size, n_targets).
        """
        with torch.no_grad():
            outputs = [model(x_num, x_cat) for model in self.models]
        return torch.stack(outputs)  # (n_models, batch_size, n_targets)

    def expand_categorical_embedding(
        self,
        feature_name: str,
        new_categories: List[str],
        initialization: str = "mean",
    ) -> None:
        """
        Expand embeddings on all models in the ensemble.

        Iterates through every model in the ensemble and applies the embedding
        expansion logic.

        Args:
            feature_name (str): The name of the feature to expand.
            new_categories (List[str]): List of new category labels.
            initialization (str, optional): Initialization method. Defaults to "mean".
        """
        for model in self.models:
            cast(Model, model).expand_categorical_embedding(
                feature_name, new_categories, initialization
            )
