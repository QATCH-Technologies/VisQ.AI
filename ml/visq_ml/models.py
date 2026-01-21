"""
Models module for the VisQ.AI.

This module defines the primary neural network architecture (`Model`) and its
ensemble wrapper (`EnsembleModel`). The architecture is designed for tabular
data with a specific focus on incorporating physics-informed priors for
excipient effects in protein formulation.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-01-15

Version:
    1.1 (Physics Layer Expansion Support)
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
    from visq_ml.config import (
        CONC_THRESHOLDS,
        CONC_TYPE_PAIRS,
        EXCIPIENT_PRIORS,
        EXCIPIENT_TYPE_MAPPING,
    )
    from visq_ml.layers import (
        EmbeddingDropout,
        LearnablePhysicsPrior,
        LearnableSoftThresholdPrior,
        ResidualBlock,
    )


class Model(nn.Module):
    """
    Main Neural Network Architecture for Viscosity Prediction.
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
            if (
                self.p_class_idx != -1
                and col in EXCIPIENT_TYPE_MAPPING
                and col in CONC_TYPE_PAIRS.values()
            ):
                conc_col_name = [k for k, v in CONC_TYPE_PAIRS.items() if v == col][0]

                if conc_col_name in self.split_indices:
                    n_classes = len(self.cat_maps["Protein_class_type"])
                    n_regimes = len(self.cat_maps["Regime"])
                    n_excipients = len(self.cat_maps[col])

                    # Build Threshold Tensor
                    init_thresh = torch.ones(n_excipients)
                    vocab_list = self.cat_maps[col]

                    for idx, name in enumerate(vocab_list):
                        name_lower = name.lower()
                        for key, val in CONC_THRESHOLDS.items():
                            if key in name_lower:
                                init_thresh[idx] = float(val)
                                break

                    phys_layer = LearnableSoftThresholdPrior(
                        n_classes,
                        n_regimes,
                        n_excipients,
                        initial_thresholds=init_thresh,
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
        p_classes = self.cat_maps["Protein_class_type"]
        regimes = self.cat_maps["Regime"]
        excipients = self.cat_maps[col_name]

        tensor = cast(Tensor, layer.static_scores)
        with torch.no_grad():
            for p_i, p_val in enumerate(p_classes):
                for r_i, r_val in enumerate(regimes):
                    for e_i, e_val in enumerate(excipients):
                        p_key = p_val.lower().strip()
                        r_key = r_val.lower().strip()
                        e_key = e_val.lower().strip()
                        prior_dict = None

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
                    idx_start, idx_end = self.split_indices[conc_name]
                    val_raw = x_num[:, idx_start:idx_end]

                    # Ignore "none" categories
                    none_idx = self.none_indices.get(col_name, -1)
                    if none_idx != -1:
                        mask = (e_idx != none_idx).float().unsqueeze(1)
                    else:
                        mask = torch.ones_like(e_idx).float().unsqueeze(1)
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
        Expand an embedding layer AND Physics Priors to accommodate new categories.
        """
        if feature_name not in self.cat_feature_names:
            raise ValueError(f"Feature {feature_name} not in categorical features")

        idx = self.cat_feature_names.index(feature_name)
        old_emb = cast(Embedding, self.embeddings[idx])
        old_vocab_size = len(self.cat_maps[feature_name])
        num_new = len(new_categories)
        new_vocab_size = old_vocab_size + num_new

        # 1. Resize Embedding Layer
        new_emb = nn.Embedding(new_vocab_size, old_emb.embedding_dim)
        with torch.no_grad():
            new_emb.weight[:old_vocab_size] = old_emb.weight
            if initialization == "mean":
                mean_embedding = old_emb.weight.mean(dim=0)
                # Add slight noise to break symmetry
                noise = (
                    torch.randn(
                        num_new, old_emb.embedding_dim, device=old_emb.weight.device
                    )
                    * 0.01
                )
                new_emb.weight[old_vocab_size:] = mean_embedding + noise
            elif initialization == "zero":
                new_emb.weight[old_vocab_size:] = 0.0

            new_emb.to(old_emb.weight.device)

        self.embeddings[idx] = new_emb
        self.cat_maps[feature_name].extend(new_categories)

        # 2. Resize Physics Layers (if applicable)
        # Physics tensors are shaped: [Protein, Regime, Excipient]

        is_p_class = feature_name == "Protein_class_type"
        is_regime = feature_name == "Regime"

        if is_p_class or is_regime:
            # If Protein (dim 0) or Regime (dim 1) changes, ALL physics layers must resize
            expand_dim = 0 if is_p_class else 1
            for layer in self.physics_layers:
                if hasattr(layer, "expand_indices"):
                    layer.expand_indices(expand_dim, num_new)

        elif idx < len(self.physics_layers):
            # If an Excipient changes, only THAT specific layer resizes (dim 2)
            layer = self.physics_layers[idx]
            if hasattr(layer, "expand_indices"):
                layer.expand_indices(2, num_new)


class EnsembleModel(nn.Module):
    """
    Ensemble Wrapper for managing multiple Model instances.
    """

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    @property
    def cat_maps(self) -> Dict:
        return cast(Model, self.models[0]).cat_maps

    @property
    def cat_feature_names(self) -> List[str]:
        return cast(Model, self.models[0]).cat_feature_names

    @property
    def embeddings(self) -> nn.ModuleList:
        return cast(Model, self.models[0]).embeddings

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        outputs = [model(x_num, x_cat) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

    def get_individual_predictions(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = [model(x_num, x_cat) for model in self.models]
        return torch.stack(outputs)

    def expand_categorical_embedding(
        self,
        feature_name: str,
        new_categories: List[str],
        initialization: str = "mean",
    ) -> None:
        """
        Expand embeddings on all models in the ensemble.
        """
        for model in self.models:
            cast(Model, model).expand_categorical_embedding(
                feature_name, new_categories, initialization
            )
