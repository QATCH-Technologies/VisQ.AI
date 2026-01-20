"""Unified Inference and Adaptive Learning Module for VisQAI.

This module provides the `ViscosityPredictor` class, which serves as the central
interface for:
1.  **Inference**: Loading checkpoints (single or ensemble) and generating predictions.
2.  **Uncertainty Quantification**: estimating prediction confidence intervals using
    Monte Carlo Dropout.
3.  **Adaptive Learning**: Updating the model on new data via vocabulary expansion
    and Gated Residual Adapters.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-01-15

Version:
    1.0
"""

import enum
import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config import TARGETS
    from .layers import LearnablePhysicsPrior, ResidualAdapter
    from .management import (
        expand_processor_and_model,
        load_model_checkpoint,
        save_model_checkpoint,
    )
    from .models import EnsembleModel, Model
    from .utils import (
        inverse_log_transform,
        log_transform_targets,
        to_tensors,
    )
except ImportError:
    from config import TARGETS
    from layers import LearnablePhysicsPrior, ResidualAdapter
    from management import (
        expand_processor_and_model,
        load_model_checkpoint,
        save_model_checkpoint,
    )
    from utils import (
        inverse_log_transform,
        log_transform_targets,
        to_tensors,
    )

    from models import EnsembleModel, Model


class ViscosityPredictor:
    """Manages model lifecycle, inference, and online adaptation.

    This class handles the complexity of loading models (single or ensemble),
    processing data, and applying post-training adaptation techniques. It supports
    lazy loading to minimize resource usage until necessary.

    Attributes:
        device (str): Computation device ('cpu' or 'cuda').
        is_ensemble (bool): Whether the predictor manages an ensemble of models.
        adapter (Optional[nn.Module]): The trained ResidualAdapter, if adaptation has occurred.
        model (Optional[Union[Model, EnsembleModel]]): The underlying PyTorch model(s).
        processor (Optional[DataProcessor]): The data processing pipeline coupled with the model.
        best_params (Optional[Dict]): Hyperparameters associated with the loaded model.
        base_vocab_sizes (Dict[str, int]): Snapshot of vocabulary sizes at load time. Used
            for gating logic during adaptation.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, List[str]],
        device: str = "cpu",
        is_ensemble: bool = False,
    ):
        """Initializes the predictor configuration.

        Args:
            checkpoint_path (Union[str, List[str]]): Path to a `.pt` checkpoint file,
                a directory containing checkpoints (if ensemble), or a list of file paths.
            device (str, optional): Device to run inference on. Defaults to "cpu".
            is_ensemble (bool, optional): If True, treats `checkpoint_path` as a source
                for multiple models to be averaged. Defaults to False.
        """
        self.device = device
        self.is_ensemble = is_ensemble
        self.adapter = None
        self._checkpoint_path = checkpoint_path
        self._hydrated = False
        self.base_vocab_sizes: Dict[str, int] = {}

        self.model = None
        self.processor = None
        self.best_params = None
        self.n_models = None

    def hydrate(self) -> "ViscosityPredictor":
        """Lazily loads the model and processor from disk.

        If the model is already loaded, this is a no-op.

        Returns:
            ViscosityPredictor: Self, for method chaining.
        """
        if self._hydrated:
            return self

        if self.is_ensemble:
            self._load_ensemble(self._checkpoint_path)
        else:
            if isinstance(self._checkpoint_path, list):
                raise ValueError("Single model mode requires a string path.")
            self._load_single_model(self._checkpoint_path)

        if self.processor is None:
            raise RuntimeError("Failed to load processor from checkpoint.")

        self.base_vocab_sizes = {k: len(v) for k, v in self.processor.cat_maps.items()}
        if self.model is None:
            raise RuntimeError("Failed to load model from checkpoint.")

        self.model.eval()
        self._hydrated = True
        return self

    def _ensure_hydrated(self):
        """Internal check to ensure model is loaded before operations."""
        if not self._hydrated:
            self.hydrate()

    def _load_single_model(self, checkpoint_path: str):
        """Loads a single model checkpoint and refreshes physics priors.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        self.checkpoint_path = checkpoint_path
        self.model, self.processor, self.best_params = load_model_checkpoint(
            checkpoint_path, device=self.device
        )
        self.n_models = 1
        if hasattr(self.model, "physics_layers"):
            for i, layer in enumerate(self.model.physics_layers):
                if isinstance(layer, LearnablePhysicsPrior):
                    col_name = self.model.cat_feature_names[i]
                    self.model._init_static_priors(layer, col_name)

    def _load_ensemble(self, checkpoint_paths: Union[str, List[str]]):
        """Loads multiple models for ensemble inference.

        Args:
            checkpoint_paths (Union[str, List[str]]): List of paths or a directory/glob pattern.

        Raises:
            ValueError: If no valid checkpoint files are found.
        """
        if isinstance(checkpoint_paths, str):
            if "*" in checkpoint_paths or Path(checkpoint_paths).is_dir():
                pattern = (
                    str(Path(checkpoint_paths) / "*.pt")
                    if Path(checkpoint_paths).is_dir()
                    else checkpoint_paths
                )
                checkpoint_list = sorted(glob.glob(pattern))
            else:
                checkpoint_list = [checkpoint_paths]
        else:
            checkpoint_list = checkpoint_paths

        if not checkpoint_list:
            raise ValueError("No checkpoint files provided for ensemble")
        self.model, self.processor, self.best_params = load_model_checkpoint(
            checkpoint_list[0], device=self.device
        )
        models: List[nn.Module] = [self.model]
        for ckpt in checkpoint_list[1:]:
            m, _, _ = load_model_checkpoint(ckpt, device=self.device)
            models.append(m)

        self.model = EnsembleModel(models)
        self.n_models = len(models)

    def learn(
        self,
        df_new: pd.DataFrame,
        y_new: np.ndarray,
        epochs: int = 500,
        lr: float = 5e-3,
        analog_protein: Optional[str] = None,
    ):
        """Adapts the model to new data using transfer learning.

        This process involves:
        1.  Detecting and expanding categorical vocabulary (e.g., new proteins).
        2.  Initializing new embeddings (optionally using an analog).
        3.  Training a Gated Residual Adapter to correct predictions for the new domain.

        Args:
            df_new (pd.DataFrame): The new input features.
            y_new (np.ndarray): The corresponding target values.
            epochs (int, optional): Number of training epochs for the adapter. Defaults to 500.
            lr (float, optional): Learning rate for the adapter. Defaults to 5e-3.
            analog_protein (Optional[str], optional): If a new protein is detected,
                initialize its embedding based on this existing protein's embedding.
        """
        self._ensure_hydrated()
        if self.processor is None:
            raise RuntimeError("Processor not initialized despite hydration.")

        new_cats = self.processor.detect_new_categories(df_new)
        expanded_any = False
        for feature, categories in new_cats.items():
            for cat in categories:
                sim_cat = analog_protein if feature == "Protein_type" else None
                self._smart_expand_category(feature, cat, similar_category=sim_cat)
                expanded_any = True

        if not expanded_any:
            self._train_gated_adapter(df_new, y_new, n_epochs=epochs, lr=lr)

    def _smart_expand_category(
        self,
        feature_name: str,
        new_category: str,
        similar_category: Optional[str] = None,
    ):
        """Expands embedding layers to accommodate a new category.

        Initializes the new embedding vector. If `similar_category` is provided,
        initializes with that vector + noise. Otherwise, uses the mean of all
        existing embeddings.

        Args:
            feature_name (str): The name of the categorical feature (e.g., "Protein_type").
            new_category (str): The new label to add.
            similar_category (Optional[str], optional): An existing label to copy weights from.
        """
        if self.processor is None or self.model is None:
            raise RuntimeError(
                "Model and Processor must be initialized before expansion."
            )
        if feature_name not in self.processor.categorical_features:
            return
        if new_category in self.processor.cat_maps[feature_name]:
            return

        self.processor.add_categories(feature_name, [new_category])
        idx = self.model.cat_feature_names.index(feature_name)
        if hasattr(self.model, "cat_feature_names"):
            idx = self.model.cat_feature_names.index(feature_name)
        else:
            return
        module = self.model.embeddings[idx]
        if not isinstance(module, nn.Embedding):
            raise TypeError(f"Expected nn.Embedding at index {idx}, got {type(module)}")
        old_emb = module
        new_vocab_size = old_emb.num_embeddings + 1
        new_emb_layer = nn.Embedding(new_vocab_size, old_emb.embedding_dim)

        with torch.no_grad():
            new_emb_layer.weight[:-1] = old_emb.weight
            if (
                similar_category
                and similar_category in self.processor.cat_maps[feature_name]
            ):
                try:
                    sim_idx = self.processor.cat_maps[feature_name].index(
                        similar_category
                    )
                    noise = torch.randn(old_emb.embedding_dim) * 0.01
                    new_emb_layer.weight[-1] = (old_emb.weight[sim_idx] * 1.5) + noise
                except ValueError:
                    new_emb_layer.weight[-1] = old_emb.weight.mean(dim=0)
            else:
                new_emb_layer.weight[-1] = old_emb.weight.mean(dim=0)

        self.model.embeddings[idx] = new_emb_layer.to(self.device)
        self.model.cat_maps[feature_name].append(new_category)

    def _train_gated_adapter(
        self, df_new: pd.DataFrame, y_new: np.ndarray, n_epochs: int, lr: float
    ):
        """Trains the Residual Adapter on prediction residuals.

        Args:
            df_new (pd.DataFrame): Input features.
            y_new (np.ndarray): Target values (linear space).
            n_epochs (int): Training epochs.
            lr (float): Learning rate.
        """
        if self.processor is None or self.model is None:
            raise RuntimeError("Processor and Model must be initialized.")
        X_num, X_cat = self.processor.transform(df_new)
        cat_dims = [len(m) for m in self.processor.cat_maps.values()]
        num_dim = X_num.shape[1]

        self.adapter = ResidualAdapter(num_dim, cat_dims, embed_dim=16).to(self.device)
        optimizer = optim.Adam(self.adapter.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        y_log = log_transform_targets(y_new)
        y_log_t = torch.tensor(y_log, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            X_num_t, X_cat_t = to_tensors(X_num, X_cat)
            X_num_t = X_num_t.to(self.device)
            X_cat_t = X_cat_t.to(self.device)
            # Base model prediction
            base_preds = self.model(X_num_t, X_cat_t)
        target_residuals = y_log_t - base_preds
        self.adapter.train()

        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred_residuals = self.adapter(X_num_t, X_cat_t)
            loss = loss_fn(pred_residuals, target_residuals)
            loss.backward()
            optimizer.step()

        self._attach_gated_adapter()

    def hydrate_adapter(
        self, df_support: pd.DataFrame, n_epochs: int = 500, lr: float = 5e-3
    ):
        """Restores or trains an adapter state from a support set.

        Useful for few-shot scenarios where we want to re-learn an adapter from
        a small set of examples without fully re-training.

        Args:
            df_support (pd.DataFrame): Support set features.
            n_epochs (int, optional): Training epochs. Defaults to 500.
            lr (float, optional): Learning rate. Defaults to 5e-3.
        """
        self._ensure_hydrated()

        if self.processor is None or self.model is None:
            raise RuntimeError("Processor and Model must be initialized.")

        X_num, X_cat = self.processor.transform(df_support)
        y_true = df_support[TARGETS].values
        y_log = log_transform_targets(y_true)

        # Setup Architecture
        cat_dims = [len(m) for m in self.processor.cat_maps.values()]
        num_dim = X_num.shape[1]

        self.adapter = ResidualAdapter(num_dim, cat_dims, embed_dim=16).to(self.device)

        with torch.no_grad():
            X_num_t, X_cat_t, y_log_t = to_tensors(X_num, X_cat, y_log)
            X_num_t = X_num_t.to(self.device)
            X_cat_t = X_cat_t.to(self.device)
            y_log_t = y_log_t.to(self.device)
            base_preds = self.model(X_num_t, X_cat_t)

        target_residuals = y_log_t - base_preds

        optimizer = optim.Adam(self.adapter.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.adapter.train()
        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred = self.adapter(X_num_t, X_cat_t)
            loss = loss_fn(pred, target_residuals)
            loss.backward()
            optimizer.step()

        self._attach_gated_adapter()

    def _attach_gated_adapter(self):
        """Attaches the adapter to the model with conditional gating.

        Modifies `self.model.forward` to include the adapter output ONLY if the
        input contains categorical values that were not present in the base model's
        vocabulary at load time (e.g., a new protein).
        """
        if self.adapter is None:
            return
        adapter_ref = self.adapter

        self.adapter.eval()
        if self.model is None:
            raise RuntimeError("Model must be initialized.")

        original_forward = cast(Callable[..., Any], self.model.forward)
        gating_thresholds = []
        for feat, base_size in self.base_vocab_sizes.items():
            if feat in self.model.cat_feature_names:
                idx = self.model.cat_feature_names.index(feat)
                gating_thresholds.append((idx, base_size))

        def new_forward(
            x_num: torch.Tensor,
            x_cat: torch.Tensor,
            return_features: bool = False,
            return_physics_details: bool = False,
        ):
            base_output = original_forward(
                x_num,
                x_cat,
                return_features=return_features,
                return_physics_details=return_physics_details,
            )

            if return_features or return_physics_details:
                if isinstance(base_output, tuple):
                    pred = base_output[0]
                    extras = base_output[1:]
                else:
                    pred = base_output
                    extras = ()
            else:
                pred = base_output
                extras = None
            adapt = adapter_ref(x_num, x_cat)

            mask = torch.zeros(x_cat.size(0), 1, dtype=torch.bool, device=x_cat.device)
            for col_idx, cutoff in gating_thresholds:
                is_new = x_cat[:, col_idx] >= cutoff
                mask = mask | is_new.unsqueeze(1)

            final_pred = pred + (adapt * mask.float())
            if extras is not None and len(extras) > 0:
                return (final_pred,) + extras
            return final_pred

        self.model.forward = new_forward  # type: ignore

    def predict(
        self, df: pd.DataFrame, return_log_space: bool = False, batch_size: int = 256
    ) -> np.ndarray:
        """Runs batch inference on the dataframe.

        Args:
            df (pd.DataFrame): Input features.
            return_log_space (bool, optional): If True, returns raw log-transformed predictions.
                If False, returns inverse-transformed (linear) predictions. Defaults to False.
            batch_size (int, optional): Inference batch size. Defaults to 256.

        Returns:
            np.ndarray: Matrix of predictions (n_samples, n_targets).
        """
        self._ensure_hydrated()
        if self.processor is None or self.model is None:
            raise RuntimeError(
                "Model and Processor must be initialized before prediction."
            )

        self.model.eval()

        new_cats = self.processor.detect_new_categories(df)
        if new_cats:
            for feature, categories in new_cats.items():
                if isinstance(self.model, EnsembleModel):
                    for sub_model in self.model.models:
                        specific_model = cast(Model, sub_model)
                        expand_processor_and_model(
                            self.processor, specific_model, feature, categories
                        )
                elif isinstance(self.model, Model):
                    expand_processor_and_model(
                        self.processor, self.model, feature, categories
                    )

        X_num, X_cat = self.processor.transform(df)
        X_num_t, X_cat_t = to_tensors(X_num, X_cat)
        X_num_t = X_num_t.to(self.device)
        X_cat_t = X_cat_t.to(self.device)

        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_num_t), batch_size):
                batch_num = X_num_t[i : i + batch_size]
                batch_cat = X_cat_t[i : i + batch_size]
                pred = self.model(batch_num, batch_cat)
                predictions.append(pred.cpu().numpy())

        predictions = np.vstack(predictions)
        if not return_log_space:
            predictions = inverse_log_transform(predictions)
        return predictions

    def predict_with_uncertainty(
        self, df: pd.DataFrame, n_samples: int = 30, confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """Runs Monte Carlo Dropout inference to estimate uncertainty.

        Args:
            df (pd.DataFrame): Input features.
            n_samples (int, optional): Number of stochastic forward passes. Defaults to 30.
            confidence_level (float, optional): Desired confidence interval width (0.0-1.0).
                Defaults to 0.95.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing:
                - 'mean': Mean prediction across MC samples.
                - 'std': Standard deviation of predictions.
                - 'lower_ci': Lower bound of the confidence interval.
                - 'upper_ci': Upper bound of the confidence interval.
        """
        self._ensure_hydrated()
        if self.processor is None or self.model is None:
            raise RuntimeError(
                "Model and Processor must be initialized before prediction."
            )
        self.model.train()  # Enable Dropout
        new_cats = self.processor.detect_new_categories(df)
        if new_cats:
            for feature, categories in new_cats.items():
                if isinstance(self.model, EnsembleModel):
                    for sub_model in self.model.models:
                        specific_model = cast(Model, sub_model)
                        expand_processor_and_model(
                            self.processor, specific_model, feature, categories
                        )
                elif isinstance(self.model, Model):
                    expand_processor_and_model(
                        self.processor, self.model, feature, categories
                    )

        X_num, X_cat = self.processor.transform(df)
        X_num_t, X_cat_t = to_tensors(X_num, X_cat)
        X_num_t = X_num_t.to(self.device)
        X_cat_t = X_cat_t.to(self.device)

        mc_preds = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(X_num_t, X_cat_t)
                mc_preds.append(pred.cpu().numpy())

        mc_preds = np.array(mc_preds)
        mc_preds_linear = inverse_log_transform(mc_preds)

        mean_pred = mc_preds_linear.mean(axis=0)
        std_pred = mc_preds_linear.std(axis=0)

        alpha = 1 - confidence_level
        lower = np.percentile(mc_preds_linear, (alpha / 2) * 100, axis=0)
        upper = np.percentile(mc_preds_linear, (1 - alpha / 2) * 100, axis=0)

        self.model.eval()

        return {
            "mean": mean_pred,
            "std": std_pred,
            "lower_ci": lower,
            "upper_ci": upper,
        }

    def save_checkpoint(self, filepath: str):
        """Saves the current state (model + processor + params) to disk.

        Args:
            filepath (str): Destination path for the checkpoint file.
        """
        self._ensure_hydrated()
        if self.processor is None or self.model is None or self.best_params is None:
            raise RuntimeError(
                "Model and Processor and params must be initialized before saving a checkpoint."
            )
        save_model_checkpoint(self.model, self.processor, self.best_params, filepath)


if __name__ == "__main__":
    vp = ViscosityPredictor("models/experiments/20260120_102246/model_0.pt")
    df = pd.read_csv("exper.csv")
    output = vp.predict(df)
    for i, row in enumerate(output):
        print(i, row)
