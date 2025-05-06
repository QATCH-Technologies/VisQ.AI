import inspect
import keras_tuner as kt
import tensorflow as tf
import keras
from typing import (
    Any, Callable, Dict, Optional, Union
)


class GenericHyperModel(kt.HyperModel):
    """A KerasTuner HyperModel that can drive *any* builder + HP spec + compile_args."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        builder: Callable[..., keras.Model],
        hyperparam_space: Dict[str, Dict[str, Any]],
        compile_args: Union[Dict[str, Any],
                            Callable[[Dict[str, Any]], Dict[str, Any]]],
    ):
        """
        Args:
            input_dim:  Number of input features.
            output_dim: Number of outputs.
            builder:    callable(input_dim, output_dim, **hp_kwargs) → uncompiled Model.
            hyperparam_space: 
               dict of {hp_name: {type: "Choice"|"Int"|"Float", …}} describing your HPs.
            compile_args:
               either a static dict for model.compile(**compile_args)
               or a function compile_args(hp_values) → dict.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.builder = builder
        self.hyperparam_space = hyperparam_space
        self.compile_args = compile_args

        # Sanity-check that your hyperparam names match builder signature:
        sig = inspect.signature(self.builder)
        allowed = set(sig.parameters) - {"input_dim", "output_dim"}
        extra = set(hyperparam_space) - allowed
        if extra:
            raise ValueError(f"Unknown HPs for builder: {extra}")

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        # 1) draw all HPs
        hp_values: Dict[str, Any] = {}
        for name, cfg in self.hyperparam_space.items():
            kind = cfg["type"]
            if kind == "Choice":
                hp_values[name] = hp.Choice(
                    name, cfg["values"], default=cfg.get("default")
                )
            elif kind == "Int":
                hp_values[name] = hp.Int(
                    name,
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    step=cfg.get("step", 1),
                    default=cfg.get("default"),
                )
            elif kind == "Float":
                hp_values[name] = hp.Float(
                    name,
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    sampling=cfg.get("sampling"),
                    default=cfg.get("default"),
                )
            else:
                raise ValueError(f"Unsupported HP type: {kind}")

        # 2) build & compile
        model = self.builder(
            self.input_dim, self.output_dim, **hp_values
        )
        if callable(self.compile_args):
            compile_kwargs = self.compile_args(hp_values)
        else:
            compile_kwargs = self.compile_args
        model.compile(**compile_kwargs)
        return model
