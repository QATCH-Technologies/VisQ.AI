# visQAI/src/model/predictor.py
"""
Module: predictor

Provides utilities to load predictor implementations either from Python source files
or serialized binaries, and wraps them in a unified BasePredictor interface.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-05-01

Version:
    1.0.0
"""
import importlib.util
import os
import sys
import pickle
from pathlib import Path
from types import ModuleType
from typing import Any, Optional
import joblib


def load_module_from_file(module_path: str) -> ModuleType:
    """
    Dynamically load a Python module from a file path.

    Args:
        module_path (str): Filesystem path to the .py module to load.

    Returns:
        ModuleType: The imported module object.

    Raises:
        TypeError: If module_path is not a string.
        FileNotFoundError: If the specified module file does not exist.
        ImportError: If the module spec cannot be created or loaded.
    """
    if not isinstance(module_path, str):
        raise TypeError(
            f"module_path must be a string, got {type(module_path)}")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Module file not found: {module_path}")
    module_name = Path(module_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error loading module '{module_path}': {e}")
    return module


def load_binary_predictor(binary_path: str) -> Any:
    """
    Load a predictor object from a serialized binary file.

    Supports .joblib, .pkl, and .pickle extensions.

    Args:
        binary_path (str): Filesystem path to the serialized predictor.

    Returns:
        Any: The deserialized predictor object.

    Raises:
        TypeError: If binary_path is not a string.
        FileNotFoundError: If the binary file does not exist.
        ValueError: If the file extension is not supported.
        pickle.UnpicklingError: If loading a .pickle file fails.
    """
    if not isinstance(binary_path, str):
        raise TypeError(
            f"binary_path must be a string, got {type(binary_path)}")
    if not os.path.isfile(binary_path):
        raise FileNotFoundError(f"Binary file not found: {binary_path}")
    ext = Path(binary_path).suffix.lower()
    try:
        if ext in {'.joblib', '.pkl'}:
            return joblib.load(binary_path)
        if ext == '.pickle':
            with open(binary_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load predictor from '{binary_path}': {e}")
    raise ValueError(f"Unsupported predictor format: {ext}")


class BasePredictor:
    """
    Unified interface for loading and invoking predictors from Python source or binary formats.

    Attributes:
        VERSION (str): Version string of the BasePredictor interface.

    Instance Attributes:
        _path (Path): Path to the predictor file or module.
        _module (Optional[ModuleType]): Loaded Python module, if source-based.
        _instance (Optional[Any]): Loaded predictor instance, if binary-based.
        model (Optional[Any]): Underlying model object from the predictor.
        preprocessor (Optional[Any]): Optional preprocessor object, if provided.
    """

    VERSION = "1.0.0"

    def __init__(self, predictor_path: str):
        """
        Initialize the BasePredictor by loading the underlying predictor.

        Args:
            predictor_path (str): Path to the predictor file (.py or binary).

        Raises:
            TypeError: If predictor_path is not a string.
            FileNotFoundError: If the predictor file does not exist.
            AttributeError: If the loaded predictor lacks a callable 'predict'.
        """
        if not isinstance(predictor_path, str):
            raise TypeError(
                f"predictor_path must be a string, got {type(predictor_path)}")
        self._path: Path = Path(predictor_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Predictor file not found: {self._path}")
        self._module: Optional[ModuleType] = None
        self._instance: Optional[Any] = None
        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self._load_predictor()

    def _load_predictor(self) -> None:
        """
        Internal method to load the predictor based on file extension.

        Raises:
            AttributeError: If the loaded object/module does not define a callable `predict(data)`.
        """
        ext = self._path.suffix.lower()
        if ext == '.py':
            module = load_module_from_file(str(self._path))
            if not hasattr(module, 'predict') or not callable(module.predict):
                raise AttributeError(
                    f"Module '{self._path}' must define a callable 'predict(data)'."
                )
            self._module = module
            self.model = getattr(module, 'model', None)
            self.preprocessor = getattr(module, 'preprocessor', None)
        else:
            predictor = load_binary_predictor(str(self._path))
            if not hasattr(predictor, 'predict') or not callable(predictor.predict):
                raise AttributeError(
                    f"Object loaded from '{self._path}' must have a callable 'predict(data)'."
                )
            self._instance = predictor
            self.model = getattr(predictor, 'model', None)
            self.preprocessor = getattr(predictor, 'preprocessor', None)

    def predict(self, data: Any, *args, **kwargs) -> Any:
        """
        Invoke the predictor's `predict` method on input data.

        Args:
            data (Any): Input data for prediction.
            *args: Positional arguments forwarded to the predictor.
            **kwargs: Keyword arguments forwarded to the predictor.

        Returns:
            Any: Prediction result from the underlying predictor.

        Raises:
            RuntimeError: If no predictor has been loaded successfully.
        """
        if data is None:
            raise ValueError("Input data for prediction cannot be None.")
        if self._module:
            return self._module.predict(data, *args, **kwargs)
        if self._instance:
            return self._instance.predict(data, *args, **kwargs)
        raise RuntimeError(
            "No predictor loaded; check the predictor_path provided."
        )

    def reload(self) -> None:
        """
        Reload the predictor from its current path, resetting any cached state.

        Raises:
            Exception: Propagates any errors from underlying load operations.
        """
        self._module = None
        self._instance = None
        self.model = None
        self.preprocessor = None
        self._load_predictor()

    def load(self, new_path: str) -> None:
        """
        Change the predictor path and reload the underlying predictor.

        Args:
            new_path (str): New filesystem path to the predictor.

        Raises:
            TypeError: If new_path is not a string.
            FileNotFoundError: If the new predictor file does not exist.
        """
        if not isinstance(new_path, str):
            raise TypeError(f"new_path must be a string, got {type(new_path)}")
        self._path = Path(new_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Predictor file not found: {self._path}")
        self.reload()

    @property
    def path(self) -> str:
        """
        str: Current filesystem path of the predictor.
        """
        return str(self._path)

    @path.setter
    def path(self, new_path: str) -> None:
        """
        Set a new path for the predictor and reload it.

        Args:
            new_path (str): Filesystem path to the new predictor.

        Raises:
            TypeError: If new_path is not a string.
            FileNotFoundError: If the new predictor file does not exist.
        """
        if not isinstance(new_path, str):
            raise TypeError(f"new_path must be a string, got {type(new_path)}")
        self._path = Path(new_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Predictor file not found: {self._path}")
        self.reload()

    @property
    def predictor_type(self) -> str:
        """
        str: Type of the loaded predictor: 'module' for source, 'binary' for serialized.
        """
        return 'module' if self._module else 'binary'

    def __repr__(self) -> str:
        """
        Return an unambiguous representation of the BasePredictor.

        Returns:
            str: Informative string including version, type, and path.
        """
        return (
            f"<BasePredictor version={self.VERSION} "
            f"type={self.predictor_type} path={self._path}>"
        )
