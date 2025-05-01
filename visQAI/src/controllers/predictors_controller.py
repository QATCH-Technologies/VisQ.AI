# visQAI/src/controllers/predictor_controller.py
"""
Module: predictors_controller

This module provides a controller class for managing predictor files in a storage directory.
It supports CRUD operations: adding, listing, retrieving, updating, and removing predictors.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-05-01

Version:
    1.0.0
"""
import json
from pathlib import Path
from typing import Any, Dict, List
from src.model.predictor import BasePredictor


class PredictorsController:
    """
    Controller for managing predictor files with persistent indexing.

    This class allows adding, listing, retrieving, updating, and removing predictor
    files stored in a specified directory. Supported file types include Python modules
    and binary model files.

    Attributes:
        SUPPORTED_EXT(set[str]): Allowed file extensions for predictor files.
        storage(Path): Directory where predictor files are stored.
        index_file(Path): JSON file path used to maintain an index of stored predictors.
        _index(Dict[str, Dict[str, Any]]): In-memory index mapping predictor names to metadata.
    """

    SUPPORTED_EXT = {'.py', '.pkl', '.joblib', '.pickle'}

    def __init__(self, storage_dir: str):
        """
        Initialize the controller and ensure the storage directory and index file exist.

        Args:
            storage_dir(str): Path to the directory for storing predictor files.

        Raises:
            OSError: If the storage directory cannot be created or accessed.
        """
        self.storage = Path(storage_dir)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage / 'index.json'
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the index from the JSON file, or return an empty index if the file does not exist.

        Returns:
            Dict[str, Dict[str, Any]]: The loaded index mapping predictor names to metadata.
        """
        if self.index_file.exists():
            with open(self.index_file) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """
        Persist the in -memory index to the JSON file with indentation for readability.

        Returns:
            None
        """
        with open(self.index_file, 'w') as f:
            json.dump(self._index, f, indent=2)

    def list(self) -> List[str]:
        """
        List all registered predictor names in the index.

        Returns:
            List[str]: A list of predictor names.
        """
        return list(self._index.keys())

    def add(self, name: str, predictor_path: str) -> None:
        """
        Add a new predictor file to storage and update the index.

        Args:
            name(str): Unique name to register for the predictor.
            predictor_path(str): Path to the source predictor file to add.

        Raises:
            ValueError: If a predictor with the given name already exists or if the file extension is unsupported.
            FileNotFoundError: If the source predictor file does not exist.
        """
        if name in self._index:
            raise ValueError(f"Predictor '{name}' already exists.")
        src = Path(predictor_path)
        if not src.exists():
            raise FileNotFoundError(f"File not found: {src}")
        ext = src.suffix.lower()
        if ext not in self.SUPPORTED_EXT:
            raise ValueError(f"Unsupported extension: {ext}")
        dest = self.storage / f"{name}{ext}"
        dest.write_bytes(src.read_bytes())
        self._index[name] = {'path': str(dest), 'ext': ext}
        self._save_index()

    def get(self, name: str) -> BasePredictor:
        """
        Retrieve a registered predictor by name and instantiate it.

        Args:
            name(str): Name of the predictor to retrieve.

        Returns:
            BasePredictor: An instance of the predictor initialized from its stored file.

        Raises:
            KeyError: If no predictor with the given name exists in the index.
        """
        if name not in self._index:
            raise KeyError(f"No predictor '{name}'")
        info = self._index[name]
        return BasePredictor(info['path'])

    def update(self, name: str, predictor_path: str) -> None:
        """
        Replace an existing predictor file with a new one and update the index.

        Args:
            name(str): Name of the predictor to update.
            predictor_path(str): Path to the new predictor file.

        Raises:
            KeyError: If no predictor with the given name exists.
            ValueError: If the new file extension is unsupported.
        """
        if name not in self._index:
            raise KeyError(f"No predictor '{name}' to update")
        old = Path(self._index[name]['path'])
        if old.exists():
            old.unlink()
        src = Path(predictor_path)
        ext = src.suffix.lower()
        if ext not in self.SUPPORTED_EXT:
            raise ValueError(f"Unsupported extension: {ext}")
        dest = self.storage / f"{name}{ext}"
        dest.write_bytes(src.read_bytes())
        self._index[name] = {'path': str(dest), 'ext': ext}
        self._save_index()

    def remove(self, name: str) -> None:
        """
        Remove a registered predictor file and delete its index entry.

        Args:
            name(str): Name of the predictor to remove.

        Raises:
            KeyError: If no predictor with the given name exists.
        """
        if name not in self._index:
            raise KeyError(f"No predictor '{name}' to remove")
        file = Path(self._index[name]['path'])
        if file.exists():
            file.unlink()
        del self._index[name]
        self._save_index()
