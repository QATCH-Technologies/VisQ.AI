import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PredictorInfo:
    """
    Metadata container for a predictor asset.
    """
    name: str
    description: str
    created_at: datetime


class PredictorsController:
    """
    Manages predictor assets on the filesystem under an 'objects' directory.
    Each predictor has its own subdirectory named after the predictor name,
    containing the predictor, model, preprocessor, and metadata.
    """

    def __init__(self, base_dir: str = None):
        # Base directory for storing predictor objects
        self.base_dir = Path(base_dir or os.getcwd()) / "objects"
        self.base_dir.mkdir(exist_ok=True)

    def get_predictors(self) -> List[PredictorInfo]:
        """
        Returns a list of PredictorInfo for all stored predictors.
        """
        predictors: List[PredictorInfo] = []
        for p in self.base_dir.iterdir():
            if not p.is_dir():
                continue
            meta_file = p / "metadata.json"
            if meta_file.exists():
                data = json.loads(meta_file.read_text())
                name = data.get("name", p.name)
                description = data.get("description", "")
                created_at = datetime.fromisoformat(data.get("created_at"))
            else:
                name = p.name
                description = ""
                created_at = datetime.fromtimestamp(p.stat().st_ctime)
            predictors.append(PredictorInfo(
                name=name, description=description, created_at=created_at))
        return predictors

    def add_predictor(
        self,
        name: str,
        description: str,
        predictor_path: str,
        model_path: str,
        preprocessor_path: str
    ) -> PredictorInfo:
        """
        Creates a new predictor directory under objects/<name> by copying the
        given files, and writes metadata.json with name, description, and timestamp.

        :return: PredictorInfo for the newly added predictor
        """
        target_dir = self.base_dir / name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir()
        # Copy asset files
        shutil.copy2(predictor_path, target_dir / "predictor.pkl")
        shutil.copy2(model_path, target_dir / "model.pkl")
        shutil.copy2(preprocessor_path, target_dir / "preprocessor.pkl")

        created_at = datetime.now()
        metadata = {
            "name": name,
            "description": description,
            "created_at": created_at.isoformat()
        }
        (target_dir / "metadata.json").write_text(json.dumps(metadata))
        return PredictorInfo(name=name, description=description, created_at=created_at)

    def get_predictor(self, name: str) -> Optional[PredictorInfo]:
        """
        Retrieves metadata for a single predictor by name.

        :return: PredictorInfo or None if not found
        """
        target_dir = self.base_dir / name
        if not target_dir.exists():
            return None
        meta_file = target_dir / "metadata.json"
        if meta_file.exists():
            data = json.loads(meta_file.read_text())
            description = data.get("description", "")
            created_at = datetime.fromisoformat(data.get("created_at"))
        else:
            description = ""
            created_at = datetime.fromtimestamp(target_dir.stat().st_ctime)
        return PredictorInfo(name=name, description=description, created_at=created_at)

    def update_predictor(
        self,
        name: str,
        new_name: str,
        description: str,
        predictor_path: Optional[str] = None,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None
    ) -> Optional[PredictorInfo]:
        """
        Updates an existing predictor's files, metadata, and optionally renames it.

        :return: PredictorInfo for the updated predictor, or None if original not found
        """
        target_dir = self.base_dir / name
        if not target_dir.exists():
            return None
        # Overwrite asset files if new paths provided
        if predictor_path:
            shutil.copy2(predictor_path, target_dir / "predictor.pkl")
        if model_path:
            shutil.copy2(model_path, target_dir / "model.pkl")
        if preprocessor_path:
            shutil.copy2(preprocessor_path, target_dir / "preprocessor.pkl")
        # Load existing creation timestamp
        meta_file = target_dir / "metadata.json"
        if meta_file.exists():
            data = json.loads(meta_file.read_text())
            created_at = datetime.fromisoformat(data.get("created_at"))
        else:
            created_at = datetime.fromtimestamp(target_dir.stat().st_ctime)
        # Rename directory if name changed
        final_name = new_name.strip() or name
        if final_name != name:
            new_dir = self.base_dir / final_name
            if new_dir.exists():
                shutil.rmtree(new_dir)
            target_dir.rename(new_dir)
            target_dir = new_dir
        # Write updated metadata
        metadata = {
            "name": final_name,
            "description": description,
            "created_at": created_at.isoformat()
        }
        (target_dir / "metadata.json").write_text(json.dumps(metadata))
        return PredictorInfo(name=final_name, description=description, created_at=created_at)

    def delete_predictor(self, name: str) -> bool:
        """
        Deletes the predictor directory and all its contents.

        :return: True if deleted, False if not found
        """
        target_dir = self.base_dir / name
        if target_dir.exists():
            shutil.rmtree(target_dir)
            return True
        return False

    def predictor_exists(self, name: str) -> bool:
        """
        Checks if a predictor with the given name exists under objects.
        """
        return (self.base_dir / name).is_dir()

    # Utility methods to load assets
    def load_preprocessor(self, name: str):
        from joblib import load
        path = self.base_dir / name / "preprocessor.pkl"
        return load(path)

    def load_model(self, name: str, model_filename: str = "model.pkl"):
        import joblib
        path = self.base_dir / name / model_filename
        return joblib.load(path)

    def load_predictor_class(self, name: str, predictor_filename: str = "predictor.pkl"):
        import joblib
        path = self.base_dir / name / predictor_filename
        return joblib.load(path)
