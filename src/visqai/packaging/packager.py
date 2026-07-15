"""
packager.py
===========
SecurePredictorPackager: builds a cryptographically signed, zipped deployment
artifact from trained checkpoint(s) + the runtime-inference-needed source
modules.

Moved from scripts/packager.py. The hardcoded 8-file source list
(config.py, data.py, inference.py, layers.py, loss.py, management.py,
models.py, utils.py under `ml\\visq_ml`) matched neither this package nor
ml/cnp_mk2 -- it was a leftover reference to the `visq_ml` package generation
deleted from this repo 4 commits before this refactor started (scripts/ was
never updated to follow). It's replaced here with the actual runtime-
inference module set: features/, models/cnp.py, physics/priors.py,
preprocessing/pipeline.py, inference/predictor.py -- training/eval/CLI code
is deliberately NOT packaged, since it was never meant to ship to a client.
"""

from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from visqai.packaging.signer import ModuleSigner

# Runtime-inference-needed modules only (relative to src/visqai/), in the
# order they'll appear under `src/` in the package zip.
DEFAULT_SOURCE_MODULES = [
    "features/categorical.py",
    "features/charge.py",
    "physics/priors.py",
    "preprocessing/pipeline.py",
    "models/cnp.py",
    "inference/predictor.py",
]


class SecurePredictorPackager:
    """Package a visqai model with the runtime-inference source modules it needs."""

    def __init__(
        self,
        output_dir: str = r"models\production",
        source_dir: str = "src/visqai",
        source_files: Optional[List[str]] = None,
        requirements_path: str = "requirements.txt",
        readme_path: str = "README.md",
        private_key_path: Optional[str] = None,
        enable_signing: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.source_dir = Path(source_dir)
        self.source_files = source_files or list(DEFAULT_SOURCE_MODULES)

        self.requirements_path = Path(requirements_path)
        self.readme_path = Path(readme_path)
        self.enable_signing = enable_signing

        self.signer = ModuleSigner(private_key_path) if self.enable_signing else None

        if not self.source_dir.exists() or not self.source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

    def package(
        self,
        model_paths: Union[str, List[str]],
        package_name: Optional[str] = None,
        model_created_date: Optional[str] = None,
        client: Optional[str] = None,
        author: Optional[str] = None,
        version: Optional[str] = None,
        notes: Optional[str] = None,
        save_private_key: Optional[str] = None,
        is_ensemble: Optional[bool] = None,
    ) -> str:
        """Create the secure zip package."""
        if isinstance(model_paths, str):
            model_paths = [model_paths]

        for model_path in model_paths:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

        if is_ensemble is None:
            is_ensemble = len(model_paths) > 1

        if package_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = "ensemble" if is_ensemble else "single"
            package_name = f"visqai_{model_type}_{timestamp}"
        if not package_name.endswith(".zip"):
            package_name += ".zip"

        zip_path = self.output_dir / package_name
        signatures: Dict[str, str] = {}
        contents_list: List[str] = []

        print(f"Creating {'ensemble' if is_ensemble else 'single'} model package...")
        print(f"Source Directory: {self.source_dir}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for i, model_path in enumerate(model_paths):
                model_filename = f"model/checkpoint_{i}.pt" if is_ensemble else "model/checkpoint.pt"
                zipf.write(model_path, model_filename)
                contents_list.append(model_filename)
                if self.enable_signing:
                    signatures[model_filename] = self.signer.sign_file(Path(model_path))
            print(f"  Added {len(model_paths)} model checkpoint(s)")

            for filename in self.source_files:
                file_path = self.source_dir / filename
                if not file_path.exists():
                    print(f"  WARNING: Source file not found, skipping: {filename}")
                    continue

                target_path_str = (Path("src") / filename).as_posix()
                zipf.write(file_path, target_path_str)
                contents_list.append(target_path_str)
                if self.enable_signing:
                    signatures[target_path_str] = self.signer.sign_file(file_path)

            print(f"  Added {len(self.source_files)} library modules to src/")

            for extra in (self.requirements_path, self.readme_path):
                if extra.exists():
                    zipf.write(extra, extra.name)
                    contents_list.append(extra.name)
                    if self.enable_signing:
                        signatures[extra.name] = self.signer.sign_file(extra)
                    print(f"  Added: {extra.name}")

            metadata = self._create_metadata(
                model_paths=model_paths,
                is_ensemble=is_ensemble,
                contents=contents_list,
                model_created_date=model_created_date,
                client=client,
                author=author,
                version=version,
                notes=notes,
            )
            metadata_json = json.dumps(metadata, indent=2)
            zipf.writestr("model/metadata.json", metadata_json)
            if self.enable_signing:
                signatures["model/metadata.json"] = self.signer.sign_bytes(metadata_json.encode("utf-8"))
            print("  Added: model/metadata.json")

            if self.enable_signing:
                self._add_security_files(zipf, signatures)
                print("  Added: security signatures")

        if save_private_key and self.enable_signing:
            self.signer.save_private_key(save_private_key)
            print(f"Private key saved to: {save_private_key}")

        print(f"\nPackage created successfully: {zip_path}")
        print(f"Size: {zip_path.stat().st_size / (1024*1024):.2f} MB")

        return str(zip_path)

    def _create_metadata(
        self,
        model_paths: List[str],
        is_ensemble: bool,
        contents: List[str],
        model_created_date: Optional[str] = None,
        client: Optional[str] = None,
        author: Optional[str] = None,
        version: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        total_size = sum(os.stat(p).st_size for p in model_paths)
        return {
            "version": version or "1.0",
            "model_type": "ensemble" if is_ensemble else "single",
            "n_models": len(model_paths),
            "model_created_date": model_created_date or "Unknown",
            "packaged_date": datetime.now().isoformat(),
            "client": client or "Unknown",
            "author": author or "Unknown",
            "notes": notes or "",
            "total_model_size_mb": round(total_size / (1024 * 1024), 2),
            "cryptographically_signed": self.enable_signing,
            "architecture": "VisQAI Standard",
            "package_contents": contents,
        }

    def _add_security_files(self, zipf: zipfile.ZipFile, signatures: Dict[str, str]) -> None:
        public_key_pem = self.signer.get_public_key_pem()
        zipf.writestr("security/public_key.pem", public_key_pem)
        signatures_json = json.dumps(signatures, indent=2)
        zipf.writestr("security/signatures.json", signatures_json)


def get_latest_checkpoints(experiments_dir: str = "models/experiments") -> List[str]:
    """Finds .pt files in the most recently modified directory within experiments_dir."""
    exp_path = Path(experiments_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")

    subdirs = [d for d in exp_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No experiment directories found in {experiments_dir}")

    latest_dir = max(subdirs, key=os.path.getmtime)
    print(f"Located latest experiment: {latest_dir.name}")

    checkpoints = list(latest_dir.glob("*.pt")) + list(latest_dir.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt/.pth files found in {latest_dir}")

    return [str(cp) for cp in checkpoints]
