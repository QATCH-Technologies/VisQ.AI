"""
packager_o_net.py

Packages model assets and inference code into a signed .visq archive.
Supports declaring multiple code modules with explicit load ordering and
an entry-point class, so each package is fully self-describing.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-26

Version:
    2.0 (Multi-module manifest support)
"""

import base64
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


# ==========================================
# 1. Security Module (RSA Signing)
# ==========================================
class ModuleSigner:
    """RSA-based signing for secure package verification."""

    def __init__(self, private_key_path: Optional[str] = None):
        if private_key_path and os.path.exists(private_key_path):
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
            print(f"Loaded private key from {private_key_path}")
        else:
            print("Generating NEW private key (For demo only)...")
            self.private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )

    def sign_bytes(self, data: bytes) -> str:
        """Sign bytes and return base64-encoded signature."""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def get_public_key_pem(self) -> str:
        public_key = self.private_key.public_key()
        pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return pem.decode("utf-8")


# ==========================================
# 2. CNP Packager Class
# ==========================================
class CNPModelPackager:
    """Creates signed .visq packages with self-describing module manifests.

    Args:
        output_dir: Directory where the .visq archive will be written.
        model_source_dir: Directory containing model weights and preprocessors.
        code_source_files: One or more Python source files to bundle. Each is
            a dict with keys ``path`` (str) and ``type`` (str, e.g.
            ``"inference_code"``, ``"model_definition"``, ``"utils"``).
            For backward-compat a bare string path is accepted and treated
            as type ``"inference_code"``.
        entry_point_class: Fully-qualified ``"filename.py:ClassName"`` that the
            loader should instantiate. Defaults to auto-detection of the first
            ``inference_code`` file.
        private_key_path: Optional RSA private key for signing.
    """

    def __init__(
        self,
        output_dir: str,
        model_source_dir: str,
        code_source_files: Any = None,
        entry_point_class: Optional[str] = None,
        private_key_path: Optional[str] = None,
        # ---- backward-compat single-file shorthand ----
        code_source_file: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_source_dir)
        self.signer = ModuleSigner(private_key_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Normalise code_source_files into List[Dict]
        self.code_sources: List[Dict[str, str]] = []
        if code_source_files:
            for entry in code_source_files:
                if isinstance(entry, str):
                    entry = {"path": entry, "type": "inference_code"}
                self.code_sources.append(entry)
        elif code_source_file:
            # Legacy single-file API
            self.code_sources.append({"path": code_source_file, "type": "inference_code"})

        self.entry_point_class = entry_point_class

    def package(self, package_name: str, version: str, notes: str = "") -> Path:
        """Creates a signed .visq (zip) package containing model assets and code.

        Args:
            package_name: Human-readable name embedded in the manifest.
            version: Semantic version string.
            notes: Optional release notes.

        Returns:
            Path to the generated .visq file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_name = f"{package_name}_v{version}_{timestamp}.visq"
        output_path = self.output_dir / final_name

        print(f"Packaging {final_name}...")

        # --------------------------------------------------
        # 1. Identify binary/data assets from model_dir
        # --------------------------------------------------
        data_assets: Dict[str, Path] = {
            "model_checkpoint": self.model_dir / "best_model.pth",
            "preprocessor": self.model_dir / "preprocessor.pkl",
            "physics_scaler": self.model_dir / "physics_scaler.pkl",
        }
        for k, p in data_assets.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing required asset: {k} -> {p}")

        # --------------------------------------------------
        # 2. Validate code modules and build load_order
        # --------------------------------------------------
        code_paths: Dict[str, Path] = {}
        load_order: List[str] = []
        for entry in self.code_sources:
            p = Path(entry["path"])
            if not p.exists():
                raise FileNotFoundError(f"Missing code module: {entry['type']} -> {p}")
            code_paths[p.name] = p
            load_order.append(p.name)

        # Merge into a single asset dict for hashing
        all_assets: Dict[str, Path] = {}
        asset_types: Dict[str, str] = {}
        for key, path in data_assets.items():
            all_assets[path.name] = path
            asset_types[path.name] = key
        for entry in self.code_sources:
            p = Path(entry["path"])
            all_assets[p.name] = p
            asset_types[p.name] = entry["type"]

        # --------------------------------------------------
        # 3. Calculate hashes & signatures
        # --------------------------------------------------
        manifest_files: Dict[str, Any] = {}
        for archive_name, path in all_assets.items():
            with open(path, "rb") as f:
                content = f.read()

            file_hash = hashes.Hash(hashes.SHA256(), backend=default_backend())
            file_hash.update(content)
            digest = file_hash.finalize().hex()

            signature = self.signer.sign_bytes(content)

            manifest_files[archive_name] = {
                "type": asset_types[archive_name],
                "sha256": digest,
                "signature": signature,
                "size_bytes": len(content),
            }

        # --------------------------------------------------
        # 4. Resolve entry point
        # --------------------------------------------------
        entry_point: Dict[str, str] = {}
        if self.entry_point_class:
            # Explicit "filename.py:ClassName"
            module_file, class_name = self.entry_point_class.split(":")
            entry_point = {"module": module_file, "class": class_name}
        elif load_order:
            # Default: first inference_code module, class auto-detected at load
            first_inference = next(
                (Path(e["path"]).name for e in self.code_sources if "inference" in e["type"]),
                load_order[0],
            )
            entry_point = {"module": first_inference}

        # --------------------------------------------------
        # 5. Assemble manifest
        # --------------------------------------------------
        manifest = {
            "package_name": package_name,
            "version": version,
            "architecture": "CrossSampleCNP",
            "created_at": datetime.now().isoformat(),
            "notes": notes,
            "files": manifest_files,
            "modules": {
                "load_order": load_order,
                "entry_point": entry_point,
            },
            "public_key": self.signer.get_public_key_pem(),
        }

        # --------------------------------------------------
        # 6. Write ZIP
        # --------------------------------------------------
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for archive_name, path in all_assets.items():
                zf.write(path, arcname=archive_name)

            manifest_json = json.dumps(manifest, indent=2)
            zf.writestr("manifest.json", manifest_json)

            manifest_sig = self.signer.sign_bytes(manifest_json.encode("utf-8"))
            zf.writestr("manifest.sig", manifest_sig)

        print(f"✅ Package created successfully at: {output_path}")
        return output_path


# ==========================================
# 3. Execution Script
# ==========================================
def main():
    MODEL_DIR = "models/experiments/o_net_v3_debug_aug"
    OUTPUT_DIR = "models/production"

    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory '{MODEL_DIR}' not found. Did you run training?")
        return

    # -----------------------------------------------------------
    # Define ALL code modules that this package needs at runtime.
    # load_order is inferred from list order.
    # -----------------------------------------------------------
    code_modules = [
        # If you have shared utilities, list them first so they're
        # available when later modules import them:
        # {"path": "ml/test/model_definitions.py", "type": "model_definition"},
        # {"path": "ml/test/utils_cnp.py",         "type": "utils"},
        {"path": "ml/test/inference_o_net.py", "type": "inference_code"},
    ]

    for mod in code_modules:
        if not os.path.exists(mod["path"]):
            print(f"Error: Code module '{mod['path']}' not found.")
            return

    packager = CNPModelPackager(
        output_dir=OUTPUT_DIR,
        model_source_dir=MODEL_DIR,
        code_source_files=code_modules,
        entry_point_class="inference_o_net.py:ViscosityPredictorCNP",
    )

    packager.package(
        package_name="VisQ",
        version="2.0.0",
        notes="Multi-module manifest support.",
    )


if __name__ == "__main__":
    main()
