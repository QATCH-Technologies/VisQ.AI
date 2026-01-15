"""
packager.py - Enhanced secure packaging for VisQAI models

Updates:
- Supports modular 'src' package structure
- Recursive directory signing and archiving
- Preserves package hierarchy in the zip file
- Handles metadata for the new architecture
"""

import base64
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class ModuleSigner:
    """RSA-based signing for secure package verification."""

    def __init__(self, private_key_path: Optional[str] = None):
        if private_key_path and os.path.exists(private_key_path):
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
        else:
            self.private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )

    def sign_file(self, filepath: Path) -> str:
        """Sign a file and return base64-encoded signature."""
        with open(filepath, "rb") as f:
            content = f.read()
        signature = self.private_key.sign(
            content,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def sign_bytes(self, content: bytes) -> str:
        """Sign bytes and return base64-encoded signature."""
        signature = self.private_key.sign(
            content,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def get_public_key_pem(self) -> bytes:
        """Export public key in PEM format."""
        public_key = self.private_key.public_key()
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def save_private_key(self, filepath: str) -> None:
        """Save private key to file with restricted permissions."""
        pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with open(filepath, "wb") as f:
            f.write(pem)
        try:
            os.chmod(filepath, 0o600)
        except Exception:
            pass


class SecurePredictorPackager:
    """
    Package VisQAI models with the modular `src` library.
    """

    def __init__(
        self,
        output_dir: str = r"models\production",
        library_dir: str = "src",
        inference_path: str = "inference.py",
        requirements_path: str = "requirements.txt",
        readme_path: str = "README.md",
        private_key_path: Optional[str] = None,
        enable_signing: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.library_dir = Path(library_dir)
        self.inference_path = Path(inference_path)
        self.requirements_path = Path(requirements_path)
        self.readme_path = Path(readme_path)
        self.enable_signing = enable_signing

        if self.enable_signing:
            self.signer = ModuleSigner(private_key_path)
        else:
            self.signer = None

        # Validate existence
        if not self.library_dir.exists() or not self.library_dir.is_dir():
            raise FileNotFoundError(f"Library directory not found: {self.library_dir}")
        if not self.inference_path.exists():
            raise FileNotFoundError(
                f"Inference script not found: {self.inference_path}"
            )

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
        is_ensemble: bool = None,
    ) -> str:
        """
        Create the secure zip package.
        """
        # Normalize model_paths to list
        if isinstance(model_paths, str):
            model_paths = [model_paths]

        # Validate models
        for model_path in model_paths:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

        # Auto-detect ensemble
        if is_ensemble is None:
            is_ensemble = len(model_paths) > 1

        # Generate package name
        if package_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = "ensemble" if is_ensemble else "single"
            package_name = f"visqai_{model_type}_{timestamp}"

        if not package_name.endswith(".zip"):
            package_name += ".zip"

        zip_path = self.output_dir / package_name
        signatures = {}
        contents_list = []

        print(f"Creating {'ensemble' if is_ensemble else 'single'} model package...")
        print(f"Source Library: {self.library_dir}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:

            # 1. Add Model Checkpoints
            for i, model_path in enumerate(model_paths):
                if is_ensemble:
                    model_filename = f"model/checkpoint_{i}.pt"
                else:
                    model_filename = "model/checkpoint.pt"

                zipf.write(model_path, model_filename)
                contents_list.append(model_filename)
                if self.enable_signing:
                    signatures[model_filename] = self.signer.sign_file(Path(model_path))
            print(f"  Added {len(model_paths)} model checkpoint(s)")

            # 2. Add Modular Library (Recursively)
            # We map local 'src/' to 'src/src/' in the zip
            for file_path in self.library_dir.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix != ".pyc"
                    and "__pycache__" not in file_path.parts
                ):
                    # Calculate relative path for zip structure
                    rel_path = file_path.relative_to(self.library_dir.parent)
                    target_path = Path("src") / rel_path

                    zipf.write(file_path, target_path)
                    contents_list.append(str(target_path))

                    if self.enable_signing:
                        signatures[str(target_path)] = self.signer.sign_file(file_path)
            print(f"  Added library module: src/{self.library_dir.name}")

            # 3. Add Inference Script
            target_inf_path = "src/inference.py"
            zipf.write(self.inference_path, target_inf_path)
            contents_list.append(target_inf_path)
            if self.enable_signing:
                signatures[target_inf_path] = self.signer.sign_file(self.inference_path)
            print(f"  Added inference script: {target_inf_path}")

            # 4. Add Extras (Requirements, Readme)
            extras = [self.requirements_path, self.readme_path]
            for extra in extras:
                if extra.exists():
                    zipf.write(extra, extra.name)
                    contents_list.append(extra.name)
                    if self.enable_signing:
                        signatures[extra.name] = self.signer.sign_file(extra)
                    print(f"  Added: {extra.name}")

            # 5. Metadata
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
                signatures["model/metadata.json"] = self.signer.sign_bytes(
                    metadata_json.encode("utf-8")
                )
            print(f"  Added: model/metadata.json")

            # 6. Security Files
            if self.enable_signing:
                self._add_security_files(zipf, signatures)
                print(f"  Added: security signatures")

        # Save private key if requested
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
        """Create comprehensive metadata dictionary."""
        total_size = sum(os.stat(p).st_size for p in model_paths)

        metadata = {
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
            "architecture": "Modular VisQAI (src)",
            "package_contents": contents,
        }
        return metadata

    def _add_security_files(
        self, zipf: zipfile.ZipFile, signatures: Dict[str, str]
    ) -> None:
        """Add security files (public key and signatures)."""
        public_key_pem = self.signer.get_public_key_pem()
        zipf.writestr("security/public_key.pem", public_key_pem)

        signatures_json = json.dumps(signatures, indent=2)
        zipf.writestr("security/signatures.json", signatures_json)


def main():
    """Example usage."""
    # Ensure dummy files exist for the example to run if this file is executed directly
    if not os.path.exists("src"):
        print(
            "Error: 'src' directory missing. Please ensure directory structure exists."
        )
        return

    packager = SecurePredictorPackager(
        output_dir=r"models\production",
        library_dir="src",
        inference_path="inference.py",
        enable_signing=True,
    )

    # Example: Packaging an ensemble (Update paths to real checkpoints)
    # ensemble_checkpoints = ["checkpoints/ensemble_model_0.pt", "checkpoints/ensemble_model_1.pt"]

    # For demo, we just print instructions
    print("Packager Initialized.")
    print("To package models, ensure checkpoints exist and run:")
    print("packager.package(model_paths=['path/to/ckpt.pt'], package_name='MyPackage')")


if __name__ == "__main__":
    main()
