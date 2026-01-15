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
    Package VisQAI models with specific `src` modules.
    """

    def __init__(
        self,
        output_dir: str = r"models\production",
        source_dir: str = "ml/src",
        source_files: List[str] = None,
        requirements_path: str = "requirements.txt",
        readme_path: str = "README.md",
        private_key_path: Optional[str] = None,
        enable_signing: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.source_dir = Path(source_dir)
        # Default source files if none provided
        self.source_files = source_files or [
            "config.py",
            "data.py",
            "inference.py",
            "layers.py",
            "loss.py",
            "management.py",
            "models.py",
            "utils.py",
        ]

        self.requirements_path = Path(requirements_path)
        self.readme_path = Path(readme_path)
        self.enable_signing = enable_signing

        if self.enable_signing:
            self.signer = ModuleSigner(private_key_path)
        else:
            self.signer = None

        # Validate existence of source directory
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
        print(f"Source Directory: {self.source_dir}")

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

            # 2. Add Source Modules (Specific List)
            for filename in self.source_files:
                file_path = self.source_dir / filename

                if not file_path.exists():
                    print(f"  WARNING: Source file not found, skipping: {filename}")
                    continue

                # Target path in zip: src/<filename>
                target_path = Path("src") / filename

                zipf.write(file_path, target_path)
                contents_list.append(str(target_path))

                if self.enable_signing:
                    signatures[str(target_path)] = self.signer.sign_file(file_path)

            print(f"  Added {len(self.source_files)} library modules to src/")

            # 3. Add Extras (Requirements, Readme)
            extras = [self.requirements_path, self.readme_path]
            for extra in extras:
                if extra.exists():
                    zipf.write(extra, extra.name)
                    contents_list.append(extra.name)
                    if self.enable_signing:
                        signatures[extra.name] = self.signer.sign_file(extra)
                    print(f"  Added: {extra.name}")

            # 4. Metadata
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

            # 5. Security Files
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
            "architecture": "VisQAI Standard",
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


def get_latest_checkpoints(experiments_dir: str = "models/experiments") -> List[str]:
    """
    Finds .pt files in the most recently modified directory within experiments_dir.
    """
    exp_path = Path(experiments_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")

    # Get all subdirectories
    subdirs = [d for d in exp_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No experiment directories found in {experiments_dir}")

    # Sort by modification time (or name if timestamps are in name)
    # Using modification time ensures we get the actual latest run
    latest_dir = max(subdirs, key=os.path.getmtime)
    print(f"Located latest experiment: {latest_dir.name}")

    # Find .pt files
    checkpoints = list(latest_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt files found in {latest_dir}")

    # Return as list of strings
    return [str(cp) for cp in checkpoints]


def main():
    """Execution script."""

    # Define the specific modules to package
    source_modules = [
        "config.py",
        "data.py",
        "inference.py",
        "layers.py",
        "loss.py",
        "management.py",
        "models.py",
        "utils.py",
    ]

    try:
        # 1. Locate Checkpoints
        print("Locating latest checkpoints...")
        checkpoints = get_latest_checkpoints("models/experiments")

        # 2. Initialize Packager
        packager = SecurePredictorPackager(
            output_dir=r"models\production",
            source_dir="ml/src",
            source_files=source_modules,
            enable_signing=True,
        )

        # 3. Create Package
        packager.package(
            model_paths=checkpoints,
            package_name=None,  # Will auto-generate name
            notes="Auto-packaged from latest experiment",
            version="1.1",
        )

    except Exception as e:
        print(f"\n[ERROR] Packaging failed: {e}")


if __name__ == "__main__":
    main()
