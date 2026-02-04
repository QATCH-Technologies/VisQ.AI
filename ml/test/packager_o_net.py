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
    def __init__(
        self,
        output_dir: str,
        model_source_dir: str,  # Where best_model.pth lives
        code_source_file: str,  # Path to inference_cnp.py
        private_key_path: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_source_dir)
        self.code_file = Path(code_source_file)
        self.signer = ModuleSigner(private_key_path)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def package(self, package_name: str, version: str, notes: str = "") -> Path:
        """
        Creates a signed .visq (zip) package containing model assets and code.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_name = f"{package_name}_v{version}_{timestamp}.visq"
        output_path = self.output_dir / final_name

        print(f"Packaging {final_name}...")

        # 1. Identify Assets
        assets = {
            "model_checkpoint": self.model_dir / "best_model.pth",
            "preprocessor": self.model_dir / "preprocessor.pkl",
            "inference_code": self.code_file,
        }

        # Verify existence
        for k, p in assets.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing required asset: {k} -> {p}")

        # 2. Calculate Hashes & Signatures
        manifest_files = {}

        for key, path in assets.items():
            with open(path, "rb") as f:
                content = f.read()

            file_hash = hashes.Hash(hashes.SHA256(), backend=default_backend())
            file_hash.update(content)
            digest = file_hash.finalize().hex()

            signature = self.signer.sign_bytes(content)

            # Use just the filename in the archive
            archive_name = path.name

            manifest_files[archive_name] = {
                "type": key,
                "sha256": digest,
                "signature": signature,
                "size_bytes": len(content),
            }

        # 3. Create Manifest
        manifest = {
            "package_name": package_name,
            "version": version,
            "architecture": "CrossSampleCNP",
            "created_at": datetime.now().isoformat(),
            "notes": notes,
            "files": manifest_files,
            "public_key": self.signer.get_public_key_pem(),
        }

        # 4. Write ZIP
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add Assets
            for key, path in assets.items():
                zf.write(path, arcname=path.name)

            # Add Manifest
            manifest_json = json.dumps(manifest, indent=2)
            zf.writestr("manifest.json", manifest_json)

            # Add Manifest Signature (Sign the manifest itself!)
            manifest_sig = self.signer.sign_bytes(manifest_json.encode("utf-8"))
            zf.writestr("manifest.sig", manifest_sig)

        print(f"✅ Package created successfully at: {output_path}")
        return output_path


# ==========================================
# 3. Execution Script
# ==========================================
def main():
    # Configuration
    MODEL_DIR = "models/experiments/o_net"  # Where train_visq.py saved outputs
    CODE_FILE = "ml/test/inference_o_net.py"  # The inference script we just wrote
    OUTPUT_DIR = "models/production"  # Where to put the .visq package

    # Check if files exist before running
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Model directory '{MODEL_DIR}' not found. Did you run training?")
        return
    if not os.path.exists(CODE_FILE):
        print(f"Error: Inference code '{CODE_FILE}' not found.")
        return

    # Initialize Packager
    packager = CNPModelPackager(
        output_dir=OUTPUT_DIR,
        model_source_dir=MODEL_DIR,
        code_source_file=CODE_FILE,
        # private_key_path="keys/my_private_key.pem" # Optional: Use real key if you have one
    )

    # Run
    packager.package(
        package_name="VisQ-ICL",
        version="1.0.0",
        notes="Initial release of Cross-Sample CNP model with memory caching.",
    )


if __name__ == "__main__":
    main()
