"""
Light integration test for SecurePredictorPackager: builds a package against
a tiny dummy checkpoint + the real source-module list, and confirms the zip
contains the expected member names and a validating signature.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from visqai.packaging.packager import SecurePredictorPackager, DEFAULT_SOURCE_MODULES

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def dummy_checkpoint(tmp_path):
    ckpt_path = tmp_path / "dummy_checkpoint.pth"
    torch.save({"state_dict": {}, "config": {"hidden_dim": 8}, "static_dim": 4}, ckpt_path)
    return ckpt_path


def test_package_contains_expected_members_and_valid_signatures(tmp_path, dummy_checkpoint):
    out_dir = tmp_path / "out"
    packager = SecurePredictorPackager(
        output_dir=str(out_dir),
        source_dir=str(REPO_ROOT / "src" / "visqai"),
        enable_signing=True,
    )
    zip_path = packager.package(
        model_paths=str(dummy_checkpoint),
        package_name="test_package",
        version="0.0-test",
        author="pytest",
    )

    assert Path(zip_path).exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "model/checkpoint.pt" in names
        assert "model/metadata.json" in names
        assert "security/public_key.pem" in names
        assert "security/signatures.json" in names
        for module in DEFAULT_SOURCE_MODULES:
            assert f"src/{module}" in names

        metadata = json.loads(zf.read("model/metadata.json"))
        assert metadata["version"] == "0.0-test"
        assert metadata["author"] == "pytest"
        assert metadata["cryptographically_signed"] is True

        public_key = serialization.load_pem_public_key(zf.read("security/public_key.pem"))
        signatures = json.loads(zf.read("security/signatures.json"))

        for member, sig_b64 in signatures.items():
            content = zf.read(member)
            import base64

            signature = base64.b64decode(sig_b64)
            public_key.verify(
                signature,
                content,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )  # raises InvalidSignature if it doesn't validate


def test_package_single_vs_ensemble_naming(tmp_path, dummy_checkpoint):
    out_dir = tmp_path / "out"
    packager = SecurePredictorPackager(
        output_dir=str(out_dir), source_dir=str(REPO_ROOT / "src" / "visqai"), enable_signing=False
    )
    zip_path = packager.package(model_paths=str(dummy_checkpoint))
    assert "single" in Path(zip_path).name
