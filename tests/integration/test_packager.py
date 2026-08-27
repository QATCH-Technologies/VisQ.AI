"""
Light integration test for SecurePredictorPackager: builds a package against
a tiny dummy checkpoint + the real source-module list, and confirms the zip
contains the expected member names and a validating signature.

Also covers get_latest_checkpoints's dated <date>/<time> discovery (visqai.
paths.latest_checkpoint_dir) and the packager's edge cases/error paths that
the original single happy-path test left uncovered.
"""

from __future__ import annotations

import base64
import json
import os
import time
import zipfile
from pathlib import Path

import pytest
import torch
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from visqai import constants
from visqai.packaging.packager import SecurePredictorPackager, DEFAULT_SOURCE_MODULES, get_latest_checkpoints

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = str(REPO_ROOT / "src" / "visqai")


@pytest.fixture
def dummy_checkpoint(tmp_path):
    ckpt_path = tmp_path / "dummy_checkpoint.pth"
    torch.save({"state_dict": {}, "config": {"hidden_dim": 8}, "static_dim": 4}, ckpt_path)
    return ckpt_path


def _verify_signature(public_key, content: bytes, sig_b64: str) -> None:
    signature = base64.b64decode(sig_b64)
    public_key.verify(
        signature,
        content,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )  # raises InvalidSignature if it doesn't validate


def test_package_contains_expected_members_and_valid_signatures(tmp_path, dummy_checkpoint):
    out_dir = tmp_path / "out"
    packager = SecurePredictorPackager(output_dir=str(out_dir), source_dir=SOURCE_DIR, enable_signing=True)
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
            _verify_signature(public_key, zf.read(member), sig_b64)


def test_package_single_vs_ensemble_naming(tmp_path, dummy_checkpoint):
    out_dir = tmp_path / "out"
    packager = SecurePredictorPackager(output_dir=str(out_dir), source_dir=SOURCE_DIR, enable_signing=False)
    zip_path = packager.package(model_paths=str(dummy_checkpoint))
    assert "single" in Path(zip_path).name
    assert Path(zip_path).suffix == ".visq"


def test_package_name_already_ending_in_visq_is_not_double_suffixed(tmp_path, dummy_checkpoint):
    out_dir = tmp_path / "out"
    packager = SecurePredictorPackager(output_dir=str(out_dir), source_dir=SOURCE_DIR, enable_signing=False)
    zip_path = packager.package(model_paths=str(dummy_checkpoint), package_name="already_named.visq")
    assert Path(zip_path).name == "already_named.visq"


def test_output_dir_defaults_to_a_dated_run_dir(monkeypatch, tmp_path, dummy_checkpoint):
    """output_dir=None (the default) should auto-generate <production_root>/
    <date>/<time> rather than requiring a hand-picked location."""
    fake_production_root = tmp_path / "production"
    monkeypatch.setattr(constants, "PRODUCTION_DIR", fake_production_root)

    packager = SecurePredictorPackager(source_dir=SOURCE_DIR, enable_signing=False)

    assert packager.output_dir.parent.parent == fake_production_root
    assert packager.output_dir.exists()  # __init__ creates it

    zip_path = packager.package(model_paths=str(dummy_checkpoint))
    assert Path(zip_path).parent == packager.output_dir


def test_ensemble_package_writes_one_checkpoint_per_model(tmp_path, dummy_checkpoint):
    second_checkpoint = tmp_path / "dummy_checkpoint_2.pth"
    torch.save({"state_dict": {}, "config": {"hidden_dim": 8}, "static_dim": 4}, second_checkpoint)

    out_dir = tmp_path / "out"
    packager = SecurePredictorPackager(output_dir=str(out_dir), source_dir=SOURCE_DIR, enable_signing=True)
    zip_path = packager.package(model_paths=[str(dummy_checkpoint), str(second_checkpoint)])

    assert "ensemble" in Path(zip_path).name
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "model/checkpoint_0.pt" in names
        assert "model/checkpoint_1.pt" in names
        assert "model/checkpoint.pt" not in names  # single-model naming not used

        metadata = json.loads(zf.read("model/metadata.json"))
        assert metadata["model_type"] == "ensemble"
        assert metadata["n_models"] == 2


def test_package_raises_on_missing_model_file(tmp_path):
    out_dir = tmp_path / "out"
    packager = SecurePredictorPackager(output_dir=str(out_dir), source_dir=SOURCE_DIR, enable_signing=False)
    with pytest.raises(FileNotFoundError):
        packager.package(model_paths=str(tmp_path / "does_not_exist.pth"))


def test_init_raises_on_missing_source_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        SecurePredictorPackager(output_dir=str(tmp_path / "out"), source_dir=str(tmp_path / "no_such_src"))


def test_missing_source_file_is_skipped_not_fatal(tmp_path, dummy_checkpoint):
    out_dir = tmp_path / "out"
    packager = SecurePredictorPackager(
        output_dir=str(out_dir),
        source_dir=SOURCE_DIR,
        source_files=["features/categorical.py", "does/not/exist.py"],
        enable_signing=False,
    )
    zip_path = packager.package(model_paths=str(dummy_checkpoint))
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "src/features/categorical.py" in names
        assert "src/does/not/exist.py" not in names


def test_enable_signing_false_omits_security_files(tmp_path, dummy_checkpoint):
    out_dir = tmp_path / "out"
    packager = SecurePredictorPackager(output_dir=str(out_dir), source_dir=SOURCE_DIR, enable_signing=False)
    zip_path = packager.package(model_paths=str(dummy_checkpoint))
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "security/public_key.pem" not in names
        assert "security/signatures.json" not in names
        metadata = json.loads(zf.read("model/metadata.json"))
        assert metadata["cryptographically_signed"] is False


def test_save_private_key_writes_a_loadable_key(tmp_path, dummy_checkpoint):
    out_dir = tmp_path / "out"
    key_path = tmp_path / "private_key.pem"
    packager = SecurePredictorPackager(output_dir=str(out_dir), source_dir=SOURCE_DIR, enable_signing=True)
    packager.package(model_paths=str(dummy_checkpoint), save_private_key=str(key_path))

    assert key_path.exists()
    with open(key_path, "rb") as f:
        # Raises if it isn't a valid, loadable PEM private key.
        serialization.load_pem_private_key(f.read(), password=None)


def test_reusing_a_saved_private_key_reproduces_verifiable_signatures(tmp_path, dummy_checkpoint):
    """SecurePredictorPackager(private_key_path=...) should load the same
    key rather than generating a fresh one each time."""
    out_dir = tmp_path / "out"
    key_path = tmp_path / "private_key.pem"
    first = SecurePredictorPackager(output_dir=str(out_dir), source_dir=SOURCE_DIR, enable_signing=True)
    first.package(model_paths=str(dummy_checkpoint), package_name="first", save_private_key=str(key_path))
    first_public_pem = first.signer.get_public_key_pem()

    second = SecurePredictorPackager(
        output_dir=str(out_dir), source_dir=SOURCE_DIR, enable_signing=True, private_key_path=str(key_path)
    )
    assert second.signer.get_public_key_pem() == first_public_pem


def test_metadata_defaults_when_optional_fields_omitted(tmp_path, dummy_checkpoint):
    out_dir = tmp_path / "out"
    packager = SecurePredictorPackager(output_dir=str(out_dir), source_dir=SOURCE_DIR, enable_signing=False)
    zip_path = packager.package(model_paths=str(dummy_checkpoint))
    with zipfile.ZipFile(zip_path) as zf:
        metadata = json.loads(zf.read("model/metadata.json"))
    assert metadata["version"] == "1.0"
    assert metadata["client"] == "Unknown"
    assert metadata["author"] == "Unknown"
    assert metadata["model_created_date"] == "Unknown"
    assert metadata["notes"] == ""


# --------------------------------------------------------------------------
# get_latest_checkpoints: the <date>/<time> nested discovery logic (visqai.
# paths.latest_checkpoint_dir), previously untested.
# --------------------------------------------------------------------------

def _make_checkpoint_dir(root: Path, date: str, time_: str, filenames):
    d = root / date / time_
    d.mkdir(parents=True)
    for name in filenames:
        (d / name).write_bytes(b"not a real checkpoint")
    return d


def test_get_latest_checkpoints_picks_the_most_recent_time_within_the_most_recent_date(tmp_path):
    root = tmp_path / "checkpoints"
    _make_checkpoint_dir(root, "2026-08-26", "09-00-00", ["best_model.pth"])
    time.sleep(0.01)
    older_today = _make_checkpoint_dir(root, "2026-08-27", "08-00-00", ["best_model.pth"])
    time.sleep(0.01)
    newest = _make_checkpoint_dir(root, "2026-08-27", "14-32-05", ["best_model.pth"])

    result = get_latest_checkpoints(str(root))
    assert len(result) == 1
    assert Path(result[0]).parent == newest
    assert Path(result[0]).parent != older_today


def test_get_latest_checkpoints_finds_both_pt_and_pth(tmp_path):
    root = tmp_path / "checkpoints"
    d = _make_checkpoint_dir(root, "2026-08-27", "10-00-00", ["a.pt", "b.pth", "notes.txt"])

    result = {Path(p).name for p in get_latest_checkpoints(str(root))}
    assert result == {"a.pt", "b.pth"}


def test_get_latest_checkpoints_raises_when_root_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_latest_checkpoints(str(tmp_path / "no_such_dir"))


def test_get_latest_checkpoints_raises_when_no_dated_subdirs(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        get_latest_checkpoints(str(root))


def test_get_latest_checkpoints_raises_when_latest_dir_has_no_checkpoint_files(tmp_path):
    root = tmp_path / "checkpoints"
    _make_checkpoint_dir(root, "2026-08-27", "10-00-00", ["metadata.json"])  # no .pt/.pth
    with pytest.raises(FileNotFoundError):
        get_latest_checkpoints(str(root))
