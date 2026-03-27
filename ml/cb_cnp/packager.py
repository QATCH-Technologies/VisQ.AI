"""
packager_cb_cnp.py
====================
Packages all CBM-CNP model assets and code modules into a signed .visq archive.

What changed from packager_o_net.py
------------------------------------
1. **Data assets** — adds four CBM-specific concept proxy scalers
   (``concept_proxy_*.npy``).  These are optional: absent files are silently
   skipped so the same packager handles legacy CrossSampleCNP checkpoints.

2. **Architecture field** — was hardcoded ``"CrossSampleCNP"``.  Now read
   directly from ``best_model.pth`` via ``_read_checkpoint_metadata()`` so the
   manifest always reflects the actual saved class.

3. **CBM metadata block** — when the checkpoint is a ``ConceptBottleneckCNP``
   the manifest gains a ``"cbm_metadata"`` section containing ``n_concepts``,
   ``concept_names``, ``concept_activations``, and ``concept_gate_values``.
   The loader can surface these without unpickling the checkpoint.

4. **Module load ordering** — the new modular system has real import
   dependencies between its files.  Load order is now explicit and documented:

       constants.py        (no inter-module deps — must be first)
       models.py           (imports constants)
       data_pipeline.py    (imports constants)
       batch_utils.py      (no package deps — pure torch)
       inference_cb_cnp.py (imports models, data_pipeline)

   The manifest stores this as ``modules.load_order`` so the app loader can
   exec/import them in the right sequence without inspecting the source.

5. **Entry point** — updated to ``inference_cb_cnp.py:ViscosityPredictorCNP``.

The ``ModuleSigner`` class and the ZIP-signing protocol are unchanged.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-27

Version:
    3.0 (CBM modular system support)
"""

import base64
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


# ============================================================
# 1. Security — RSA signing (unchanged from v2)
# ============================================================


class ModuleSigner:
    """RSA-based signing for secure package verification."""

    def __init__(self, private_key_path: Optional[str] = None) -> None:
        if private_key_path and os.path.exists(private_key_path):
            with open(private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
            print(f"Loaded private key from {private_key_path}")
        else:
            print("Generating NEW private key (demo only — store this key for production)...")
            self.private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )

    def sign_bytes(self, data: bytes) -> str:
        """Sign bytes and return a base64-encoded signature string."""
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
        pem = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return pem.decode("utf-8")


# ============================================================
# 2. Checkpoint introspection helper
# ============================================================


def _read_checkpoint_metadata(model_path: Path) -> Dict[str, Any]:
    """
    Load only the metadata fields from a model checkpoint without
    instantiating the full PyTorch model.

    Reads: ``model_class``, ``n_concepts``, ``concept_names``,
    ``concept_activations``, ``concept_gate_values``, ``config``,
    ``static_dim``.

    Parameters
    ----------
    model_path : Path

    Returns
    -------
    dict with the above keys (absent keys are omitted).
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    meta: Dict[str, Any] = {}

    for key in (
        "model_class",
        "n_concepts",
        "concept_names",
        "concept_activations",
        "concept_gate_values",
        "config",
        "static_dim",
    ):
        if key in checkpoint:
            meta[key] = checkpoint[key]

    return meta


# ============================================================
# 3. Packager
# ============================================================


class CNPModelPackager:
    """
    Creates signed .visq packages for the CBM-CNP inference system.

    The archive contains:
      - Model weights and preprocessors  (``data_assets``)
      - CBM concept proxy scalers        (``cbm_assets``, optional)
      - Ordered Python source modules    (``code_sources``)
      - A self-describing ``manifest.json`` and its RSA signature

    Parameters
    ----------
    output_dir : str
        Directory where the .visq archive will be written.
    model_source_dir : str
        Directory produced by ``train.py`` (contains best_model.pth etc.).
    code_source_files : list[dict | str], optional
        Python modules to bundle.  Each entry is either a dict with keys
        ``"path"`` and ``"type"``, or a bare path string (treated as
        ``"inference_code"`` for backward compatibility).
        **List order defines the loader's import sequence.**
    entry_point_class : str, optional
        ``"filename.py:ClassName"`` the loader should instantiate.
        Auto-detected from the first ``inference_code`` module if omitted.
    private_key_path : str, optional
        Path to a PEM RSA private key.  A fresh key is generated if absent.
    code_source_file : str, optional
        Legacy single-file shorthand (v2 backward compatibility).
    """

    def __init__(
        self,
        output_dir: str,
        model_source_dir: str,
        code_source_files: Any = None,
        entry_point_class: Optional[str] = None,
        private_key_path: Optional[str] = None,
        # backward-compat single-file shorthand
        code_source_file: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_source_dir)
        self.signer = ModuleSigner(private_key_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Normalise code_source_files → List[Dict]
        self.code_sources: List[Dict[str, str]] = []
        if code_source_files:
            for entry in code_source_files:
                if isinstance(entry, str):
                    entry = {"path": entry, "type": "inference_code"}
                self.code_sources.append(entry)
        elif code_source_file:
            self.code_sources.append({"path": code_source_file, "type": "inference_code"})

        self.entry_point_class = entry_point_class

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _collect_data_assets(self) -> Tuple[Dict[str, Path], Dict[str, Path]]:
        """
        Separate model assets into required and optional groups.

        Required assets raise ``FileNotFoundError`` if absent.
        Optional CBM assets (concept proxy scalers) are silently skipped —
        they are only written by ``train.py`` for ``ConceptBottleneckCNP``
        checkpoints.

        Returns
        -------
        required : dict {archive_name: Path}
        optional : dict {archive_name: Path}  (only paths that exist)
        """
        required_specs = {
            "best_model.pth": self.model_dir / "best_model.pth",
            "preprocessor.pkl": self.model_dir / "preprocessor.pkl",
            "physics_scaler.pkl": self.model_dir / "physics_scaler.pkl",
        }
        optional_specs = {
            "concept_proxy_mean.npy": self.model_dir / "concept_proxy_mean.npy",
            "concept_proxy_std.npy": self.model_dir / "concept_proxy_std.npy",
            "concept_proxy_signs.npy": self.model_dir / "concept_proxy_signs.npy",
            "concept_proxy_activations.npy": self.model_dir / "concept_proxy_activations.npy",
        }

        required: Dict[str, Path] = {}
        for name, path in required_specs.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Required model asset missing: {name}\n"
                    f"  Expected at: {path}\n"
                    f"  Run train.py to generate all assets before packaging."
                )
            required[name] = path

        optional: Dict[str, Path] = {}
        for name, path in optional_specs.items():
            if path.exists():
                optional[name] = path
            else:
                print(f"  (optional) {name} not found — skipping (baseline CNP checkpoint).")

        return required, optional

    def _collect_code_assets(self) -> Tuple[Dict[str, Path], List[str]]:
        """
        Validate that all declared code modules exist on disk.

        Returns
        -------
        code_paths  : dict {filename: Path}
        load_order  : list[str]  — archive filenames in declaration order
        """
        code_paths: Dict[str, Path] = {}
        load_order: List[str] = []
        for entry in self.code_sources:
            p = Path(entry["path"])
            if not p.exists():
                raise FileNotFoundError(f"Code module not found: {entry['type']} -> {p}")
            code_paths[p.name] = p
            load_order.append(p.name)
        return code_paths, load_order

    def _build_cbm_metadata(self, checkpoint_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract CBM-specific fields for the manifest.

        Returns ``None`` for baseline ``CrossSampleCNP`` checkpoints so the
        manifest omits the section entirely rather than storing empty values.
        """
        if checkpoint_meta.get("model_class") != "ConceptBottleneckCNP":
            return None
        return {
            "n_concepts": checkpoint_meta.get("n_concepts"),
            "concept_names": checkpoint_meta.get("concept_names"),
            "concept_activations": checkpoint_meta.get("concept_activations"),
            "concept_gate_values": checkpoint_meta.get("concept_gate_values"),
        }

    def _hash_and_sign(
        self,
        all_assets: Dict[str, Path],
        asset_types: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Compute SHA-256 hash and RSA signature for every asset file.

        Returns the ``files`` block written into ``manifest.json``.
        """
        manifest_files: Dict[str, Any] = {}
        for archive_name, path in all_assets.items():
            content = path.read_bytes()

            h = hashes.Hash(hashes.SHA256(), backend=default_backend())
            h.update(content)
            digest = h.finalize().hex()

            manifest_files[archive_name] = {
                "type": asset_types[archive_name],
                "sha256": digest,
                "signature": self.signer.sign_bytes(content),
                "size_bytes": len(content),
            }
        return manifest_files

    def _resolve_entry_point(self, load_order: List[str]) -> Dict[str, str]:
        """
        Determine the manifest entry_point dict.

        Explicit ``"filename.py:ClassName"`` takes precedence; otherwise the
        first declared ``inference_code`` module is used with the class left
        for the loader to auto-detect.
        """
        if self.entry_point_class:
            module_file, class_name = self.entry_point_class.split(":")
            return {"module": module_file, "class": class_name}

        if load_order:
            first_inference = next(
                (Path(e["path"]).name for e in self.code_sources if "inference" in e["type"]),
                load_order[0],
            )
            return {"module": first_inference}

        return {}

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def package(self, package_name: str, version: str, notes: str = "") -> Path:
        """
        Build and write a signed .visq archive.

        Parameters
        ----------
        package_name : str   Human-readable name embedded in the manifest.
        version      : str   Semantic version string (e.g. ``"3.0.0"``).
        notes        : str   Optional release notes.

        Returns
        -------
        Path  Path to the generated .visq file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_name = f"{package_name}_v{version}_{timestamp}.visq"
        output_path = self.output_dir / final_name
        print(f"Packaging {final_name}...")

        # --------------------------------------------------
        # Step 1: Collect and validate all assets
        # --------------------------------------------------
        required_assets, optional_assets = self._collect_data_assets()
        code_paths, load_order = self._collect_code_assets()

        # --------------------------------------------------
        # Step 2: Read architecture from checkpoint
        # --------------------------------------------------
        checkpoint_meta = _read_checkpoint_metadata(required_assets["best_model.pth"])
        architecture = checkpoint_meta.get("model_class", "CrossSampleCNP")
        cbm_meta = self._build_cbm_metadata(checkpoint_meta)

        print(f"  Architecture: {architecture}")
        if cbm_meta:
            print(
                f"  CBM: {cbm_meta['n_concepts']} concepts "
                f"({len([g for g in (cbm_meta.get('concept_gate_values') or []) if g > 0.5])} open gates)"
            )

        # --------------------------------------------------
        # Step 3: Flatten into a single asset registry for signing
        # --------------------------------------------------
        all_assets: Dict[str, Path] = {}
        asset_types: Dict[str, str] = {}

        # Required model assets
        for name, path in required_assets.items():
            all_assets[name] = path
            asset_types[name] = "model_checkpoint" if name.endswith(".pth") else "preprocessor"

        # Optional CBM concept proxy scalers
        for name, path in optional_assets.items():
            all_assets[name] = path
            asset_types[name] = "cbm_scaler"

        # Code modules (in load_order sequence)
        for entry in self.code_sources:
            p = Path(entry["path"])
            all_assets[p.name] = p
            asset_types[p.name] = entry["type"]

        # --------------------------------------------------
        # Step 4: Hash and sign every asset
        # --------------------------------------------------
        manifest_files = self._hash_and_sign(all_assets, asset_types)

        # --------------------------------------------------
        # Step 5: Assemble manifest
        # --------------------------------------------------
        manifest: Dict[str, Any] = {
            "package_name": package_name,
            "version": version,
            "architecture": architecture,
            "created_at": datetime.now().isoformat(),
            "notes": notes,
            "files": manifest_files,
            "modules": {
                # load_order tells the app loader the exact sequence in which
                # to exec/import the code modules so cross-file imports resolve.
                "load_order": load_order,
                "entry_point": self._resolve_entry_point(load_order),
            },
            "public_key": self.signer.get_public_key_pem(),
        }

        # CBM metadata block — omitted entirely for baseline CNP packages
        if cbm_meta is not None:
            manifest["cbm_metadata"] = cbm_meta

        # --------------------------------------------------
        # Step 6: Write ZIP
        # --------------------------------------------------
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for archive_name, path in all_assets.items():
                zf.write(path, arcname=archive_name)

            manifest_json = json.dumps(manifest, indent=2)
            zf.writestr("manifest.json", manifest_json)
            zf.writestr("manifest.sig", self.signer.sign_bytes(manifest_json.encode("utf-8")))

        print(f"Package created: {output_path}")
        print(
            f"   {len(all_assets)} assets | "
            f"{len(load_order)} code modules | "
            f"architecture: {architecture}"
        )
        return output_path


# ============================================================
# 4. Execution script
# ============================================================


def main() -> None:
    """
    Package the CBM-CNP model.

    Module load ordering is intentional and must be preserved:

        constants.py         no inter-module deps — imported by models + data_pipeline
        models.py            imports constants
        data_pipeline.py     imports constants
        batch_utils.py       no package deps (pure torch)
        inference_cb_cnp.py imports models, data_pipeline

    The loader will exec these files in this order so that by the time
    inference_cb_cnp.py runs, all its sibling imports are already available.
    """
    MODEL_DIR = "models/experiments/cbm_cnp_v4"
    OUTPUT_DIR = "models/production"

    if not os.path.exists(MODEL_DIR):
        print(
            f"Error: Model directory '{MODEL_DIR}' not found. "
            "Run train.py first to generate all assets."
        )
        return

    # -------------------------------------------------------
    # Code modules in load order.
    # Type values are embedded in the manifest and can be read
    # by the loader to distinguish runtime roles.
    # -------------------------------------------------------
    code_modules = [
        # Foundation: no inter-module deps, must be first
        {
            "path": "ml/cb_cnp/constants.py",
            "type": "config",
        },
        # Architecture: depends on constants
        {
            "path": "ml/cb_cnp/models.py",
            "type": "model_definition",
        },
        # Feature engineering: depends on constants
        {
            "path": "ml/cb_cnp/data_pipeline.py",
            "type": "data_pipeline",
        },
        # Tensor helpers: no package deps
        {
            "path": "ml/cb_cnp/batch_utils.py",
            "type": "utils",
        },
        # Entry point: imports models + data_pipeline — must be last
        {
            "path": "ml/cb_cnp/inference.py",
            "type": "inference_code",
        },
    ]

    for mod in code_modules:
        if not os.path.exists(mod["path"]):
            print(f"Error: Code module '{mod['path']}' not found.")
            return

    packager = CNPModelPackager(
        output_dir=OUTPUT_DIR,
        model_source_dir=MODEL_DIR,
        code_source_files=code_modules,
        entry_point_class="inference.py:InferenceCNP",
    )

    packager.package(
        package_name="VisQAI",
        version="3.0.0",
        notes=(
            "CBM-CNP modular system: constants, models, data_pipeline, "
            "batch_utils, inference. Adds concept state and intervention endpoints."
        ),
    )


if __name__ == "__main__":
    main()
