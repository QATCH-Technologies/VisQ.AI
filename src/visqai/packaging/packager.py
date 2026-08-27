"""Package a trained VisQAI predictor for secure client-side deployment.

The packager creates a `.visq` ZIP artifact containing the trained model
checkpoint(s), the runtime inference source modules required to execute the
predictor, optional dependency and documentation files, model metadata, and
cryptographic signatures.

Only runtime-inference code is included by default. Training, evaluation, and
CLI modules are intentionally excluded because they are not required by a
deployed predictor and should not be distributed as part of the client
runtime.

The package layout is organized as follows::

    model/
        checkpoint.pt              # Single-model package
        checkpoint_0.pt            # Ensemble package
        checkpoint_1.pt
        metadata.json
    src/
        <runtime inference modules>
    requirements.txt               # If present
    README.md                      # If present
    security/
        public_key.pem             # When signing is enabled
        signatures.json             # When signing is enabled

Source modules are signed individually, as are model checkpoints, metadata,
and optional package-level dependency/documentation files. This allows the
deployment runtime to verify the integrity and provenance of packaged
artifacts before use.

The default runtime module set is defined by `DEFAULT_SOURCE_MODULES` and
can be overridden when constructing :class:`SecurePredictorPackager`.
"""

from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime
from pathlib import Path

from visqai import constants, paths
from visqai.packaging.signer import ModuleSigner

"""Default runtime-inference modules included in deployment packages.

Paths are relative to the configured `source_dir` and are written beneath
the package's `src/` directory. The list intentionally excludes training,
evaluation, and command-line modules.
"""
DEFAULT_SOURCE_MODULES = [
    "features/categorical.py",
    "features/charge.py",
    "features/priors.py",
    "features/dataprocessor.py",
    "models/cnp.py",
    "inference/predictor.py",
]


class SecurePredictorPackager:
    """Build signed VisQAI model deployment packages.

    This module provides :class:`SecurePredictorPackager`, which assembles trained
    VisQAI model checkpoints and the runtime source modules required for
    inference into a self-contained `.visq` ZIP artifact.

    A package may contain either a single checkpoint or multiple checkpoints
    forming an ensemble. Runtime source modules, dependency specifications,
    documentation, model metadata, and cryptographic verification artifacts can
    also be included. When signing is enabled, each packaged artifact is signed
    individually using :class:`visqai.packaging.signer.ModuleSigner`, with the
    corresponding public key and signature manifest stored in the package.

    Only code required to execute the deployed predictor is included by default.
    Training, evaluation, experimentation, and CLI modules are intentionally
    excluded to keep the deployment artifact focused on its inference-time
    runtime requirements.

    The module also provides :func:`get_latest_checkpoints` for locating model
    checkpoints in the most recently generated checkpoint directory.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        source_dir: str = "src/visqai",
        source_files: list[str] | None = None,
        requirements_path: str = "requirements.txt",
        readme_path: str = "README.md",
        private_key_path: str | None = None,
        enable_signing: bool = True,
    ) -> None:
        """Initialize a secure VisQAI predictor packager.

        Args:
            output_dir: Directory in which generated `.visq` artifacts are written.
                When `None`, a new dated production directory is created using
                :func:`visqai.paths.dated_run_dir`.
            source_dir: Root directory containing the runtime-inference source
                modules.
            source_files: Relative source-module paths to include in the package. If
                `None`, :data:`DEFAULT_SOURCE_MODULES` is used.
            requirements_path: Path to the dependency specification to include when
                it exists.
            readme_path: Path to the deployment README to include when it exists.
            private_key_path: Optional path to an existing private signing key used by
                :class:`ModuleSigner`.
            enable_signing: Whether to cryptographically sign packaged artifacts.
                When enabled, a :class:`ModuleSigner` is created and security files
                are included in the package.

        Raises:
            FileNotFoundError: If `source_dir` does not exist or is not a directory.
        """
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else paths.dated_run_dir(constants.PRODUCTION_DIR)
        )
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
        model_paths: str | list[str],
        package_name: str | None = None,
        model_created_date: str | None = None,
        client: str | None = None,
        author: str | None = None,
        version: str | None = None,
        notes: str | None = None,
        save_private_key: str | None = None,
        is_ensemble: bool | None = None,
    ) -> str:
        """Create a signed deployment package from one or more model checkpoints.

        Args:
            model_paths: Path to a model checkpoint or a list of checkpoint paths.
                Multiple checkpoints are packaged as an ensemble.
            package_name: Optional output package filename. A timestamped name is
                generated when omitted. The `.visq` suffix is added when absent.
            model_created_date: Optional date associated with model creation.
            client: Optional client or deployment recipient identifier stored in
                package metadata.
            author: Optional model author stored in package metadata.
            version: Optional package or model version stored in metadata. Defaults
                to `"1.0"` when omitted.
            notes: Optional free-form deployment notes stored in metadata.
            save_private_key: Optional path at which the signing private key should
                be saved after packaging.
            is_ensemble: Whether to package the checkpoints as an ensemble. When
                `None`, this is inferred from whether more than one checkpoint was
                supplied.

        Returns:
            The filesystem path of the generated `.visq` package.

        Raises:
            FileNotFoundError: If any supplied model checkpoint does not exist.

        Notes:
            When signing is enabled, each model checkpoint, included source module,
            optional dependency/documentation file, and generated metadata file is
            individually signed. The corresponding public key and signature manifest
            are stored under `security/`.
        """
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
        if not package_name.endswith(".visq"):
            package_name += ".visq"

        zip_path = self.output_dir / package_name
        signatures: dict[str, str] = {}
        contents_list: list[str] = []

        print(f"Creating {'ensemble' if is_ensemble else 'single'} model package...")
        print(f"Source Directory: {self.source_dir}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for i, model_path in enumerate(model_paths):
                model_filename = (
                    f"model/checkpoint_{i}.pt" if is_ensemble else "model/checkpoint.pt"
                )
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
                signatures["model/metadata.json"] = self.signer.sign_bytes(
                    metadata_json.encode("utf-8")
                )
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
        model_paths: list[str],
        is_ensemble: bool,
        contents: list[str],
        model_created_date: str | None = None,
        client: str | None = None,
        author: str | None = None,
        version: str | None = None,
        notes: str | None = None,
    ) -> dict[str, object]:
        """Create the metadata manifest for a deployment package.

        Args:
            model_paths: Checkpoint paths included in the package.
            is_ensemble: Whether the package contains multiple ensemble models.
            contents: Package-relative paths of all artifacts included before the
                metadata file is added.
            model_created_date: Optional model creation date.
            client: Optional client identifier.
            author: Optional model author.
            version: Optional package version.
            notes: Optional deployment notes.

        Returns:
            A dictionary containing package versioning, model type and count,
            timestamps, model size, signing status, architecture identifier, and the
            package-relative contents list.
        """
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

    def _add_security_files(self, zipf: zipfile.ZipFile, signatures: dict[str, str]) -> None:
        """Add cryptographic verification artifacts to a deployment package.

        Args:
            zipf: Open ZIP archive to which the security files are written.
            signatures: Mapping from package-relative artifact paths to their
                cryptographic signatures.

        The method writes the signer's public key to `security/public_key.pem` and
        the serialized artifact-to-signature mapping to
        `security/signatures.json`.
        """
        public_key_pem = self.signer.get_public_key_pem()
        zipf.writestr("security/public_key.pem", public_key_pem)
        signatures_json = json.dumps(signatures, indent=2)
        zipf.writestr("security/signatures.json", signatures_json)


def get_latest_checkpoints(experiments_dir: str | Path = constants.CHECKPOINTS_DIR) -> list[str]:
    """Locate checkpoints from the most recently produced checkpoint directory.

    Args:
        experiments_dir: Root directory containing dated checkpoint runs. By
            default, uses :data:`visqai.constants.CHECKPOINTS_DIR`.

    Returns:
        A list of paths to `.pt` and `.pth` checkpoint files in the latest
        checkpoint directory.

    Raises:
        FileNotFoundError: If the latest checkpoint directory contains no
            supported checkpoint files.

    Notes:
        Directory selection is delegated to
        :func:`visqai.paths.latest_checkpoint_dir`, which determines the most
        recent dated checkpoint run rather than relying on a manually specified
        checkpoint filename.
    """
    latest_dir = paths.latest_checkpoint_dir(experiments_dir)
    print(f"Located latest checkpoint: {latest_dir.relative_to(Path(experiments_dir))}")

    checkpoints = list(latest_dir.glob("*.pt")) + list(latest_dir.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt/.pth files found in {latest_dir}")

    return [str(cp) for cp in checkpoints]
