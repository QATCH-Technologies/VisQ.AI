"""Provide RSA-based cryptographic signing for deployment artifacts.

The :class:`ModuleSigner` class manages an RSA private key and uses
RSA-PSS with SHA-256 to produce signatures for files and in-memory byte
content. Signatures are returned in Base64-encoded form so they can be stored
in package manifests and other text-based metadata.

A caller may provide an existing PEM-encoded private key for reproducible
signing across packages. When no usable key is supplied, a new 2048-bit RSA
key is generated for the signer instance.

The corresponding public key can be exported in PEM format for inclusion in
a deployment artifact, allowing a verifier to authenticate signed package
contents without receiving the private key.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import cast
from contextlib import suppress
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class ModuleSigner:
    """Provide cryptographic signing utilities for VisQAI deployment artifacts.

    This module defines :class:`ModuleSigner`, which manages an RSA private key and
    provides signing and public-key export operations used to authenticate files
    included in secure model packages.

    Signatures are generated with RSA-PSS using SHA-256 and returned in Base64
    encoding for convenient storage in package manifests. A caller may supply an
    existing PEM-encoded private key to maintain a consistent signing identity
    across packages; otherwise, a new RSA key pair is generated for the signer.

    The private key is never included automatically in a deployment artifact.
    Only the corresponding public key and generated signatures are intended for
    distribution.
    """

    def __init__(self, private_key_path: str | None = None) -> None:
        """Initialize an RSA module signer.

        Args:
            private_key_path: Optional path to an existing PEM-encoded RSA private
                key. If the path is provided and exists, the key is loaded without a
                password. Otherwise, a new 2048-bit RSA private key is generated.
        """
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
        """Sign the contents of a file using RSA-PSS with SHA-256.

        Args:
            filepath: Path to the file whose complete binary contents should be
                signed.

        Returns:
            A Base64-encoded RSA-PSS signature suitable for storage in a text-based
            signature manifest.
        """
        with open(filepath, "rb") as f:
            content = f.read()

        signable_key = cast(rsa.RSAPrivateKey, self.private_key)

        signature = signable_key.sign(
            content,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def sign_bytes(self, content: bytes) -> str:
        """Sign an in-memory byte sequence using RSA-PSS with SHA-256.

        Args:
            content: Binary content to sign.

        Returns:
            A Base64-encoded RSA-PSS signature suitable for storage in a text-based
            signature manifest.
        """
        rsa_key = cast(rsa.RSAPrivateKey, self.private_key)
        signature = rsa_key.sign(
            content,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def get_public_key_pem(self) -> bytes:
        """Export the signer's public key in PEM-encoded SubjectPublicKeyInfo format.

        Returns:
            The PEM-encoded public key corresponding to the signer's private key.
            The returned key can be distributed with a deployment package for
            signature verification.
        """
        public_key = self.private_key.public_key()
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def save_private_key(self, filepath: str) -> None:
        """Write the signer's private key to a PEM-encoded file.

        Args:
            filepath: Destination path for the private key.

        Notes:
            The key is written without encryption. On platforms that support
            `os.chmod`, the file permissions are restricted to owner read/write
            access where possible. Callers are responsible for protecting the
            resulting file and should treat it as sensitive signing material.

            Permission-setting failures are ignored to preserve compatibility with
            platforms where POSIX-style file permissions are unavailable or
            limited.
        """
        pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        with open(filepath, "wb") as f:
            f.write(pem)

        with suppress(Exception):
            os.chmod(filepath, 0o600)
