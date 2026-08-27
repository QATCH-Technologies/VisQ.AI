"""Shared configuration and constants for the VisQAI package.

This module centralizes values that are intentionally shared across multiple
VisQAI subsystems, including shear-rate schema definitions, plotting style
tokens, training defaults, descriptor out-of-distribution clipping, and
machine-independent data/model storage roots.

Constants that represent empirical calibration for a single algorithm remain
in the module that consumes them so their associated rationale and behavioral
context stay local. Likewise, private implementation constants and
function-reference mappings are kept out of this module.

Storage roots are resolved dynamically per machine. Explicit environment
variable overrides take precedence, followed by Dropbox's configured mount
point when available, with a per-user default path used as a final fallback.
Importing this module therefore does not require Dropbox to be installed or
configured.

The module is intended to provide a single source of truth for shared
configuration rather than to contain application logic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

"""Define the canonical shear-rate and viscosity column schema shared across
data processing, evaluation, and visualization.

The full five-rate schema is represented both as ordered column/rate sequences
and as a column-to-rate mapping so consumers can use whichever representation
best matches their operation. Human-readable labels are also provided for
plotting.

The narrower `VISC_COLS` / `PRED_COLS` subset is intentionally retained
separately for metrics and convergence analyses that operate only at selected
shear rates; it is not an alternative definition of the full viscosity
spectrum.
"""
SHEAR_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]
SHEAR_RATES = [100.0, 1000.0, 10000.0, 100000.0, 15000000.0]
N_SHEARS = len(SHEAR_COLS)

SHEAR_MAP = dict(zip(SHEAR_COLS, SHEAR_RATES))

# Long-form labels (parity/profile plots).
SHEAR_LABELS = {
    "Viscosity_100": "100 s⁻¹",
    "Viscosity_1000": "1 000 s⁻¹",
    "Viscosity_10000": "10 000 s⁻¹",
    "Viscosity_100000": "100 000 s⁻¹",
    "Viscosity_15000000": "15 000 000 s⁻¹",
}

# Short-form labels, positional (convergence/shape plots).
SHORT_SHEAR_LABELS = ["100", "1k", "10k", "100k", "15M"]

SHEAR_COLORS = {
    "Viscosity_100": "#2596be",
    "Viscosity_1000": "#17a589",
    "Viscosity_10000": "#e67e22",
    "Viscosity_100000": "#e74c3c",
    "Viscosity_15000000": "#7d3c98",
}

"""Define the narrower viscosity columns used by single-rate analyses.

This subset is intentionally distinct from `SHEAR_COLS`: it contains only
the shear-rate measurements required by convergence-replay scalar metrics,
`features.dataprocessor.prepare_df` row validation, and
`eval.metrics`. The corresponding `PRED_COLS` names are the expected
prediction columns for those same measurements.

The current configuration uses the 1,000 s⁻¹ viscosity measurement because
the convergence analysis historically tracked that rate specifically.
"""
VISC_COLS = [
    "Viscosity_1000",
]
PRED_COLS = [f"Pred_{c}" for c in VISC_COLS]

"""Define the canonical visual palette and font used by VisQAI plots.

The plotting implementation remains in :mod:`visqai.eval.style`; this section
contains only the shared visual values that those functions apply.

The palette consolidates values that were previously maintained independently
by the parity-test and learning-curve plotting code. The `learning_curve_ibal`
palette is treated as canonical because it provides the more complete set of
accent colors and typography tokens.

The consolidated palette includes some cosmetic shade changes relative to the
former parity-test implementation. These changes affect rendered appearance
but do not affect model behavior, evaluation results, or numerical correctness.
"""

C_DEEP_BLUE = "#2596be"
C_BRIGHT_BLUE = "#13B5F0"
C_CYAN_MED = "#4EC4EB"
C_CYAN_PALE = "#8DD9F7"
C_GREEN = "#4caf50"
C_ORANGE = "#ff9800"
C_TEXT = "#24292f"
C_MUTED = "#6b7280"
C_BORDER = "#d1d5db"
C_BORDER_LT = "#e5e7eb"
C_BG_LIGHT = "#f3f4f6"
C_BG_LIGHTEST = "#f9fafb"
C_WHITE = "#ffffff"
C_PURPLE = "#9b59b6"
C_RED_SOFT = "#e74c3c"
C_ACCENT = "#d95f3b"
C_CONTEXT = "#f39c12"

FONT_MAIN = "DejaVu Sans"

# Training defaults
DEFAULT_PARAMS = {
    "hidden_dim": 128,
    "latent_dim": 64,
    "dropout": 0.15,
    "lr": 5e-4,
    "weight_decay": 1e-4,
}

"""Define the shared descriptor out-of-distribution clipping threshold.

`DESCRIPTOR_OOD_CLIP_SIGMA` specifies the maximum magnitude, in
`StandardScaler` output units, permitted for numeric descriptor features
before they are passed to the model. The same threshold is applied during
training and inference so that an out-of-distribution descriptor cannot
produce an unbounded network activation merely because its value falls well
outside the distribution represented by the training fold.

The value is centralized here because it is a behavioral calibration shared by
both the training and inference paths. This removes the need to maintain
independent copies that could silently diverge.

This constant is distinct from
`eval.logo_eval.FOLD_RANGE_N_SIGMA`. Although both use the same conventional
sigma threshold, `FOLD_RANGE_N_SIGMA` is a passive diagnostic used to report
descriptor-range conditions during evaluation, whereas this constant actively
clips features before model consumption.
"""

DESCRIPTOR_OOD_CLIP_SIGMA: float = 5.0

"""Define machine-resolved roots for VisQAI data and model storage.

These constants provide the storage locations consumed by :mod:`visqai.paths`;
path-resolution behavior remains in that module. The roots are resolved per
machine rather than embedding a user-specific absolute path, making the
configuration portable across developer workstations, deployment
environments, and CI systems.

Resolution follows this precedence order:

1. `VISQAI_DATA_ROOT` and `VISQAI_MODELS_ROOT` environment variables,
   when explicitly configured.
2. The Dropbox mount point discovered from Dropbox's `info.json`, preferring
   the configured business/team account over a personal account.
3. `Path.home() / "QATCH Dropbox"` as a per-user fallback for installations
   using Dropbox's conventional directory layout.

Import-time resolution is deliberately best-effort. Failure to discover a
Dropbox installation does not raise an exception, allowing unrelated
configuration such as feature or model constants to remain importable in
environments such as CI. If a resolved root is missing or unusable, the
corresponding data-access operation in :mod:`visqai.paths` is responsible for
raising the appropriate error when the path is actually needed.
"""


def _dropbox_root() -> Path | None:
    """Discover the configured Dropbox root directory on the current machine.

    Searches Dropbox's `info.json` in the standard Windows application-data
    locations and returns the first configured account root, preferring a
    business account over a personal account. Discovery is best-effort:
    missing environment variables, unreadable files, malformed JSON, or
    missing account entries are ignored and result in `None` if no usable
    root can be found.

    Returns:
        Path | None: The configured Dropbox root directory, or `None` when
        Dropbox configuration cannot be discovered.
    """
    for env_var in ("LOCALAPPDATA", "APPDATA"):
        base = os.environ.get(env_var)
        if not base:
            continue
        info_path = Path(base) / "Dropbox" / "info.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for account_type in ("business", "personal"):
            account = info.get(account_type)
            if not isinstance(account, dict):
                continue
            root = account.get("root_path") or account.get("path")
            if root:
                return Path(root)
    return None


def _formulations_ml_root() -> Path:
    """Resolve the root directory for the shared Formulations ML workspace.

    Uses the machine's discovered Dropbox root when available and falls back
    to the conventional per-user `~/QATCH Dropbox` location when Dropbox
    configuration cannot be discovered. The shared `QATCH Team Folder` and
    `Formulations ML` components are fixed workspace names; only the
    machine-specific Dropbox root is resolved dynamically.

    Returns:
        Path: The resolved `QATCH Team Folder/Formulations ML` directory.
    """
    root = _dropbox_root() or (Path.home() / "QATCH Dropbox")
    return root / "QATCH Team Folder" / "Formulations ML"


def _env_or(env_var: str, default: Path) -> Path:
    """Resolve a path from an environment variable with a fallback value.

    An explicitly configured environment variable takes precedence over the
    supplied default. Empty or unset environment variables are treated as
    absent.

    Args:
        env_var: Name of the environment variable to inspect.
        default: Path to use when the environment variable is unset or empty.

    Returns:
        Path: The environment-provided path when configured; otherwise,
        `default`.
    """
    value = os.environ.get(env_var)
    return Path(value) if value else default


DATA_ROOT = _env_or("VISQAI_DATA_ROOT", _formulations_ml_root() / "data")
MODELS_ROOT = _env_or("VISQAI_MODELS_ROOT", _formulations_ml_root() / "models")

DATA_LATEST_DIR = DATA_ROOT / "latest"
DATA_LEGACY_DIR = DATA_ROOT / "legacy"
CHECKPOINTS_DIR = MODELS_ROOT / "checkpoints"
PRODUCTION_DIR = MODELS_ROOT / "production"
