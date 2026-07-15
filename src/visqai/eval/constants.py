"""
constants.py
============
Shear-rate column/label constants, unified from the three independent
spellings previously scattered across ml/cnp_mk2/ibal_parity_test.py
(SHEAR_COLS/SHEAR_RATES/SHEAR_LABELS/SHEAR_COLORS) and
ml/cnp_mk2/learning_curve_ibal.py (SHAPE_VISC_COLS/SHAPE_SHEAR_RATES/
SHAPE_SHEAR_LABELS, plus VISC_COLS/PRED_COLS -- the narrower single-rate
subset used for convergence-replay scalar metrics). All five full-spectrum
constants covered the identical 5 shear rates; only the label FORMAT
differed (a dict of long labels for parity plots vs. a list of short labels
for convergence plots) so both are kept, not merged away.
"""

from __future__ import annotations

SHEAR_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]
SHEAR_RATES = [100.0, 1000.0, 10000.0, 100000.0, 15000000.0]
N_SHEARS = len(SHEAR_COLS)

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

# Single-rate subset used for convergence-replay scalar metrics (not a
# duplicate of SHEAR_COLS -- a deliberately narrower, different-purpose
# config: learning_curve_ibal.py tracks convergence at 1000 s⁻¹ only).
VISC_COLS = [
    "Viscosity_1000",
]
PRED_COLS = [f"Pred_{c}" for c in VISC_COLS]
