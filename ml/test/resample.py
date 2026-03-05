"""
resample.py
===========
Synthetic data augmentation for protein viscosity formulation data.

PASS 1 – Concentration-bin upsampling (original)
-------------------------------------------------
Upsamples each protein class to TARGET_SAMPLES_PER_TYPE by perturbing
existing profiles using Carreau-Yasuda parameter jitter or log-space noise.
Prioritises flat (plateau) profiles to avoid the model under-representing
high-shear Newtonian behaviour.

PASS 2 – Component response augmentation (new)
-----------------------------------------------
Generates synthetic samples that sweep excipient/stabilizer concentration
across the ranges that are poorly covered in the core data.  This is the
primary mechanism for giving the decoder training signal about *how* each
protein responds to different amounts of Sucrose, Trehalose, Arginine,
Lysine, and Proline.

PASS 3 – Protein concentration interpolation (new)
---------------------------------------------------
For each unique (protein_type × buffer_type × buffer_pH × buffer_conc) group
that has at least MIN_CONC_POINTS_PASS3 distinct protein concentration
measurements, fits a log-linear model

    log₁₀(η) ≈ slope · log₁₀(c_protein) + intercept

independently for each of the five shear-rate viscosity columns.  A small
number of synthetic rows are then interpolated within the observed
concentration range and added to the pool.  Monotonicity across shear rates
is enforced and small log-space noise is injected to prevent exact
duplicates.

Strategy
~~~~~~~~
For each (protein_type × component) pair where at least one
(no-component, with-component) matched pair exists:

1. Collect "anchor pairs" — rows of the same protein at similar protein
   concentration and pH where one has no component and one has a known
   component concentration.

2. For each anchor pair, interpolate / extrapolate the viscosity delta
   (Δ log₁₀ η) to the target concentration grid using a simple monotone
   model in log(c) space.  The log-linear model is deliberately simple
   to avoid over-fitting sparse response curves.

3. Propagate the predicted Δ log₁₀ η to the full 5-point shear profile
   by:
   a. Fitting a Carreau-Yasuda model to the base profile.
   b. Scaling η₀ (zero-shear viscosity) by the predicted factor while
      keeping the shape parameters (η_inf, K, a, n) fixed.
   c. Falling back to a uniform log-space shift if CY fitting fails.

4. Add calibrated noise: ±σ_noise * |Δ| on the predicted delta to reflect
   genuine batch-to-batch variability observed in the core data.

For proteins with NO component data at all (e.g. Bevacizumab excipients,
Nivolumab stabilisers), response parameters are transferred from the
same-class median with higher noise (±TRANSFER_NOISE_FRAC).

Output
~~~~~~
The augmented CSV is written to OUTPUT_FILE.  All synthetic rows have IDs
of the form:
  <original_id>_cr<component_abbrev>_c<conc>_d<draw>
to make them easily identifiable.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeWarning, curve_fit

warnings.simplefilter("ignore", OptimizeWarning)


# ============================================================
# Global Configuration
# ============================================================

INPUT_FILE = "data/raw/formulation_data_03042026.csv"
OUTPUT_FILE = "data/processed/augmented_formulation_data.csv"

# --- Pass 1 settings ---
TARGET_SAMPLES_PER_TYPE: int = 200
N_CONCENTRATION_BINS: int = 5

# --- Pass 2 settings ---

# Minimum number of anchor pairs required to attempt direct fitting.
# Proteins below this threshold receive transferred curves instead.
MIN_ANCHOR_PAIRS: int = 2

# Number of noisy draw variants per (anchor_pair × target_conc).
# Kept low (2) to avoid overwhelming real data with one anchor's signal.
N_NOISE_DRAWS: int = 2

# 1-sigma noise level on the predicted Δlog₁₀η, expressed as a fraction
# of the predicted delta magnitude. Reflects observed intra-protein variance.
NOISE_SIGMA_FRAC: float = 0.20

# Noise level for transferred (class-level) response curves — wider to
# reflect higher uncertainty when extrapolating to a new protein.
TRANSFER_NOISE_FRAC: float = 0.35

# Matching tolerances for identifying anchor pairs.
CONC_MATCH_TOL: float = 20.0  # mg/mL — protein concentration window
PH_MATCH_TOL: float = 0.5  # pH units

# Maximum samples to generate per (protein × component) pair in Pass 2.
# Prevents any one protein/component from dominating the synthetic pool.
MAX_PER_PROTEIN_COMPONENT: int = 120

# Target concentration grids for each component type.
# Units match the dataset: M for stabilisers, mM for excipients.
STABILIZER_TARGET_CONCS: list[float] = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]
EXCIPIENT_TARGET_CONCS: list[float] = [
    25.0,
    50.0,
    75.0,
    100.0,
    125.0,
    150.0,
    175.0,
    200.0,
    250.0,
    300.0,
]

# --- Pass 3 settings ---

# Minimum unique protein concentrations in a (protein × buffer) group before
# attempting an interpolation fit.
# [ENHANCEMENT-1] Reduced to 1 because the buffer-viscosity anchor (below)
# always provides a second point, so even a single measurement is sufficient
# to define an interpolation curve.  This unblocks belatacept, trastuzumab,
# and poly-hIgG groups that previously had only 1 baseline concentration.
MIN_CONC_POINTS_PASS3: int = 1

# Number of base interpolated protein-concentration samples per group.
# [ENHANCEMENT-4] Increased from 10; large gaps get additional points on top
# of this via the gap-targeted allocation below.
N_INTERP_SAMPLES_PASS3: int = 15

# 1-sigma log₁₀-viscosity noise applied to each interpolated point.
INTERP_NOISE_SIGMA_PASS3: float = 0.02

# Minimum fractional distance from an existing concentration before a
# candidate interpolation point is accepted (avoids near-duplicates).
INTERP_EXCLUSION_FRAC_PASS3: float = 0.05

# [ENHANCEMENT-2] Buffer-viscosity physical anchor.
# At zero protein concentration the solution viscosity approaches that of the
# pure buffer.  Adding this as a synthetic anchor point:
#   (a) allows the extended low-concentration range to be interpolated without
#       a log-linear extrapolation singularity at c→0, and
#   (b) constrains the PCHIP curve to a physically correct baseline, preventing
#       it from curving below buffer viscosity at low concentrations.
# Value: 15 mM Histidine, pH 6.0, 25 °C ≈ 0.91 cP (literature).
# Groups with a different buffer will use the same value as an approximation;
# the effect is small since we only use the anchor to guide the low-conc region.
BUFFER_VISCOSITY_CP: float = 0.91

# Protein concentration used to represent the "c → 0" anchor in log space.
# Must be positive (log cannot accept 0).  1 mg/mL is well below any observed
# protein concentration in the dataset and places the anchor firmly outside
# the training distribution so the model learns the dilute-solution limit.
BUFFER_ANCHOR_CONC_MG_ML: float = 1.0

# [ENHANCEMENT-3] Extended lower bound for interpolation / extrapolation.
# Pass 3 previously restricted candidate points to the interior of the
# observed concentration range [c_min, c_max].  The new lower bound is
# BUFFER_ANCHOR_CONC_MG_ML, giving the model training signal all the way
# from the dilute limit up to the highest observed baseline concentration.
INTERP_LOWER_BOUND_MG_ML: float = BUFFER_ANCHOR_CONC_MG_ML

# [ENHANCEMENT-5] Gap-targeted point placement.
# Concentration gaps wider than this threshold receive additional interpolated
# points proportional to their width, ensuring large under-sampled regions
# (e.g. Nivolumab 168→240 mg/mL, 72 mg/mL gap) are explicitly covered.
GAP_FILL_THRESHOLD_MG_ML: float = 30.0
# Extra points allocated per unit of gap above the threshold (per 30 mg/mL).
GAP_FILL_DENSITY: float = 2.0  # points per 30 mg/mL of gap

# [ENHANCEMENT-6] How far (mg/mL) to extrapolate beyond the highest measured
# baseline concentration.  PCHIP with extrapolate=True continues the curve
# using the slope of the final measured interval, which for mAbs in the
# 100–330 mg/mL range is a steeply rising exponential-like region — exactly
# the regime where high-concentration drug products operate.
# Set to 0 to disable extrapolation entirely.
EXTRAPOLATION_EXTENSION_MG_ML: float = 100.0

# Protein class membership used for cross-protein response curve transfer.
# Keys are Protein_type values (case-insensitive), values are class labels.
PROTEIN_CLASS_MAP: dict[str, str] = {
    "adalimumab": "mab_igg1",
    "bevacizumab": "mab_igg1",
    "trastuzumab": "mab_igg1",
    "pembrolizumab": "mab_igg4",
    "ibalizumab": "mab_igg4",
    "nivolumab": "mab_igg4",
    "belatacept": "fc_fusion",
    "etanercept": "fc_fusion",
    "vudalimab": "bispecific",
    "poly-higg": "polyclonal",
    "bgg": "polyclonal",
    "bsa": "other",
}

# Mapping from raw component name in the CSV to a short abbreviation used
# for synthetic sample IDs and internal bookkeeping.
COMPONENT_ABBREV: dict[str, str] = {
    "sucrose": "suc",
    "trehalose": "tre",
    "arginine": "arg",
    "lysine": "lys",
    "proline": "pro",
}


# ============================================================
# Dataclass for a fitted response curve
# ============================================================


@dataclass
class ComponentResponseCurve:
    """
    Encodes the relationship between component concentration and viscosity
    change for a specific (protein_type, component_name) pair.

    The model is log-linear in concentration:
        Δlog₁₀η(c) ≈ slope * log₁₀(c / c_ref)

    where Δlog₁₀η = log₁₀(η_base) − log₁₀(η_with_component)
    (positive = viscosity reduction, negative = viscosity increase).

    Attributes
    ----------
    protein_type : str
    component_name : str
    anchor_concs : list[float]
        Observed component concentrations (native units).
    anchor_deltas : list[float]
        Observed Δlog₁₀η values at each anchor concentration.
        One value per anchor pair (multiple observations at the same
        concentration are kept separately to capture variability).
    slope : float
        log-linear slope fitted to anchor data.  Used for interpolation
        and extrapolation.
    intercept : float
        Intercept of the log-linear fit at log₁₀(c) = 0.
    n_pairs : int
        Number of anchor pairs used.
    source : str
        "fitted" | "transferred_from_<class>"
    protein_concs_anchored : list[float]
        Protein concentrations at which anchors were observed.
        Used to scale the response to new protein concentrations.
    """

    protein_type: str
    component_name: str
    anchor_concs: list[float] = field(default_factory=list)
    anchor_deltas: list[float] = field(default_factory=list)
    slope: float = 0.0
    intercept: float = 0.0
    n_pairs: int = 0
    source: str = "fitted"
    protein_concs_anchored: list[float] = field(default_factory=list)

    def predict_delta(self, target_conc: float) -> float:
        """
        Predict Δlog₁₀η at a target component concentration.

        Returns 0.0 for zero concentration (no-component reference).
        Clamps the magnitude to ±1.5 log₁₀ units to avoid unphysical
        extrapolation.
        """
        if target_conc <= 0.0:
            return 0.0
        pred = self.slope * np.log10(target_conc) + self.intercept
        return float(np.clip(pred, -1.5, 1.5))


# ============================================================
# Pass 1 helpers – Viscosity profile classification
# ============================================================

VISCOSITY_MAGNITUDE_BINS = [
    ("very_low", 0.0, 3.0),
    ("low", 5.0, 15.0),
    ("medium", 15.0, 35.0),
    ("high", 35.0, 65.0),
    ("very_high", 65.0, np.inf),
]

SHEAR_THINNING_BINS = [
    ("newtonian", 0.0, 2.0),
    ("mild_thinning", 2.0, 10.0),
    ("strong_thinning", 10.0, np.inf),
]


def classify_viscosity_profile(y_linear: np.ndarray) -> str:
    eta_low = float(y_linear[0])
    eta_high = float(y_linear[-1])
    mag_label = "very_high"
    for label, lo, hi in VISCOSITY_MAGNITUDE_BINS:
        if lo <= eta_low < hi:
            mag_label = label
            break
    ratio = eta_low / max(eta_high, 1e-9)
    st_label = "strong_thinning"
    for label, lo, hi in SHEAR_THINNING_BINS:
        if lo <= ratio < hi:
            st_label = label
            break
    return f"{mag_label}__{st_label}"


def detect_flattening(y_linear: np.ndarray) -> tuple[bool, list[int]]:
    """
    Identifies flat regions in a viscosity profile.

    A consecutive pair (i, i+1) is flat when the viscosity drop is less
    than 2 % of the value at point i.
    """
    flat_idxs = [
        i
        for i in range(len(y_linear) - 1)
        if (y_linear[i] - y_linear[i + 1]) < 0.02 * y_linear[i]
    ]
    return len(flat_idxs) > 0, flat_idxs


def normalize_protein_type(raw: str) -> str:
    return str(raw).strip().lower()


# ============================================================
# Carreau-Yasuda model
# ============================================================


def carreau_yasuda_model(gamma, eta_0, eta_inf, K, a, n):
    gamma = np.maximum(gamma, 1e-6)
    core = 1.0 + np.power(K * gamma, a)
    exponent = (1.0 - n) / a
    return eta_inf + (eta_0 - eta_inf) / np.power(core, exponent)


def fit_carreau_yasuda(
    y_linear: np.ndarray,
    shear_rates: np.ndarray,
    maxfev: int = 10000,
) -> tuple[bool, Optional[np.ndarray]]:
    eta_0_guess = max(y_linear[0], y_linear[-1] + 0.1)
    p0 = [eta_0_guess, y_linear[-1], 0.001, 2.0, 0.5]
    bounds = ([0, 0, 0, 0.1, 0], [np.inf, np.inf, np.inf, 10.0, 1.0])
    try:
        popt, _ = curve_fit(
            carreau_yasuda_model,
            shear_rates,
            y_linear,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev,
        )
        return True, popt
    except RuntimeError:
        return False, None


# ============================================================
# Pass 1 – core synthetic sample generation
# ============================================================


def generate_synthetic_sample(
    row: pd.Series,
    shear_rates: np.ndarray,
    visc_cols: list[str],
    continuous_cols: list[str],
    syn_id: str,
    popt: Optional[np.ndarray],
    fit_success: bool,
) -> pd.Series:
    """
    Generates one synthetic sample from a core row.

    Method priority
    ---------------
    1. Carreau-Yasuda parameter perturbation (preserves profile shape).
    2. Log-space noise injection with monotonicity enforcement (fallback).

    A ±2 % jitter is applied to continuous formulation features.
    """
    y_linear = np.maximum(row[visc_cols].values.astype(float), 1e-6)
    new_row = row.copy()
    new_row["ID"] = f"{row['ID']}_{syn_id}"

    if fit_success and popt is not None:
        perturbed = popt * np.random.normal(1.0, 0.03, size=len(popt))
        perturbed = np.maximum(perturbed, 1e-6)
        perturbed[4] = min(perturbed[4], 1.0)  # clamp n ≤ 1
        new_row[visc_cols] = carreau_yasuda_model(shear_rates, *perturbed)
    else:
        y_log = np.log10(y_linear)
        synth_log = y_log + np.random.normal(0, 0.02, size=len(y_log))
        for j in range(1, len(synth_log)):
            if synth_log[j] > synth_log[j - 1]:
                synth_log[j] = synth_log[j - 1]
        interp = PchipInterpolator(np.log10(shear_rates), synth_log)
        new_row[visc_cols] = 10 ** interp(np.log10(shear_rates))

    for col in continuous_cols:
        val = new_row[col]
        if pd.notnull(val) and val > 0:
            new_row[col] = val * np.random.normal(1.0, 0.02)

    return new_row


def assign_concentration_bins(
    group: pd.DataFrame,
    conc_col: str = "Protein_conc",
    n_bins: int = N_CONCENTRATION_BINS,
) -> pd.Series:
    conc = group[conc_col]
    actual_bins = min(n_bins, conc.nunique())
    if actual_bins < 2:
        return pd.Series(0, index=group.index)
    try:
        return pd.qcut(conc, q=actual_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(0, index=group.index)


def _allocate_bin_quotas(
    bin_sizes: dict[int, int],
    needed: int,
    target_per_bin: float,
) -> dict[int, int]:
    bins = sorted(bin_sizes)
    deficits = {b: max(0.0, target_per_bin - bin_sizes[b]) for b in bins}
    total_deficit = sum(deficits.values())

    if total_deficit > 0:
        raw = {b: needed * deficits[b] / total_deficit for b in bins}
    else:
        raw = {b: needed / len(bins) for b in bins}

    allocs = {b: int(v) for b, v in raw.items()}
    remainder = needed - sum(allocs.values())
    by_frac = sorted(bins, key=lambda b: raw[b] - int(raw[b]), reverse=True)
    for i in range(remainder):
        allocs[by_frac[i % len(by_frac)]] += 1
    return allocs


def upsample_protein_type_to_target(
    group: pd.DataFrame,
    shear_rates: np.ndarray,
    visc_cols: list[str],
    continuous_cols: list[str],
    target: int = TARGET_SAMPLES_PER_TYPE,
    n_bins: int = N_CONCENTRATION_BINS,
) -> list[pd.Series]:
    core_count = len(group)
    needed = target - core_count
    if needed <= 0:
        return []

    group = group.copy()
    group["_conc_bin"] = assign_concentration_bins(group, n_bins=n_bins)
    actual_bins = sorted(group["_conc_bin"].dropna().unique())
    if len(actual_bins) == 0:
        group["_conc_bin"] = 0
        actual_bins = [0]

    n_actual = len(actual_bins)
    target_per_bin = target / n_actual
    bin_sizes = {b: int((group["_conc_bin"] == b).sum()) for b in actual_bins}
    allocs = _allocate_bin_quotas(bin_sizes, needed, target_per_bin)

    synthetic_rows: list[pd.Series] = []
    for bin_id in actual_bins:
        quota = allocs.get(int(bin_id), 0)
        if quota <= 0:
            continue
        bin_df = group[group["_conc_bin"] == bin_id]
        if len(bin_df) == 0:
            continue

        sample_meta = []
        for _, row in bin_df.iterrows():
            y = np.maximum(row[visc_cols].values.astype(float), 1e-6)
            has_flat, _ = detect_flattening(y)
            fit_ok, popt = fit_carreau_yasuda(y, shear_rates)
            sample_meta.append((row, has_flat, fit_ok, popt))

        flat_pool = [(r, fo, po) for r, hf, fo, po in sample_meta if hf]
        thin_pool = [(r, fo, po) for r, hf, fo, po in sample_meta if not hf]
        priority_queue = flat_pool + thin_pool
        if not priority_queue:
            continue

        syn_count = 0
        cycle_idx = 0
        while syn_count < quota:
            row, fit_ok, popt = priority_queue[cycle_idx % len(priority_queue)]
            syn_id = f"cb{int(bin_id)}_s{syn_count + 1}"
            new_row = generate_synthetic_sample(
                row,
                shear_rates,
                visc_cols,
                continuous_cols,
                syn_id,
                popt,
                fit_ok,
            )
            synthetic_rows.append(new_row)
            syn_count += 1
            cycle_idx += 1

    return synthetic_rows


# ============================================================
# Pass 2 – Component response augmentation
# ============================================================

# -----------
# Helpers
# -----------


def _norm_comp_name(raw: str) -> str:
    """Return a lowercase canonical component name."""
    s = str(raw).strip().lower()
    for key in COMPONENT_ABBREV:
        if key in s:
            return key
    return s


def _get_visc_array(row: pd.Series, visc_cols: list[str]) -> np.ndarray:
    v = row[visc_cols].values.astype(float)
    return np.maximum(v, 1e-6)


def _apply_delta_to_profile(
    base_visc: np.ndarray,
    delta_log10: float,
    shear_rates: np.ndarray,
    noise_sigma_frac: float = NOISE_SIGMA_FRAC,
) -> np.ndarray:
    """
    Applies a viscosity shift (Δlog₁₀η) to a full shear-rate profile.

    Method
    ------
    1. Fit a Carreau-Yasuda model to the base profile.
    2. Scale η₀ by the predicted factor (10^Δ).  Keep η_inf, K, a, n.
    3. Fall back to a uniform log-space shift if CY fit fails.

    Additive Gaussian noise on Δ is sampled once per call, so each draw
    produces a distinct noisy variant.
    """
    rng = np.random.default_rng()
    noise = rng.normal(0.0, noise_sigma_frac * abs(delta_log10))
    actual_delta = delta_log10 + noise

    # Try CY-based propagation (preserves shear-thinning character).
    fit_ok, popt = fit_carreau_yasuda(base_visc, shear_rates)
    if fit_ok and popt is not None:
        eta_0_new = popt[0] / (10**actual_delta)  # positive Δ → lower η
        eta_0_new = max(eta_0_new, 1e-3)
        new_popt = popt.copy()
        new_popt[0] = eta_0_new
        # η_inf must stay ≤ η_0.
        new_popt[1] = min(popt[1], eta_0_new * 0.95)
        result = carreau_yasuda_model(shear_rates, *new_popt)
    else:
        # Fallback: uniform log-space shift.
        log_base = np.log10(base_visc)
        log_new = log_base - actual_delta  # subtract → lower η for positive Δ
        # Preserve relative monotonicity.
        for j in range(1, len(log_new)):
            if log_new[j] > log_new[j - 1]:
                log_new[j] = log_new[j - 1]
        result = 10**log_new

    return np.maximum(result, 1e-6)


# -----------
# Step 1: Collect anchor pairs
# -----------


def _collect_anchor_pairs(
    df: pd.DataFrame,
    protein_type: str,
    type_col: str,
    conc_col: str,
    component_name: str,
    visc_cols: list[str],
) -> list[tuple[pd.Series, pd.Series]]:
    """
    Returns (base_row, comp_row) pairs for a specific protein/component.

    base_row  — no-component sample
    comp_row  — same protein at similar conc/pH with the component present

    The "no-component" condition is defined as both Stabilizer_type=='none'
    AND Excipient_type=='none', so we always use a clean baseline.
    """
    sub = df[df["Protein_type"].str.lower() == protein_type.lower()]
    no_comp = sub[
        (sub["Stabilizer_type"].str.lower() == "none")
        & (sub["Excipient_type"].str.lower() == "none")
    ]
    with_comp = sub[
        sub[type_col].str.lower().str.contains(component_name, na=False)
        & (sub[conc_col] > 0)
    ]

    pairs: list[tuple[pd.Series, pd.Series]] = []
    for _, rc in with_comp.iterrows():
        # Match base rows within the protein-conc and pH window.
        candidates = no_comp[
            (np.abs(no_comp["Protein_conc"] - rc["Protein_conc"]) < CONC_MATCH_TOL)
            & (np.abs(no_comp["Buffer_pH"] - rc["Buffer_pH"]) < PH_MATCH_TOL)
        ]
        if candidates.empty:
            continue
        # Choose the single closest candidate by protein concentration.
        best_idx = (candidates["Protein_conc"] - rc["Protein_conc"]).abs().idxmin()
        best_base = candidates.loc[best_idx]

        # Both rows must have valid viscosity data.
        base_visc = _get_visc_array(best_base, visc_cols)
        comp_visc = _get_visc_array(rc, visc_cols)
        if np.any(np.isnan(base_visc)) or np.any(np.isnan(comp_visc)):
            continue
        if base_visc[0] <= 0 or comp_visc[0] <= 0:
            continue

        pairs.append((best_base, rc))

    return pairs


# -----------
# Step 2: Fit response curves
# -----------


def _fit_response_curve(
    protein_type: str,
    component_name: str,
    pairs: list[tuple[pd.Series, pd.Series]],
    type_col: str,
    conc_col: str,
    visc_cols: list[str],
) -> ComponentResponseCurve:
    """
    Fits a log-linear response curve from anchor pairs.

    Δlog₁₀η = slope * log₁₀(c) + intercept

    The fit is performed with least-squares over all available pairs.
    If only one concentration point exists, the slope is estimated from
    the single point and the physics prior that the effect is zero at
    infinitesimally small concentrations (i.e. the line passes through
    (log₁₀(c_min / 100), 0)).
    """
    curve = ComponentResponseCurve(
        protein_type=protein_type,
        component_name=component_name,
        source="fitted",
    )

    xs: list[float] = []  # log₁₀(component_conc)
    ys: list[float] = []  # Δlog₁₀η
    pconcs: list[float] = []

    for base_row, comp_row in pairs:
        c = float(comp_row[conc_col])
        if c <= 0:
            continue
        eta_base = comp_row[visc_cols[0]]  # 100 s⁻¹ viscosity as signal
        eta_ref = base_row[visc_cols[0]]
        if eta_base <= 0 or eta_ref <= 0:
            continue
        delta = np.log10(eta_ref) - np.log10(eta_base)
        xs.append(np.log10(c))
        ys.append(delta)
        pconcs.append(float(comp_row["Protein_conc"]))
        curve.anchor_concs.append(c)
        curve.anchor_deltas.append(delta)
        curve.protein_concs_anchored.append(float(comp_row["Protein_conc"]))

    curve.n_pairs = len(xs)
    if not xs:
        return curve

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)

    if len(xs) == 1:
        # Single point: slope estimated assuming the effect is zero at
        # a concentration 100× smaller than the observed point.
        slope = ys_arr[0] / (xs_arr[0] - (xs_arr[0] - 2.0))
        intercept = ys_arr[0] - slope * xs_arr[0]
    else:
        # Least-squares line through all points.
        A = np.column_stack([xs_arr, np.ones_like(xs_arr)])
        result = np.linalg.lstsq(A, ys_arr, rcond=None)
        slope, intercept = result[0]

    curve.slope = float(slope)
    curve.intercept = float(intercept)
    return curve


# -----------
# Step 3: Build all curves (direct + transfer)
# -----------

# Component definitions: (column_for_type, column_for_conc, target_concs)
COMPONENT_DEFS: list[tuple[str, str, list[float]]] = [
    ("Stabilizer_type", "Stabilizer_conc", STABILIZER_TARGET_CONCS),
    ("Excipient_type", "Excipient_conc", EXCIPIENT_TARGET_CONCS),
]

COMPONENTS_PER_COL: dict[str, list[str]] = {
    "Stabilizer_type": ["sucrose", "trehalose"],
    "Excipient_type": ["arginine", "lysine", "proline"],
}


def build_response_curves(
    df: pd.DataFrame,
    visc_cols: list[str],
) -> dict[tuple[str, str], ComponentResponseCurve]:
    """
    Builds response curves for every (protein_type, component) combination.

    For proteins with fewer than MIN_ANCHOR_PAIRS direct observations, the
    curve is transferred from the class-level median.

    Returns
    -------
    dict mapping (protein_type_lower, component_name) → ComponentResponseCurve
    """
    all_proteins = [
        pt.lower()
        for pt in df["Protein_type"].unique()
        if str(pt).lower() not in ("none", "unknown", "nan")
    ]

    curves: dict[tuple[str, str], ComponentResponseCurve] = {}

    for type_col, conc_col, _ in COMPONENT_DEFS:
        for comp_name in COMPONENTS_PER_COL[type_col]:
            # --- Direct fitting pass ---
            for ptype in all_proteins:
                pairs = _collect_anchor_pairs(
                    df, ptype, type_col, conc_col, comp_name, visc_cols
                )
                if len(pairs) >= MIN_ANCHOR_PAIRS:
                    curve = _fit_response_curve(
                        ptype, comp_name, pairs, type_col, conc_col, visc_cols
                    )
                else:
                    # Placeholder — will be filled by transfer pass below.
                    curve = ComponentResponseCurve(
                        protein_type=ptype,
                        component_name=comp_name,
                        n_pairs=len(pairs),
                        source="pending_transfer",
                    )
                    # Still store any single-point data if it exists.
                    if len(pairs) == 1:
                        curve = _fit_response_curve(
                            ptype, comp_name, pairs, type_col, conc_col, visc_cols
                        )
                        curve.source = "single_point"
                curves[(ptype, comp_name)] = curve

            # --- Transfer pass: fill proteins with no / sparse data ---
            # Group fitted curves by protein class.
            class_curves: dict[str, list[ComponentResponseCurve]] = defaultdict(list)
            for ptype in all_proteins:
                c = curves.get((ptype, comp_name))
                if c is not None and c.n_pairs >= MIN_ANCHOR_PAIRS:
                    pclass = PROTEIN_CLASS_MAP.get(ptype, "other")
                    class_curves[pclass].append(c)

            for ptype in all_proteins:
                c = curves.get((ptype, comp_name))
                if c is not None and c.source not in ("fitted", "single_point"):
                    # Need transfer.
                    pclass = PROTEIN_CLASS_MAP.get(ptype, "other")
                    donors = class_curves.get(pclass, [])

                    # Widen the donor pool if no same-class data available.
                    if not donors:
                        donors = [
                            cv
                            for cv in curves.values()
                            if cv.component_name == comp_name
                            and cv.n_pairs >= MIN_ANCHOR_PAIRS
                        ]

                    if not donors:
                        # No transfer data — leave curve as empty.
                        continue

                    # Transfer: use median slope/intercept from donors.
                    med_slope = float(np.median([d.slope for d in donors]))
                    med_intercept = float(np.median([d.intercept for d in donors]))

                    transferred = ComponentResponseCurve(
                        protein_type=ptype,
                        component_name=comp_name,
                        slope=med_slope,
                        intercept=med_intercept,
                        n_pairs=0,
                        source=f"transferred_from_{pclass}",
                    )
                    curves[(ptype, comp_name)] = transferred

    return curves


# -----------
# Step 4: Generate synthetic samples from curves
# -----------


def _build_component_row(
    base_row: pd.Series,
    new_visc: np.ndarray,
    visc_cols: list[str],
    type_col: str,
    conc_col: str,
    component_display_name: str,
    target_conc: float,
    syn_id: str,
) -> pd.Series:
    """
    Clones base_row, replaces viscosity values and component metadata.
    """
    new_row = base_row.copy()
    new_row["ID"] = syn_id

    for i, col in enumerate(visc_cols):
        new_row[col] = new_visc[i]

    # Set the component type/conc column, and clear the other component
    # columns so the formulation is unambiguous.
    if type_col == "Stabilizer_type":
        new_row["Stabilizer_type"] = component_display_name
        new_row["Stabilizer_conc"] = target_conc
        # Leave Excipient_type/conc unchanged (may co-exist).
    else:
        new_row["Excipient_type"] = component_display_name
        new_row["Excipient_conc"] = target_conc
        # Leave Stabilizer_type/conc unchanged.

    return new_row


def _get_display_name(comp_name: str) -> str:
    """Returns the capitalised display name matching CSV conventions."""
    return comp_name.capitalize()


def generate_component_response_samples(
    df: pd.DataFrame,
    curves: dict[tuple[str, str], ComponentResponseCurve],
    shear_rates: np.ndarray,
    visc_cols: list[str],
) -> list[pd.Series]:
    """
    Main Pass-2 generator.

    For each (protein_type, component) pair with a fitted/transferred curve:

    1. Enumerate target concentration points not already well-covered in the
       core data (within ±12 % of the target concentration).

    2. For each uncovered target concentration, select base rows — no-component
       samples of the same protein — spread across the observed protein
       concentration range.  Prefer high-concentration samples (where the
       component effect is largest and most important to learn).

    3. Predict Δlog₁₀η from the curve, apply it to the base profile, and
       produce N_NOISE_DRAWS noisy variants.

    CY fits are pre-computed once per base row and reused across all
    (component, target_conc, draw) combinations for that row — this avoids
    the dominant O(n_concs × n_draws) fitting cost.

    Returns a flat list of synthetic rows.
    """
    synthetic_rows: list[pd.Series] = []

    all_proteins = [
        pt
        for pt in df["Protein_type"].unique()
        if str(pt).lower() not in ("none", "unknown", "nan")
    ]

    for ptype in all_proteins:
        sub = df[df["Protein_type"] == ptype]

        # No-component base rows for this protein.
        base_rows = sub[
            (sub["Stabilizer_type"].str.lower() == "none")
            & (sub["Excipient_type"].str.lower() == "none")
        ]
        if base_rows.empty:
            continue

        # Pre-compute CY fits once for all base rows of this protein.
        # maxfev=800 is a fast-fail limit: flat-start profiles that make
        # the optimizer loop for seconds just fall back to log-space shift.
        cy_cache: dict[int, tuple[bool, Optional[np.ndarray]]] = {}
        for idx, row in base_rows.iterrows():
            y = _get_visc_array(row, visc_cols)
            cy_cache[idx] = fit_carreau_yasuda(y, shear_rates, maxfev=800)

        protein_concs_available = sorted(base_rows["Protein_conc"].unique())
        # Emphasis on higher-concentration rows (bigger effect, more useful).
        high_conc_threshold = np.percentile(protein_concs_available, 60)

        high_conc_bases = base_rows[base_rows["Protein_conc"] >= high_conc_threshold]
        low_conc_bases = base_rows[base_rows["Protein_conc"] < high_conc_threshold]

        for type_col, conc_col, target_concs in COMPONENT_DEFS:
            for comp_name in COMPONENTS_PER_COL[type_col]:
                disp_name = _get_display_name(comp_name)

                curve = curves.get((ptype.lower(), comp_name))
                if curve is None or (curve.slope == 0.0 and curve.intercept == 0.0):
                    continue

                # Observed concentrations already in the core data.
                existing = sub[
                    sub[type_col].str.lower().str.contains(comp_name, na=False)
                    & (sub[conc_col] > 0)
                ][conc_col].values

                is_transferred = "transferred" in curve.source
                noise_frac = TRANSFER_NOISE_FRAC if is_transferred else NOISE_SIGMA_FRAC
                protein_gen_count = 0

                for t_conc in target_concs:
                    # Skip if this concentration is already well-represented.
                    if len(existing) > 0 and np.any(
                        np.abs(existing - t_conc) / (t_conc + 1e-9) < 0.12
                    ):
                        continue

                    predicted_delta = curve.predict_delta(t_conc)
                    # Skip near-zero predicted effects — they add noise without signal.
                    if abs(predicted_delta) < 0.03:
                        continue

                    # Sample: up to 2 high-conc + 1 low-conc base rows.
                    selected_bases: list[tuple[int, pd.Series]] = []
                    if not high_conc_bases.empty:
                        n_high = min(2, len(high_conc_bases))
                        chosen = np.random.choice(
                            len(high_conc_bases), n_high, replace=False
                        )
                        for ci in chosen:
                            r = high_conc_bases.iloc[ci]
                            selected_bases.append((r.name, r))
                    if not low_conc_bases.empty:
                        ci = np.random.randint(len(low_conc_bases))
                        r = low_conc_bases.iloc[ci]
                        selected_bases.append((r.name, r))

                    for row_idx, base_row in selected_bases:
                        base_visc = _get_visc_array(base_row, visc_cols)
                        fit_ok, cy_popt = cy_cache.get(row_idx, (False, None))

                        for draw in range(N_NOISE_DRAWS):
                            # Apply noise to Δ once per draw.
                            rng = np.random.default_rng()
                            noise = rng.normal(0.0, noise_frac * abs(predicted_delta))
                            actual_delta = float(
                                np.clip(predicted_delta + noise, -1.5, 1.5)
                            )

                            # Apply the delta using the pre-computed CY fit.
                            # Sign convention: positive actual_delta means viscosity
                            # REDUCTION (η_new < η_base), so we divide by 10^delta.
                            if fit_ok and cy_popt is not None:
                                eta_0_new = cy_popt[0] / (10**actual_delta)
                                eta_0_new = max(eta_0_new, 1e-3)
                                new_popt = cy_popt.copy()
                                new_popt[0] = eta_0_new
                                new_popt[1] = min(cy_popt[1], eta_0_new * 0.95)
                                new_visc = np.maximum(
                                    carreau_yasuda_model(shear_rates, *new_popt), 1e-6
                                )
                            else:
                                log_base = np.log10(base_visc)
                                log_new = log_base - actual_delta  # subtract delta
                                for j in range(1, len(log_new)):
                                    if log_new[j] > log_new[j - 1]:
                                        log_new[j] = log_new[j - 1]
                                new_visc = np.maximum(10**log_new, 1e-6)

                            abbrev = COMPONENT_ABBREV.get(comp_name, comp_name[:3])
                            conc_str = f"{t_conc:.3f}".replace(".", "p")
                            syn_id = f"{base_row['ID']}_cr{abbrev}_c{conc_str}_d{draw}"

                            new_row = _build_component_row(
                                base_row,
                                new_visc,
                                visc_cols,
                                type_col,
                                conc_col,
                                disp_name,
                                t_conc,
                                syn_id,
                            )
                            synthetic_rows.append(new_row)
                            protein_gen_count += 1

                        if protein_gen_count >= MAX_PER_PROTEIN_COMPONENT:
                            break

                    if protein_gen_count >= MAX_PER_PROTEIN_COMPONENT:
                        break

    return synthetic_rows


# ============================================================
# Pass 3 – Protein concentration interpolation (log-space)
# ============================================================


def generate_protein_conc_interpolation_samples(
    df: pd.DataFrame,
    visc_cols: list[str],
    min_conc_points: int = MIN_CONC_POINTS_PASS3,
    n_interp: int = N_INTERP_SAMPLES_PASS3,
    noise_sigma: float = INTERP_NOISE_SIGMA_PASS3,
    exclusion_frac: float = INTERP_EXCLUSION_FRAC_PASS3,
) -> list[pd.Series]:
    """
    Pass 3: Protein concentration interpolation / extrapolation.

    For each unique (Protein_type, Buffer_type, Buffer_pH, Buffer_conc) group
    restricted to baseline rows (no stabiliser / excipient / surfactant / salt):

    [ENHANCEMENT-1] Min requirement reduced to 1 measured concentration.
        A physical buffer-viscosity anchor at ~1 mg/mL is always injected,
        providing a second point so that a curve can be fitted even when only
        one protein concentration has been measured.

    [ENHANCEMENT-2] Buffer-viscosity anchor at c = BUFFER_ANCHOR_CONC_MG_ML.
        All five shear-rate viscosity columns are anchored to BUFFER_VISCOSITY_CP
        (≈0.91 cP for 15 mM His pH 6 at 25 °C), reflecting the physical
        boundary condition that η → η_buffer as c_protein → 0.  This prevents
        the interpolator from producing sub-buffer viscosities and gives the
        model accurate training signal in the dilute-solution regime.

    [ENHANCEMENT-3] Extended range: from INTERP_LOWER_BOUND_MG_ML to the
        highest observed baseline concentration (requested by user).
        Previously the range was clamped to [min_observed, max_observed].

    [ENHANCEMENT-4] PCHIP interpolation when ≥ 3 anchor points are available
        (buffer anchor + ≥ 2 measured concentrations).
        The previous log-linear fit assumes a straight line in log-log space,
        which badly misrepresents the S-shaped viscosity–concentration curve
        of mAbs (e.g. predicts 11 cP vs actual 4.5 cP at 160 mg/mL for
        Nivolumab).  PchipInterpolator follows the measured curvature exactly
        between anchor points and extrapolates monotonically beyond them.
        Log-linear is retained as the fallback when only 2 anchor points exist.

    [ENHANCEMENT-5] Gap-targeted point placement.
        Concentration gaps wider than GAP_FILL_THRESHOLD_MG_ML are allocated
        additional interpolation points proportional to their width.  This
        ensures that large under-sampled regions (e.g. Nivolumab 168→240 mg/mL)
        receive explicit synthetic training data rather than relying on the
        sliding-window context to bridge them.
    """
    from scipy.interpolate import PchipInterpolator  # local import for clarity

    synthetic_rows: list[pd.Series] = []

    group_cols = ["Protein_type", "Buffer_type", "Buffer_pH", "Buffer_conc"]
    for col in group_cols:
        if col not in df.columns:
            return synthetic_rows

    working = df.copy()
    working["Protein_conc"] = pd.to_numeric(working["Protein_conc"], errors="coerce")
    for col in ["Stabilizer_type", "Excipient_type", "Surfactant_type", "Salt_type"]:
        if col in working.columns:
            working[col] = working[col].fillna("none").astype(str)
    for col in ["Stabilizer_conc", "Excipient_conc", "Surfactant_conc", "Salt_conc"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)

    for keys, group in working.groupby(group_cols, dropna=False):
        ptype = keys[0]
        if str(ptype).lower() in ("none", "unknown", "nan"):
            continue

        # Restrict to baseline rows only.
        baseline_mask = (
            group["Stabilizer_type"].str.lower().eq("none")
            & group["Excipient_type"].str.lower().eq("none")
            & group["Surfactant_type"].str.lower().eq("none")
            & group["Salt_type"].str.lower().eq("none")
            & (group.get("Stabilizer_conc", 0) == 0)
            & (group.get("Excipient_conc", 0) == 0)
            & (group.get("Surfactant_conc", 0) == 0)
            & (group.get("Salt_conc", 0) == 0)
        )
        group = group[baseline_mask]
        group = group[group["Protein_conc"] > 0].dropna(subset=["Protein_conc"])

        unique_concs = sorted(group["Protein_conc"].unique())
        # [ENHANCEMENT-1] Lowered minimum to 1 (buffer anchor provides the 2nd point).
        if len(unique_concs) < min_conc_points:
            continue

        # ── [ENHANCEMENT-2] Inject buffer-viscosity anchor ──────────────
        # Represent c → 0 as BUFFER_ANCHOR_CONC_MG_ML (in log space).
        # All shear-rate columns get the same buffer viscosity because at
        # this concentration the protein has negligible effect on flow.
        anchor_conc = BUFFER_ANCHOR_CONC_MG_ML
        anchor_log_c = np.log10(anchor_conc)
        anchor_log_v = np.log10(BUFFER_VISCOSITY_CP)

        # Measured concentrations in log space (ascending).
        log_concs_meas = np.log10(np.array(unique_concs, dtype=float))

        # Full anchor set: buffer point + measured points.
        log_concs_all = np.concatenate([[anchor_log_c], log_concs_meas])
        # Guard: ensure strictly increasing (drop buffer anchor if it
        # overlaps the first measured concentration).
        if log_concs_all[1] <= log_concs_all[0]:
            log_concs_all = log_concs_meas
            has_buffer_anchor = False
        else:
            has_buffer_anchor = True

        # [ENHANCEMENT-3 / ENHANCEMENT-6]
        # Lower bound: INTERP_LOWER_BOUND_MG_ML (dilute limit, anchored to buffer visc).
        # Upper bound: highest measured concentration + EXTRAPOLATION_EXTENSION_MG_ML.
        # PCHIP with extrapolate=True continues the slope of the final measured
        # interval -- physically appropriate for the steep high-conc regime.
        log_c_low = np.log10(INTERP_LOWER_BOUND_MG_ML)
        c_max_obs = float(unique_concs[-1])
        c_max_ext = c_max_obs + EXTRAPOLATION_EXTENSION_MG_ML
        log_c_high = np.log10(c_max_ext)
        if log_c_high <= log_c_low:
            continue

        # ── [ENHANCEMENT-4] Fit per viscosity column ─────────────────────
        # Use PCHIP when ≥ 3 anchor points; log-linear otherwise.
        n_anchors = len(log_concs_all)
        use_pchip = n_anchors >= 3

        # For log-linear fallback (2-anchor case).
        ll_fits: dict[str, tuple[float, float]] = {}

        # For PCHIP we build one interpolator per viscosity column.
        pchip_interps: dict[str, PchipInterpolator] = {}

        for vcol in visc_cols:
            # Measured mean log-viscosity at each unique concentration.
            log_v_meas = []
            for c in unique_concs:
                vals = group.loc[group["Protein_conc"] == c, vcol].values.astype(float)
                vals = vals[vals > 0]
                log_v_meas.append(
                    float(np.nanmean(np.log10(vals))) if len(vals) > 0 else np.nan
                )
            log_v_meas = np.array(log_v_meas)

            # Drop NaN measured points.
            valid = ~np.isnan(log_v_meas)
            if valid.sum() == 0:
                continue

            # Build full anchor arrays for this column.
            if has_buffer_anchor:
                lc_fit = np.concatenate([[anchor_log_c], log_concs_meas[valid]])
                lv_fit = np.concatenate([[anchor_log_v], log_v_meas[valid]])
            else:
                lc_fit = log_concs_meas[valid]
                lv_fit = log_v_meas[valid]

            # Ensure strictly increasing x (required by PCHIP).
            order = np.argsort(lc_fit)
            lc_fit = lc_fit[order]
            lv_fit = lv_fit[order]
            # Remove any duplicates.
            _, uniq_idx = np.unique(lc_fit, return_index=True)
            lc_fit = lc_fit[uniq_idx]
            lv_fit = lv_fit[uniq_idx]

            if use_pchip and len(lc_fit) >= 3:
                # [ENHANCEMENT-6 FIX] Disable PCHIP's built-in extrapolation.
                # PCHIP extrapolates using the slope at the endpoint, which can
                # be wildly negative when the last measured interval has a dip
                # (e.g. Adalimumab: 100→10.3→6.1 cP, slope=-9 in log-log).
                # Instead we use PCHIP only within [anchor_low, c_max_obs] and
                # switch to a stable linear continuation beyond c_max_obs.
                # The linear slope is the log-log slope of the *measured* data
                # (buffer anchor excluded from slope estimation, as it can pull
                # the slope down for sparse groups), clamped to be non-negative
                # (viscosity cannot decrease with concentration).
                pchip_interps[vcol] = PchipInterpolator(
                    lc_fit, lv_fit, extrapolate=False
                )
                # Slope for extrapolation: least-squares over measured points only.
                lc_meas_valid = log_concs_meas[~np.isnan(log_v_meas)]
                lv_meas_valid = log_v_meas[~np.isnan(log_v_meas)]
                if len(lc_meas_valid) >= 2:
                    A_sl = np.column_stack([lc_meas_valid, np.ones_like(lc_meas_valid)])
                    ext_slope = float(
                        np.linalg.lstsq(A_sl, lv_meas_valid, rcond=None)[0][0]
                    )
                    ext_slope = max(ext_slope, 0.0)  # clamp: visc cannot fall with conc
                else:
                    ext_slope = 0.0
                # Anchor the linear extension to the last measured log-visc value.
                lc_ext_anchor = log_concs_meas[-1]
                lv_ext_anchor = float(
                    np.nanmean(
                        np.log10(
                            np.maximum(
                                group.loc[
                                    group["Protein_conc"] == unique_concs[-1], vcol
                                ].values.astype(float),
                                1e-6,
                            )
                        )
                    )
                )
                pchip_interps[vcol + "__ext_slope"] = ext_slope
                pchip_interps[vcol + "__ext_anchor_lc"] = lc_ext_anchor
                pchip_interps[vcol + "__ext_anchor_lv"] = lv_ext_anchor
            else:
                # Log-linear fallback (also used if PCHIP arrays ended up with < 3 unique pts).
                if len(lc_fit) >= 2:
                    A = np.column_stack([lc_fit, np.ones_like(lc_fit)])
                    sl, ic = np.linalg.lstsq(A, lv_fit, rcond=None)[0]
                else:
                    sl, ic = 0.0, float(lv_fit[0]) if len(lv_fit) else 0.0
                ll_fits[vcol] = (float(sl), float(ic))

        if not pchip_interps and not ll_fits:
            continue

        # ── [ENHANCEMENT-5] Gap-targeted candidate placement ─────────────
        # Start with a uniform grid across the full range.
        base_candidates = list(np.linspace(log_c_low, log_c_high, n_interp + 2)[1:-1])

        # Add extra points into large gaps between consecutive measured anchors.
        all_anchors_sorted = sorted(unique_concs)
        # Also consider the lower bound as a virtual anchor boundary.
        # Include the extrapolation ceiling as a boundary so the extension
        # zone gets gap-targeted points just like large measured gaps.
        boundaries = [INTERP_LOWER_BOUND_MG_ML] + all_anchors_sorted + [c_max_ext]
        for gi in range(len(boundaries) - 1):
            gap_lo, gap_hi = boundaries[gi], boundaries[gi + 1]
            gap_size = gap_hi - gap_lo
            if gap_size > GAP_FILL_THRESHOLD_MG_ML:
                n_extra = max(
                    1,
                    int(round(gap_size / GAP_FILL_THRESHOLD_MG_ML * GAP_FILL_DENSITY)),
                )
                extra_lcs = np.linspace(
                    np.log10(max(gap_lo, INTERP_LOWER_BOUND_MG_ML)),
                    np.log10(gap_hi),
                    n_extra + 2,
                )[1:-1]
                base_candidates.extend(extra_lcs.tolist())

        # Deduplicate and sort candidate log-concentrations.
        base_candidates = sorted(set(round(lc, 6) for lc in base_candidates))

        # Reject candidates too close to any existing measured concentration.
        new_log_concs: list[float] = [
            lc
            for lc in base_candidates
            if all(
                abs(10.0**lc - ec) / (ec + 1e-9) > exclusion_frac for ec in unique_concs
            )
        ]

        # Fallback: if every candidate was rejected, accept the midpoint.
        if not new_log_concs:
            mid_lc = float((log_c_low + log_c_high) / 2.0)
            if all(abs(10.0**mid_lc - ec) / (ec + 1e-9) > 1e-3 for ec in unique_concs):
                new_log_concs = [mid_lc]
            else:
                continue

        # ── Generate one synthetic row per accepted candidate ─────────────
        for i, log_c in enumerate(new_log_concs):
            new_c = round(float(10.0**log_c), 4)

            # Template: nearest existing row.
            idx_nearest = (group["Protein_conc"] - new_c).abs().idxmin()
            template_row = group.loc[idx_nearest]

            new_row = template_row.copy()
            new_row["ID"] = f"{template_row['ID']}_p3conc_{new_c:.1f}_i{i}"
            new_row["Protein_conc"] = new_c

            # Predict log₁₀(viscosity) per column, add noise, back-convert.
            visc_pred = []
            for vcol in visc_cols:
                if vcol in pchip_interps:
                    # Within the measured range: use PCHIP.
                    # Beyond c_max_obs: linear continuation with a clamped slope
                    # to prevent PCHIP endpoint divergence.
                    lc_ext_anchor = pchip_interps[vcol + "__ext_anchor_lc"]
                    lv_ext_anchor = pchip_interps[vcol + "__ext_anchor_lv"]
                    ext_slope = pchip_interps[vcol + "__ext_slope"]
                    if log_c <= lc_ext_anchor:
                        log_v_pred = float(pchip_interps[vcol](log_c))
                    else:
                        # Linear extrapolation anchored to last measured point.
                        log_v_pred = lv_ext_anchor + ext_slope * (log_c - lc_ext_anchor)
                elif vcol in ll_fits:
                    sl, ic = ll_fits[vcol]
                    log_v_pred = sl * log_c + ic
                else:
                    # Column has no fit; use buffer viscosity as floor.
                    log_v_pred = anchor_log_v

                log_v_pred += np.random.normal(0.0, noise_sigma)
                # Clamp to physical lower bound (buffer viscosity).
                log_v_pred = max(log_v_pred, anchor_log_v)
                visc_pred.append(float(10.0**log_v_pred))

            # Enforce non-increasing with shear rate (monotone shear thinning).
            for j in range(1, len(visc_pred)):
                if visc_pred[j] > visc_pred[j - 1]:
                    visc_pred[j] = visc_pred[j - 1]

            for vcol, val in zip(visc_cols, visc_pred, strict=True):
                new_row[vcol] = max(val, 1e-6)

            synthetic_rows.append(new_row)

    return synthetic_rows


# ============================================================
# Reporting helpers
# ============================================================


def _print_curve_summary(
    curves: dict[tuple[str, str], ComponentResponseCurve],
    all_proteins: list[str],
) -> None:
    print("\n  Response curve summary (Pass 2):")
    print(f"  {'Protein':<20} {'Component':<12} {'Pairs':>5}  {'Slope':>7}  {'Source'}")
    print("  " + "-" * 72)
    for comp_name in ["sucrose", "trehalose", "arginine", "lysine", "proline"]:
        for ptype in sorted(all_proteins):
            key = (ptype.lower(), comp_name)
            curve = curves.get(key)
            if curve is None:
                continue
            print(
                f"  {ptype:<20} {comp_name:<12} {curve.n_pairs:>5}  "
                f"{curve.slope:>+7.3f}  {curve.source}"
            )


# ============================================================
# Main
# ============================================================


def main() -> None:
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("File not found — check INPUT_FILE path.")
        return

    # Fill NaN component columns so string comparisons are safe.
    for col in ["Stabilizer_type", "Surfactant_type", "Excipient_type", "Salt_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("none").astype(str)

    shear_rates = np.array([100, 1000, 10000, 100000, 15_000_000], dtype=float)
    visc_cols = [
        "Viscosity_100",
        "Viscosity_1000",
        "Viscosity_10000",
        "Viscosity_100000",
        "Viscosity_15000000",
    ]
    continuous_cols = [
        "Protein_conc",
        "Temperature",
        "Buffer_conc",
        "Salt_conc",
        "Stabilizer_conc",
        "Surfactant_conc",
        "Excipient_conc",
    ]
    protein_type_col = "Protein_class_type"

    np.random.seed(42)

    # ------------------------------------------------------------------
    # Pre-augmentation summary
    # ------------------------------------------------------------------
    print("\n  Core sample counts by protein class type:")
    type_core_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        p = normalize_protein_type(row.get(protein_type_col, "unknown"))
        type_core_counts[p] = type_core_counts.get(p, 0) + 1
    for p in sorted(type_core_counts):
        n = type_core_counts[p]
        status = (
            f"→ needs {TARGET_SAMPLES_PER_TYPE - n} synthetic"
            if n < TARGET_SAMPLES_PER_TYPE
            else "→ already at/above target"
        )
        print(f"    {p:<30s}  core={n:4d}  {status}")

    # ------------------------------------------------------------------
    # Pass 1 – concentration-bin upsampling
    # ------------------------------------------------------------------
    print(
        f"\n[Pass 1] Upsampling each protein class to {TARGET_SAMPLES_PER_TYPE} samples "
        f"({N_CONCENTRATION_BINS} concentration bins, flattening-first)..."
    )
    pass1_synthetic: list[pd.Series] = []
    for p_type_raw in df[protein_type_col].unique():
        group = df[df[protein_type_col] == p_type_raw]
        synthetics = upsample_protein_type_to_target(
            group,
            shear_rates,
            visc_cols,
            continuous_cols,
            target=TARGET_SAMPLES_PER_TYPE,
            n_bins=N_CONCENTRATION_BINS,
        )
        pass1_synthetic.extend(synthetics)

    pass1_df = pd.DataFrame(pass1_synthetic)
    print(f"  Pass 1 generated {len(pass1_df)} synthetic samples.")

    # ------------------------------------------------------------------
    # Pass 2 – component response augmentation
    # ------------------------------------------------------------------
    print("\n[Pass 2] Building component response curves from core data...")
    curves = build_response_curves(df, visc_cols)

    all_proteins = [
        pt
        for pt in df["Protein_type"].unique()
        if str(pt).lower() not in ("none", "unknown", "nan")
    ]
    _print_curve_summary(curves, all_proteins)

    print("\n[Pass 2] Generating component-sweep synthetic samples...")
    pass2_synthetic = generate_component_response_samples(
        df,
        curves,
        shear_rates,
        visc_cols,
    )
    pass2_df = pd.DataFrame(pass2_synthetic)
    print(f"  Pass 2 generated {len(pass2_df)} component-sweep synthetic samples.")

    # ------------------------------------------------------------------
    # Pass 3 – protein concentration interpolation
    # ------------------------------------------------------------------
    print("\n[Pass 3] Interpolating protein concentration curves (log-space fit)...")
    pass3_synthetic = generate_protein_conc_interpolation_samples(
        df,
        visc_cols,
    )
    pass3_df = pd.DataFrame(pass3_synthetic)
    print(f"  Pass 3 generated {len(pass3_df)} concentration-interpolation samples.")

    # ------------------------------------------------------------------
    # Combine and save
    # ------------------------------------------------------------------
    augmented_df = pd.concat([df, pass1_df, pass2_df, pass3_df], ignore_index=True)
    augmented_df.to_csv(OUTPUT_FILE, index=False)

    # ------------------------------------------------------------------
    # Post-augmentation report
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("AUGMENTATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Core samples             : {len(df)}")
    print(f"  Pass 1 (profile jitter)  : {len(pass1_df)}")
    print(f"  Pass 2 (component sweep) : {len(pass2_df)}")
    print(f"  Pass 3 (conc interp)     : {len(pass3_df)}")
    print(f"  Total                    : {len(augmented_df)}")

    # Per-protein-type breakdown.
    print("\n  Per-class totals after augmentation:")
    pass1_counts: dict[str, int] = {}
    pass2_counts: dict[str, int] = {}
    pass3_counts: dict[str, int] = {}
    for _, row in pass1_df.iterrows():
        p = normalize_protein_type(row.get(protein_type_col, "unknown"))
        pass1_counts[p] = pass1_counts.get(p, 0) + 1
    for _, row in pass2_df.iterrows():
        p = normalize_protein_type(row.get(protein_type_col, "unknown"))
        pass2_counts[p] = pass2_counts.get(p, 0) + 1
    for _, row in pass3_df.iterrows():
        p = normalize_protein_type(row.get(protein_type_col, "unknown"))
        pass3_counts[p] = pass3_counts.get(p, 0) + 1

    flat_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        p = normalize_protein_type(row.get(protein_type_col, "unknown"))
        y = np.maximum(row[visc_cols].values.astype(float), 1e-6)
        has_flat, _ = detect_flattening(y)
        if has_flat:
            flat_counts[p] = flat_counts.get(p, 0) + 1

    print(
        f"  {'Class':<30} {'core':>5} {'p1':>5} {'p2':>5} {'p3':>5} {'total':>6}  core_flat"
    )
    print("  " + "-" * 72)
    for p in sorted(type_core_counts):
        core_n = type_core_counts[p]
        p1_n = pass1_counts.get(p, 0)
        p2_n = pass2_counts.get(p, 0)
        p3_n = pass3_counts.get(p, 0)
        flat_n = flat_counts.get(p, 0)
        print(
            f"  {p:<30} {core_n:>5} {p1_n:>5} {p2_n:>5} {p3_n:>5} {core_n+p1_n+p2_n+p3_n:>6}  {flat_n}"
        )

    # Component coverage after augmentation.
    print(
        "\n  Component coverage after augmentation (high-conc protein, Viscosity_100):"
    )
    aug_proteins = augmented_df[
        augmented_df["Protein_type"].str.lower().isin([p.lower() for p in all_proteins])
    ]
    for comp_name in ["sucrose", "arginine", "lysine", "proline"]:
        for col, conc_col in [
            ("Stabilizer_type", "Stabilizer_conc"),
            ("Excipient_type", "Excipient_conc"),
        ]:
            mask = aug_proteins[col].str.lower().str.contains(comp_name, na=False)
            sub = aug_proteins[mask & (aug_proteins["Protein_conc"] > 100)]
            if len(sub) == 0:
                continue
            n_concs = sub[conc_col].nunique()
            print(
                f"    {comp_name:<12} (high-conc)  "
                f"n_samples={len(sub):4d}  n_unique_concs={n_concs}"
            )

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
