import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeWarning, curve_fit

warnings.simplefilter("ignore", OptimizeWarning)

# ============================================================
# Viscosity Profile Classification
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

# ============================================================
# Upsampling configuration
# ============================================================

# Each protein type will be upsampled until it reaches this many total
# samples (core + synthetic). Types already at or above this count are
# left untouched (no downsampling of core data).
TARGET_SAMPLES_PER_TYPE: int = 200

# Number of quantile-based concentration bins used to distribute the
# synthetic quota within each protein type.
N_CONCENTRATION_BINS: int = 5


# ============================================================
# Viscosity profile helpers
# ============================================================


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

    A consecutive pair (i, i+1) is considered flat when the viscosity drop
    is less than 2 % of the value at point i — indicating a plateau rather
    than shear thinning.  This catches both flat-start profiles and
    intermediate plateaus (e.g. a Cross-model shoulder at mid-shear).

    Returns
    -------
    has_flat : bool   – True if at least one flat region exists.
    flat_idxs : list  – Indices of the left-hand point in each flat pair.
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
) -> tuple[bool, np.ndarray | None]:
    """
    Attempts a Carreau-Yasuda fit to a viscosity profile.

    Returns
    -------
    fit_success : bool
    popt        : fitted parameters, or None on failure.
    """
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
            maxfev=10000,
        )
        return True, popt
    except RuntimeError:
        return False, None


# ============================================================
# Prior / regime helpers  (unchanged)
# ============================================================


def get_regime_and_prior(row, target_chemical):
    c_class = float(row.get("C_Class", 1.0))
    ph = float(row.get("Buffer_pH", 7.0))
    pi = float(row.get("PI_mean", 7.0))
    if pd.isna(ph):
        ph = 7.0
    if pd.isna(pi):
        pi = 7.0
    delta_ph = abs(ph - pi)
    cci = c_class * np.exp(-delta_ph / 1.5)
    p_type = str(row.get("Protein_class_type", "default")).lower()
    regime = "Far"
    if "mab_igg1" in p_type:
        regime = "Near-pI" if cci >= 0.90 else ("Mixed" if cci >= 0.50 else "Far")
    elif "mab_igg4" in p_type:
        regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.40 else "Far")
    elif any(x in p_type for x in ["fc-fusion", "trispecific"]):
        regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
    elif any(x in p_type for x in ["bispecific", "adc"]):
        regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.45 else "Far")
    elif any(x in p_type for x in ["bsa", "polyclonal"]):
        regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
    else:
        regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
    lookup_key = next(
        (k for k in PRIOR_TABLE if k != "default" and k in p_type), "default"
    )
    regime_dict = PRIOR_TABLE[lookup_key].get(regime, PRIOR_TABLE["default"]["Far"])
    target_chemical = target_chemical.lower()
    if target_chemical in ["sucrose", "trehalose", "stabilizer"]:
        prior_key = "stabilizer"
    elif "arg" in target_chemical:
        prior_key = "arginine"
    elif "lys" in target_chemical:
        prior_key = "lysine"
    elif "pro" in target_chemical:
        prior_key = "proline"
    elif "nacl" in target_chemical:
        prior_key = "nacl"
    else:
        return 0
    return regime_dict.get(prior_key, 0)


# ============================================================
# Synthetic sample generation
# ============================================================


def generate_synthetic_sample(
    row: pd.Series,
    shear_rates: np.ndarray,
    visc_cols: list[str],
    continuous_cols: list[str],
    syn_id: str,
    popt: np.ndarray | None,
    fit_success: bool,
) -> pd.Series:
    """
    Generates one synthetic sample from a core row.

    Method priority
    ---------------
    1. Carreau-Yasuda parameter perturbation  (preferred — preserves profile
       shape including any flattening region).
    2. Log-space noise injection with monotonicity enforcement  (fallback when
       CY fit failed).

    In both cases a small ±2 % jitter is applied to continuous formulation
    features to avoid exact duplicates.
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
        # Enforce monotonic non-increase in log space
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


# ============================================================
# Concentration-bin upsampling
# ============================================================


def assign_concentration_bins(
    group: pd.DataFrame,
    conc_col: str = "Protein_conc",
    n_bins: int = N_CONCENTRATION_BINS,
) -> pd.Series:
    """
    Assigns quantile-based concentration bin labels (0 … k-1) within a
    protein-type group.  Falls back gracefully when there are fewer unique
    concentration values than requested bins.
    """
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
    """
    Distributes `needed` synthetic slots across bins, weighting allocation
    toward bins that are furthest below their equal share of the target.

    Rounding is resolved by assigning leftover slots to the bins with the
    largest fractional remainders (largest-remainder method).
    """
    bins = sorted(bin_sizes)
    deficits = {b: max(0.0, target_per_bin - bin_sizes[b]) for b in bins}
    total_deficit = sum(deficits.values())

    raw: dict[int, float]
    if total_deficit > 0:
        raw = {b: needed * deficits[b] / total_deficit for b in bins}
    else:
        # All bins already at or above target; distribute evenly
        raw = {b: needed / len(bins) for b in bins}

    allocs = {b: int(v) for b, v in raw.items()}
    remainder = needed - sum(allocs.values())
    # Give leftover slots to bins with the largest fractional parts
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

    # --- Guard: if binning failed (all-NaN conc), treat the whole group
    #     as a single bin so upsampling still proceeds.
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

        # Guard: skip bins that somehow ended up empty after filtering
        if len(bin_df) == 0:
            continue

        # Pre-compute flat detection and CY fits
        sample_meta = []
        for _, row in bin_df.iterrows():
            y = np.maximum(row[visc_cols].values.astype(float), 1e-6)
            has_flat, _ = detect_flattening(y)
            fit_ok, popt = fit_carreau_yasuda(y, shear_rates)
            sample_meta.append((row, has_flat, fit_ok, popt))

        flat_pool = [(r, fo, po) for r, hf, fo, po in sample_meta if hf]
        thin_pool = [(r, fo, po) for r, hf, fo, po in sample_meta if not hf]
        priority_queue = flat_pool + thin_pool

        # Guard: if priority_queue is still empty for any reason, skip
        if len(priority_queue) == 0:
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
# Main
# ============================================================


def main():
    input_file = "data/raw/formulation_data_03042026.csv"
    output_file = "data/processed/augmented_formulation_data.csv"
    print(f"Loading {input_file}...")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Please ensure the CSV file is in the same directory.")
        return

    shear_rates = np.array([100, 1000, 10000, 100000, 15000000], dtype=float)
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
    print("\n  Core sample counts by protein type:")
    type_core_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        p = normalize_protein_type(row.get(protein_type_col, "unknown"))
        type_core_counts[p] = type_core_counts.get(p, 0) + 1
    for p in sorted(type_core_counts):
        n = type_core_counts[p]
        status = (
            f"→ needs {TARGET_SAMPLES_PER_TYPE - n} synthetic"
            if n < TARGET_SAMPLES_PER_TYPE
            else "→ already at/above target, no upsampling"
        )
        print(f"    {p:<30s}  core={n:4d}  {status}")

    # ------------------------------------------------------------------
    # Per-protein-type upsampling
    # ------------------------------------------------------------------
    print(
        f"\nUpsampling each protein type to {TARGET_SAMPLES_PER_TYPE} samples "
        f"using {N_CONCENTRATION_BINS} quantile concentration bins "
        f"(flattening-first priority)..."
    )

    all_synthetic: list[pd.Series] = []
    failed_fits_total = 0

    for p_type_raw in df[protein_type_col].unique():
        p_type = normalize_protein_type(p_type_raw)
        group = df[df[protein_type_col] == p_type_raw]

        synthetics = upsample_protein_type_to_target(
            group,
            shear_rates,
            visc_cols,
            continuous_cols,
            target=TARGET_SAMPLES_PER_TYPE,
            n_bins=N_CONCENTRATION_BINS,
        )
        all_synthetic.extend(synthetics)

    synth_df = pd.DataFrame(all_synthetic)
    augmented_df = pd.concat([df, synth_df], ignore_index=True)
    augmented_df.to_csv(output_file, index=False)

    # ------------------------------------------------------------------
    # Post-augmentation report
    # ------------------------------------------------------------------
    print(f"\nSummary:")
    print(f"  Core samples      : {len(df)}")
    print(f"  Synthetic samples : {len(synth_df)}")
    print(f"  Total             : {len(augmented_df)}")

    print("\n  Per-type totals after augmentation:")
    synth_type_counts: dict[str, int] = {}
    for _, row in synth_df.iterrows():
        p = normalize_protein_type(row.get(protein_type_col, "unknown"))
        synth_type_counts[p] = synth_type_counts.get(p, 0) + 1

    flat_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        p = normalize_protein_type(row.get(protein_type_col, "unknown"))
        y = np.maximum(row[visc_cols].values.astype(float), 1e-6)
        has_flat, _ = detect_flattening(y)
        if has_flat:
            flat_counts[p] = flat_counts.get(p, 0) + 1

    for p in sorted(type_core_counts):
        core_n = type_core_counts[p]
        synth_n = synth_type_counts.get(p, 0)
        flat_n = flat_counts.get(p, 0)
        print(
            f"    {p:<30s}  core={core_n:4d}  "
            f"synth={synth_n:4d}  total={core_n + synth_n:4d}  "
            f"core_flat={flat_n:3d}"
        )

    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
