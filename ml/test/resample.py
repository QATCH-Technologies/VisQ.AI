import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeWarning, curve_fit

warnings.simplefilter("ignore", OptimizeWarning)

CONC_THRESHOLDS = {
    "arginine": 150.0,
    "lysine": 100.0,
    "proline": 200.0,
    "nacl": 150.0,
    "tween-20": 0.01,
    "tween-80": 0.01,
    "stabilizer": 0.2,
    "trehalose": 0.2,
}

PRIOR_TABLE = {
    "mab_igg1": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Far": {
            "arginine": 0,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
    },
    "mab_igg4": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Far": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
    },
    "fc-fusion": {
        "Near-pI": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": 0,
            "nacl": 0,
            "proline": -2,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -2,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
    },
    "bispecific": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
    },
    "adc": {
        "Near-pI": {
            "arginine": -2,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": -1,
            "tween-80": -1,
        },
        "Mixed": {
            "arginine": -1,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": -2,
            "tween-80": -2,
        },
    },
    "bsa": {
        "Near-pI": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
    },
    "polyclonal": {
        "Near-pI": {
            "arginine": -1,
            "lysine": -1,
            "nacl": -1,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
    },
    "default": {
        "Near-pI": {
            "arginine": -1,
            "lysine": -1,
            "nacl": 0,
            "proline": 0,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Mixed": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
        "Far": {
            "arginine": 0,
            "lysine": 0,
            "nacl": 0,
            "proline": -1,
            "stabilizer": 1,
            "tween-20": 0,
            "tween-80": 0,
        },
    },
}

MW_MAP = {
    "sucrose": 342.3,
    "trehalose": 342.3,
    "arginine": 174.2,
    "proline": 115.1,
    "nacl": 58.44,
    "default_sugar": 342.3,
}

# ============================================================
# Viscosity Profile Classification
# ============================================================

# Low-shear viscosity bins (cP) — boundaries chosen to reflect typical
# pharmaceutical formulation ranges where low-viscosity samples dominate.
#
#   very_low  : < 5 cP    (dilute / near-solvent)
#   low       : 5–20 cP   (easily injectable, abundant in datasets)
#   medium    : 20–100 cP (moderate resistance)
#   high      : 100–500 cP (challenging SC delivery)
#   very_high : > 500 cP  (highly concentrated, rare)
VISCOSITY_MAGNITUDE_BINS = [
    ("very_low", 0.0, 5.0),
    ("low", 5.0, 20.0),
    ("medium", 20.0, 100.0),
    ("high", 100.0, 500.0),
    ("very_high", 500.0, np.inf),
]

# Shear-thinning classification based on the ratio of low-shear to
# high-shear viscosity:  r = Viscosity_100 / Viscosity_15000000
#
#   newtonian        : r < 2    (nearly flat profile)
#   mild_thinning    : 2 ≤ r < 10
#   strong_thinning  : r ≥ 10
SHEAR_THINNING_BINS = [
    ("newtonian", 0.0, 2.0),
    ("mild_thinning", 2.0, 10.0),
    ("strong_thinning", 10.0, np.inf),
]

# Maximum synthetic augmentation multiplier to prevent extreme
# oversampling of a single rare profile.
MAX_BALANCE_MULTIPLIER = 8


def classify_viscosity_profile(y_linear: np.ndarray) -> str:
    """
    Assigns a 2D profile class label of the form
    "<magnitude_bin>__<shear_thinning_bin>", e.g. "high__strong_thinning".

    Parameters
    ----------
    y_linear : array of viscosity values at [100, 1000, 10000, 100000, 15000000] s⁻¹.

    Returns
    -------
    str  label used as the distribution-balancing key.
    """
    eta_low = float(y_linear[0])  # 100 s⁻¹
    eta_high = float(y_linear[-1])  # 15 000 000 s⁻¹

    # Magnitude bin
    mag_label = "very_high"
    for label, lo, hi in VISCOSITY_MAGNITUDE_BINS:
        if lo <= eta_low < hi:
            mag_label = label
            break

    # Shear-thinning bin
    ratio = eta_low / max(eta_high, 1e-9)
    st_label = "strong_thinning"
    for label, lo, hi in SHEAR_THINNING_BINS:
        if lo <= ratio < hi:
            st_label = label
            break

    return f"{mag_label}__{st_label}"


def compute_balance_multipliers(
    df: pd.DataFrame,
    visc_cols: list[str],
    base_augmentations: int,
    max_multiplier: int = MAX_BALANCE_MULTIPLIER,
) -> dict[str, int]:
    """
    Computes per-profile-class augmentation counts that push the *synthetic*
    distribution toward uniformity across profile classes.

    Strategy
    --------
    - Count core samples per class  →  n_c
    - Target: every class contributes ~equal synthetic samples.
      target_synth_per_class  = max(n_c) * base_augmentations   (anchored to
                                the most common class, which keeps its base rate)
    - synth_count_c = target_synth_per_class / n_c  per core sample,
      clamped to [base_augmentations, base_augmentations * max_multiplier]
      and rounded to the nearest integer.

    Returns
    -------
    dict mapping profile_class → augmentation_count_per_core_sample
    """
    class_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        y = row[visc_cols].values.astype(float)
        y = np.maximum(y, 1e-6)
        label = classify_viscosity_profile(y)
        class_counts[label] = class_counts.get(label, 0) + 1

    if not class_counts:
        return {}

    max_count = max(class_counts.values())

    multipliers: dict[str, int] = {}
    for label, count in class_counts.items():
        # Ideal total synthetic samples for this class (to match the majority class)
        target_total_synth = max_count * base_augmentations
        # Per core sample, how many synthetics do we need?
        raw_aug = target_total_synth / count
        # Clamp and round
        clamped = int(round(min(raw_aug, base_augmentations * max_multiplier)))
        clamped = max(clamped, base_augmentations)
        multipliers[label] = clamped

    return multipliers


def normalize_protein_type(raw: str) -> str:
    """
    Returns a lowercase, whitespace-stripped canonical protein type key
    suitable for grouping.  No mapping is applied — we group on the raw
    value so that genuinely distinct entries (e.g. 'poly-hIgG' vs
    'mab_igg1') remain separate bins.
    """
    return str(raw).strip().lower()


def compute_intra_type_multipliers(
    df: pd.DataFrame,
    visc_cols: list[str],
    protein_type_col: str,
    base_augmentations: int,
    max_multiplier: int = MAX_BALANCE_MULTIPLIER,
) -> dict[tuple[str, str], int]:
    """
    Computes per-(protein_type, profile_class) augmentation counts so that
    the synthetic distribution is balanced *within* each protein type.

    This prevents a protein type with many low-viscosity samples (e.g.
    poly-hIgG) from flooding the training set with that viscosity regime
    while its higher-viscosity representatives remain scarce.

    Strategy
    --------
    For every protein type T independently:
      - Count core samples per profile class  →  n_{T,c}
      - Anchor to the median class count within T (using median rather than
        max avoids one outlier class dictating the scale for the whole type).
      - target_synth_{T,c} = median_count_T × base_aug / n_{T,c}
      - Clamp to [1, base_aug × max_multiplier] and round.

    Using the median anchor means:
      • Over-represented classes (n > median) get a multiplier < base, i.e.
        they are downsampled relative to the default.
      • Under-represented classes (n < median) are upsampled.
      • The median class gets approximately base_aug (no change).

    Returns
    -------
    dict mapping (protein_type, profile_class) → augmentation_count_per_core_sample
    """
    # Build (protein_type, profile_class) → count
    counts: dict[tuple[str, str], int] = {}
    for _, row in df.iterrows():
        p_type = normalize_protein_type(row.get(protein_type_col, "unknown"))
        y = np.maximum(row[visc_cols].values.astype(float), 1e-6)
        p_class = classify_viscosity_profile(y)
        key = (p_type, p_class)
        counts[key] = counts.get(key, 0) + 1

    # Group counts by protein type to compute per-type median
    type_class_counts: dict[str, dict[str, int]] = {}
    for (p_type, p_class), n in counts.items():
        type_class_counts.setdefault(p_type, {})[p_class] = n

    multipliers: dict[tuple[str, str], int] = {}
    for p_type, class_counts in type_class_counts.items():
        median_count = float(np.median(list(class_counts.values())))
        for p_class, n in class_counts.items():
            raw_aug = (median_count * base_augmentations) / n
            clamped = int(round(min(raw_aug, base_augmentations * max_multiplier)))
            clamped = max(clamped, 1)  # always generate at least one synthetic
            multipliers[(p_type, p_class)] = clamped

    return multipliers


def carreau_yasuda_model(gamma, eta_0, eta_inf, K, a, n):
    gamma = np.maximum(gamma, 1e-6)
    core = 1.0 + np.power(K * gamma, a)
    exponent = (1.0 - n) / a
    return eta_inf + (eta_0 - eta_inf) / np.power(core, exponent)


def get_regime_and_prior(row, target_chemical):
    """Calculates the regime and returns the integer prior (-2 to +1) for a specific chemical."""
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

    # Determine Regime
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

    # Get Prior Table mapping
    lookup_key = next(
        (k for k in PRIOR_TABLE.keys() if k != "default" and k in p_type), "default"
    )
    regime_dict = PRIOR_TABLE[lookup_key].get(regime, PRIOR_TABLE["default"]["Far"])

    # Map target chemical to prior key
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
        return 0  # Default neutral

    return regime_dict.get(prior_key, 0)


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

    base_augmentations = 2
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Pre-compute distribution-balancing multipliers.
    #
    # Only core (real) samples are considered here — no synthetic rows
    # are included in the distribution analysis, so the balance weights
    # reflect the true underlying data distribution.
    # ------------------------------------------------------------------
    print("Analysing viscosity profile distribution...")
    balance_multipliers = compute_balance_multipliers(
        df, visc_cols, base_augmentations, MAX_BALANCE_MULTIPLIER
    )

    protein_type_col = "Protein_class_type"
    print("Analysing intra-protein-type profile distribution...")
    intra_type_multipliers = compute_intra_type_multipliers(
        df, visc_cols, protein_type_col, base_augmentations, MAX_BALANCE_MULTIPLIER
    )

    # Print distribution summary for transparency
    print("\n  Global profile class distribution (core samples):")
    class_counts_summary: dict[str, int] = {}
    for _, row in df.iterrows():
        y = np.maximum(row[visc_cols].values.astype(float), 1e-6)
        lbl = classify_viscosity_profile(y)
        class_counts_summary[lbl] = class_counts_summary.get(lbl, 0) + 1

    for lbl in sorted(class_counts_summary):
        n = class_counts_summary[lbl]
        m = balance_multipliers.get(lbl, base_augmentations)
        print(f"    {lbl:<38s}  core={n:4d}  global_synth_per_core={m:2d}")

    print("\n  Intra-type profile distribution (core samples):")
    # Group for the per-type report
    type_summary: dict[str, dict[str, int]] = {}
    for _, row in df.iterrows():
        p_type = normalize_protein_type(row.get(protein_type_col, "unknown"))
        y = np.maximum(row[visc_cols].values.astype(float), 1e-6)
        lbl = classify_viscosity_profile(y)
        type_summary.setdefault(p_type, {})[lbl] = (
            type_summary.get(p_type, {}).get(lbl, 0) + 1
        )

    for p_type in sorted(type_summary):
        print(f"    [{p_type}]")
        for lbl in sorted(type_summary[p_type]):
            n = type_summary[p_type][lbl]
            global_m = balance_multipliers.get(lbl, base_augmentations)
            intra_m = intra_type_multipliers.get((p_type, lbl), base_augmentations)
            final_m = min(global_m, intra_m)
            print(
                f"      {lbl:<38s}  core={n:4d}  "
                f"global={global_m:2d}  intra={intra_m:2d}  "
                f"→ effective={final_m:2d}"
            )
    print()

    synthetic_rows = []
    failed_fits = 0
    flat_samples_detected = 0

    print("Generating synthetic data...")

    for _, row in df.iterrows():
        y_linear = row[visc_cols].values.astype(float)
        y_linear = np.maximum(y_linear, 1e-6)

        # Classify this core sample to look up its balance multiplier
        profile_class = classify_viscosity_profile(y_linear)
        p_type = normalize_protein_type(row.get(protein_type_col, "unknown"))

        global_aug = balance_multipliers.get(profile_class, base_augmentations)
        intra_aug = intra_type_multipliers.get(
            (p_type, profile_class), base_augmentations
        )

        # Final balance weight: the more restrictive of the two axes.
        # - global_aug  lifts globally under-represented profile shapes.
        # - intra_aug   suppresses profiles that are over-represented
        #               *within their own protein type* (e.g. poly-hIgG
        #               low-viscosity), even when they look globally rare.
        # Taking min() ensures neither dimension is silently ignored.
        balance_aug = min(global_aug, intra_aug)

        # --- PLATEAU DETECTION (all adjacent shear-rate pairs) ---
        # A region is considered flat when the drop between two consecutive
        # points is less than 2 % of the left-hand value.  We scan every
        # adjacent pair so intermediate plateaus (e.g. 1 000–10 000 s⁻¹)
        # are caught alongside the classical flat-start case.
        flat_regions = [
            i
            for i in range(len(y_linear) - 1)
            if (y_linear[i] - y_linear[i + 1]) < (0.02 * y_linear[i])
        ]
        has_flat_region = len(flat_regions) > 0
        is_flat_start = 0 in flat_regions  # kept for the forced-flat gate below

        if has_flat_region:
            flat_samples_detected += 1
            # Scale boost by how many flat regions exist so profiles with
            # multiple plateaus receive proportionally more augmentation,
            # then take the larger of that or the distribution balance weight.
            flat_boost = base_augmentations * (3 + len(flat_regions) * 2)
            current_augmentations = max(flat_boost, balance_aug)
        else:
            current_augmentations = balance_aug

        # Standard Carreau-Yasuda initial guesses
        eta_0_guess = max(y_linear[0], y_linear[-1] + 0.1)
        eta_inf_guess = y_linear[-1]
        p0 = [eta_0_guess, eta_inf_guess, 0.001, 2.0, 0.5]
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
            fit_success = True
        except RuntimeError:
            fit_success = False
            failed_fits += 1

        # -------------------------------------------------------------
        # 1. GENERATE STANDARD AUGMENTATIONS
        #    Count is now governed by the balance multiplier (and the
        #    flat-start override where applicable) instead of a fixed 2.
        # -------------------------------------------------------------
        for i in range(current_augmentations):
            new_row = row.copy()
            new_row["ID"] = f"{row['ID']}_synth_{i+1}"

            if fit_success:
                perturbed_popt = popt * np.random.normal(1.0, 0.03, size=len(popt))
                perturbed_popt = np.maximum(perturbed_popt, 1e-6)
                perturbed_popt[4] = min(perturbed_popt[4], 1.0)
                new_y = carreau_yasuda_model(shear_rates, *perturbed_popt)
                new_row[visc_cols] = new_y
            else:
                y_log = np.log10(y_linear)
                noise = np.random.normal(0, 0.02, size=len(y_log))
                synth_y_log = y_log + noise

                for j in range(1, len(synth_y_log)):
                    if synth_y_log[j] > synth_y_log[j - 1]:
                        synth_y_log[j] = synth_y_log[j - 1]

                interpolator = PchipInterpolator(np.log10(shear_rates), synth_y_log)
                new_y = 10 ** interpolator(np.log10(shear_rates))
                new_row[visc_cols] = new_y

            for col in continuous_cols:
                val = new_row[col]
                if pd.notnull(val) and val > 0:
                    new_row[col] = val * np.random.normal(1.0, 0.02)

            synthetic_rows.append(new_row)

        # -------------------------------------------------------------
        # 2. GENERATE FORCED FLAT SCENARIOS (For non-flat-start curves only)
        #    Injecting a synthetic flat start is only meaningful when the
        #    real curve does not already begin flat — using is_flat_start
        #    (index 0 specifically) preserves that intent while still
        #    allowing the broader has_flat_region boost above.
        # -------------------------------------------------------------
        if not is_flat_start:
            num_forced_flat = 2
            for i in range(num_forced_flat):
                new_row = row.copy()
                new_row["ID"] = f"{row['ID']}_forced_flat_{i+1}"

                y_synth = y_linear.copy() * np.random.normal(
                    1.0, 0.02, size=len(y_linear)
                )
                plateau_end_idx = np.random.choice([1, 2])

                for j in range(plateau_end_idx + 1):
                    y_synth[j] = y_synth[0]
                for j in range(plateau_end_idx + 1, len(y_synth)):
                    if y_synth[j] >= y_synth[j - 1]:
                        y_synth[j] = y_synth[j - 1] * 0.98

                new_row[visc_cols] = y_synth

                for col in continuous_cols:
                    val = new_row[col]
                    if pd.notnull(val) and val > 0:
                        new_row[col] = val * np.random.normal(1.0, 0.02)

                synthetic_rows.append(new_row)

    synth_df = pd.DataFrame(synthetic_rows)
    augmented_df = pd.concat([df, synth_df], ignore_index=True)
    augmented_df.to_csv(output_file, index=False)

    # ------------------------------------------------------------------
    # Post-augmentation distribution report
    # ------------------------------------------------------------------
    print(f"\nSummary:")
    print(f"  Core samples          : {len(df)}")
    print(f"  Synthetic samples     : {len(synth_df)}")
    print(f"  Total                 : {len(augmented_df)}")
    print(f"  Failed CY fits        : {failed_fits}")
    print(f"  Any flat region found : {flat_samples_detected}")

    print("\n  Synthetic sample distribution by protein type and profile class:")
    synth_type_class_counts: dict[str, dict[str, int]] = {}
    for _, row in synth_df.iterrows():
        p_type = normalize_protein_type(row.get(protein_type_col, "unknown"))
        y = np.maximum(row[visc_cols].values.astype(float), 1e-6)
        lbl = classify_viscosity_profile(y)
        synth_type_class_counts.setdefault(p_type, {})[lbl] = (
            synth_type_class_counts.get(p_type, {}).get(lbl, 0) + 1
        )

    all_p_types = sorted(set(list(type_summary) + list(synth_type_class_counts)))
    for p_type in all_p_types:
        core_classes = type_summary.get(p_type, {})
        synth_classes = synth_type_class_counts.get(p_type, {})
        all_lbls = sorted(set(list(core_classes) + list(synth_classes)))
        print(f"    [{p_type}]")
        for lbl in all_lbls:
            core_n = core_classes.get(lbl, 0)
            synth_n = synth_classes.get(lbl, 0)
            print(
                f"      {lbl:<38s}  core={core_n:4d}  synth={synth_n:5d}  "
                f"total={core_n + synth_n:5d}"
            )

    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
