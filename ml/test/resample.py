import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeWarning, curve_fit

# Suppress the covariance warning since we have 0 degrees of freedom on 5 points
warnings.simplefilter("ignore", OptimizeWarning)

# ViscosityEstimator is optional — gracefully falls back if JSON models are absent
try:
    from visc_estimator import SHEAR_RATE_VALS as _EST_SR_VALS
    from visc_estimator import ViscosityEstimator

    print(
        "Estimator module found. Attempting to load viscosity models for sucrose titration augmentations..."
    )
    _ESTIMATOR_AVAILABLE = True
except ImportError:
    _ESTIMATOR_AVAILABLE = False

VISC_MODELS_PATH = "viscosity_models.json"
VISC_BOUNDARIES_PATH = "viscosity_boundaries.json"

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
# Sucrose multiplier via ViscosityEstimator
# ============================================================


def _get_sucrose_multipliers(estimator, row, conc_M, shear_rates):
    """
    Query the ViscosityEstimator for the per-shear-rate viscosity multiplier
    that sucrose at `conc_M` (molar) produces for the protein/buffer context
    described by `row`.

    Returns a numpy array of shape (len(shear_rates),) where each element is
    the multiplicative factor by which the baseline viscosity changes at that
    shear rate:
        multiplier[i] = exp(delta_log_visc[shear_rates[i]])
                      = eta_with_sucrose[sr] / eta_baseline[sr]

    This replaces the arbitrary scalar (dilution * shielding * crowding) used
    previously for sucrose, providing shear-rate-resolved estimates grounded
    in the statistical model built from real experimental data.

    Returns None if the estimator cannot produce a valid estimate, so the
    caller can fall back to the legacy scalar formula.
    """
    probe = {
        "ID": row.get("ID", "probe"),
        "Protein_type": row.get("Protein_type", ""),
        "Protein_conc": row.get("Protein_conc", np.nan),
        "Buffer_type": row.get("Buffer_type", ""),
        "Buffer_conc": row.get("Buffer_conc", np.nan),
        "Stabilizer_type": "Sucrose",
        "Stabilizer_conc": conc_M,
        # Remaining ingredient slots inactive
        "Salt_type": "none",
        "Salt_conc": 0.0,
        "Surfactant_type": "none",
        "Surfactant_conc": 0.0,
        "Excipient_type": "none",
        "Excipient_conc": 0.0,
    }

    try:
        est_result = estimator.estimate(probe)
    except Exception:
        return None

    # Find the sucrose IngredientEstimate
    sucrose_ie = next(
        (
            ie
            for ie in est_result.ingredient_estimates
            if ie.ingredient.lower() == "sucrose"
        ),
        None,
    )
    if sucrose_ie is None:
        return None

    # Convert delta_log_visc to a per-shear-rate multiplier array aligned to
    # the shear_rates used in this script (not necessarily the estimator's list)
    multipliers = np.ones(len(shear_rates))
    for i, sr in enumerate(shear_rates):
        delta = sucrose_ie.delta_log_visc.get(str(int(sr)))
        if delta is not None and np.isfinite(delta):
            multipliers[i] = np.exp(delta)

    # Sanity check: multipliers should be positive and not wildly extreme
    if np.any(multipliers <= 0) or np.any(multipliers > 1e6):
        return None

    return multipliers


def _legacy_sucrose_multiplier(row, conc_M, prior_val):
    """
    Original scalar physics multiplier for sucrose (fallback when estimator
    is unavailable or returns None).
    """
    base_protein = float(row.get("Protein_conc", 0.0))
    mw = MW_MAP.get("sucrose", 342.3)
    mass_mg_ml = conc_M * mw
    dilution = max(0.8, 1.0 - (mass_mg_ml / 1000.0))
    shielding = np.exp(-1.0 * conc_M)
    crowding = np.exp(0.0000005 * base_protein * (mass_mg_ml**2) * prior_val)
    return dilution * shielding * crowding


# ============================================================
# Main
# ============================================================


def main():
    input_file = "data/raw/formulation_data_02162026.csv"
    output_file = "data/processed/augmented_formulation_data.csv"
    print(f"Loading {input_file}...")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Please ensure the CSV file is in the same directory.")
        return

    # ── Initialise the viscosity estimator (sucrose titrations only) ──────────
    estimator = None
    if _ESTIMATOR_AVAILABLE:
        models_ok = Path(VISC_MODELS_PATH).exists()
        bounds_ok = Path(VISC_BOUNDARIES_PATH).exists()
        if models_ok and bounds_ok:
            try:
                estimator = ViscosityEstimator(
                    VISC_MODELS_PATH, VISC_BOUNDARIES_PATH, extrapolate=True
                )
                print(
                    "ViscosityEstimator loaded — sucrose titrations will use "
                    "model-derived per-shear-rate multipliers."
                )
            except Exception as e:
                print(
                    f"Warning: could not load ViscosityEstimator ({e}). "
                    "Falling back to legacy sucrose formula."
                )
        else:
            missing = [
                p
                for p, ok in [
                    (VISC_MODELS_PATH, models_ok),
                    (VISC_BOUNDARIES_PATH, bounds_ok),
                ]
                if not ok
            ]
            print(
                f"Warning: {missing} not found. "
                "Falling back to legacy sucrose formula."
            )
    else:
        print(
            "Warning: visc_estimator module not found. "
            "Falling back to legacy sucrose formula."
        )

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

    synthetic_rows = []
    base_augmentations = 2
    failed_fits = 0
    flat_samples_detected = 0
    titrations_generated = 0
    # Track how often each sucrose multiplier source was used
    sucrose_model_hits = 0
    sucrose_fallback_hits = 0

    np.random.seed(42)

    print(
        "Generating synthetic data "
        "(Carreau-Yasuda + PCHIP + Forced Flats + Titration Sweeps)..."
    )

    for idx, row in df.iterrows():
        y_linear = row[visc_cols].values.astype(float)
        y_linear = np.maximum(y_linear, 1e-6)
        base_protein = float(row.get("Protein_conc", 0.0))

        # --- FLAT PLATEAU DETECTION ---
        is_flat_start = (y_linear[0] - y_linear[1]) < (0.02 * y_linear[0])

        if is_flat_start:
            flat_samples_detected += 1
            current_augmentations = base_augmentations * 5
        else:
            current_augmentations = base_augmentations

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
        # 1. GENERATE STANDARD AUGMENTATIONS (Upsampled if flat)
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
        # 2. GENERATE FORCED FLAT SCENARIOS (For non-flat curves)
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

        # -------------------------------------------------------------
        # PRIOR-DIRECTED TITRATION SWEEPS
        # -------------------------------------------------------------
        if base_protein >= 100.0:
            titration_chemicals = [
                ("Stabilizer_type", "Stabilizer_conc", "Sucrose"),
                ("Excipient_type", "Excipient_conc", "Arginine"),
            ]
            molar_sweeps = [0.1, 0.25, 0.4, 0.5]

            for type_col, conc_col, chem_name in titration_chemicals:
                prior_val = get_regime_and_prior(row, chem_name)
                mw = MW_MAP.get(chem_name.lower(), 150.0)

                for t_idx, conc_M in enumerate(molar_sweeps):
                    new_row = row.copy()
                    new_row["ID"] = f"{row['ID']}_titrate_{chem_name}_{t_idx+1}"

                    # Inject chemical
                    if pd.isna(new_row[type_col]) or str(new_row[type_col]).lower() in [
                        "none",
                        "unknown",
                    ]:
                        new_row[type_col] = chem_name
                    new_row[conc_col] = conc_M

                    # ----------------------------------------------------------
                    # SUCROSE: use model-derived per-shear-rate multipliers
                    # ----------------------------------------------------------
                    if chem_name == "Sucrose":
                        model_multipliers = None
                        if estimator is not None:
                            model_multipliers = _get_sucrose_multipliers(
                                estimator, row, conc_M, shear_rates
                            )

                        if model_multipliers is not None:
                            # Model path — each shear rate gets its own multiplier
                            # derived from exp(delta_log_visc[sr]), reflecting the
                            # shear-rate-resolved crowding effect captured in the
                            # statistical model (changes to both K and n).
                            # Small Gaussian noise is added in log space to maintain
                            # the same sample diversity as the rest of the pipeline.
                            noise = np.random.normal(0.0, 0.02, size=len(shear_rates))
                            per_sr_multipliers = model_multipliers * np.exp(noise)
                            new_y = y_linear * per_sr_multipliers
                            sucrose_model_hits += 1
                        else:
                            # Legacy fallback — scalar multiplier, all shear rates equal
                            scalar = _legacy_sucrose_multiplier(row, conc_M, prior_val)
                            new_y = y_linear * scalar
                            sucrose_fallback_hits += 1

                    # ----------------------------------------------------------
                    # ARGININE (and any other chemical): original physics formula
                    # ----------------------------------------------------------
                    else:
                        mass_mg_ml = conc_M * mw
                        dilution = max(0.8, 1.0 - (mass_mg_ml / 1000.0))

                        if prior_val > 0:
                            shielding = np.exp(-1.0 * conc_M)
                            crowding = np.exp(
                                0.0000005 * base_protein * (mass_mg_ml**2) * prior_val
                            )
                        elif prior_val < 0:
                            shielding = np.exp(prior_val * 0.8 * conc_M)
                            crowding = 1.0
                        else:
                            shielding = 1.0
                            crowding = 1.0

                        physics_multiplier = dilution * shielding * crowding
                        new_y = y_linear * physics_multiplier

                    new_row[visc_cols] = new_y
                    titrations_generated += 1
                    synthetic_rows.append(new_row)

    synth_df = pd.DataFrame(synthetic_rows)
    augmented_df = pd.concat([df, synth_df], ignore_index=True)
    augmented_df.to_csv(output_file, index=False)

    print(f"Generated {titrations_generated} Prior-Directed Titrations.")
    if sucrose_model_hits + sucrose_fallback_hits > 0:
        print(
            f"  Sucrose multiplier source — "
            f"model: {sucrose_model_hits}, "
            f"legacy fallback: {sucrose_fallback_hits}"
        )


if __name__ == "__main__":
    main()
