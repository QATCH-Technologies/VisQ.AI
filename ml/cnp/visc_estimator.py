"""
viscosity_estimator.py
======================
Loads viscosity_models.json and viscosity_boundaries.json produced by
viscosity_model_grid.py and estimates full viscosity profiles from
formulation input rows.

Multi-ingredient estimation strategy
-------------------------------------
Each single-ingredient model stores a prediction grid of
  viscosity(added_conc, shear_rate)
evaluated relative to a protein+buffer baseline.  When multiple
ingredients are present, their effects are combined by summing their
individual delta_log_viscosity contributions on top of the shared
protein baseline -- a first-order additivity assumption in log space.

    log eta_total(gamma_dot) = log eta_baseline(gamma_dot)
                               + sum_i [ log eta_ingredient_i(conc_i, gamma_dot)
                                         - log eta_baseline_i(gamma_dot) ]

Usage
-----
    from viscosity_estimator import ViscosityEstimator

    est = ViscosityEstimator("viscosity_models.json", "viscosity_boundaries.json")

    # Single row as a dict
    result = est.estimate(row_dict)

    # Batch from a CSV file
    results_df = est.estimate_csv("formulations.csv")
    results_df.to_csv("predictions.csv", index=False)

CLI
---
    python viscosity_estimator.py formulations.csv predictions.csv
    python viscosity_estimator.py formulations.csv predictions.csv --extrapolate
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Shear rates the models were built on
SHEAR_RATE_VALS = [100, 1_000, 10_000, 100_000, 15_000_000]
SHEAR_RATE_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]
LOG_SHEAR = np.log(np.array(SHEAR_RATE_VALS, dtype=float))

# Ingredient columns in the input CSV
INGREDIENT_PAIRS = [
    ("Salt_type", "Salt_conc"),
    ("Stabilizer_type", "Stabilizer_conc"),
    ("Surfactant_type", "Surfactant_conc"),
    ("Excipient_type", "Excipient_conc"),
]

# Values that mean "no ingredient present"
NONE_VALUES = {"none", "None", "NONE", "", "nan", "NaN"}


# ============================================================
# Data classes
# ============================================================


@dataclass
class IngredientEstimate:
    """Contribution of a single ingredient to the viscosity estimate."""

    ingredient: str
    conc: float
    model_source: str  # direct / class_inferred / global_average / no_model
    conc_in_range: bool  # False if conc was outside the model training range
    delta_log_visc: dict  # {shear_rate_str: delta_log_visc}


@dataclass
class ViscosityEstimate:
    """Full viscosity profile estimate for one input row."""

    row_id: Any
    protein: str
    conc_level: str  # Low / Medium / High (resolved from Protein_conc)
    buffer_key: str  # e.g. "Histidine_20mM"
    baseline_log_visc: dict  # {shear_rate_str: log_visc} with no ingredients
    ingredient_estimates: list
    predicted_viscosity: dict  # {shear_rate_str: cP}
    warnings: list = field(default_factory=list)

    def to_flat_dict(self) -> dict:
        """Flatten to a single dict suitable for a DataFrame row."""
        row = {
            "ID": self.row_id,
            "Protein": self.protein,
            "Conc_Level": self.conc_level,
            "Buffer": self.buffer_key,
        }
        for sr in SHEAR_RATE_VALS:
            row[f"Pred_Viscosity_{sr}"] = self.predicted_viscosity.get(str(sr))

        for ie in self.ingredient_estimates:
            slug = ie.ingredient.replace("-", "_").replace(" ", "_")
            row[f"Model_source_{slug}"] = ie.model_source
            row[f"Conc_in_range_{slug}"] = ie.conc_in_range

        row["Warnings"] = "; ".join(self.warnings) if self.warnings else ""
        return row


# ============================================================
# Estimator
# ============================================================


class ViscosityEstimator:
    """
    Loads pre-fitted viscosity models from JSON and estimates viscosity
    profiles for new formulation inputs.

    Parameters
    ----------
    models_path : str or Path
        Path to viscosity_models.json
    boundaries_path : str or Path
        Path to viscosity_boundaries.json
    extrapolate : bool
        If True, allow estimation outside the trained concentration range
        (with a warning).  If False, clamp to the nearest boundary.
    """

    def __init__(self, models_path, boundaries_path, extrapolate=False):
        self.extrapolate = extrapolate

        with open(models_path) as f:
            raw = json.load(f)
        self._models = raw["models"]
        self._shear_rates = raw["metadata"]["config"]["shear_rates"]

        with open(boundaries_path) as f:
            raw_b = json.load(f)
        self._boundaries = raw_b["boundaries"]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def estimate(self, row: dict) -> ViscosityEstimate:
        """
        Estimate the viscosity profile for a single formulation row.

        Parameters
        ----------
        row : dict
            A single row from the input CSV, keyed by column name.
            String or numeric values are both accepted.

        Returns
        -------
        ViscosityEstimate
        """
        warn_msgs = []
        row_id = row.get("ID", None)
        protein = str(row.get("Protein_type", "")).strip()

        # 1. Resolve protein concentration bin
        try:
            prot_conc = float(row.get("Protein_conc", np.nan))
        except (ValueError, TypeError):
            prot_conc = np.nan

        conc_level = self._resolve_conc_level(protein, prot_conc, warn_msgs)

        # 2. Resolve buffer key
        buf_type = str(row.get("Buffer_type", "")).strip()
        try:
            buf_conc = float(row.get("Buffer_conc", np.nan))
        except (ValueError, TypeError):
            buf_conc = np.nan

        buf_key = _make_buf_key(buf_type, buf_conc)

        # 3. Compute protein+buffer baseline log-viscosity
        baseline_log_visc = self._get_baseline_log_visc(
            protein, buf_key, conc_level, warn_msgs
        )

        # 4. Collect active ingredients
        active_ingredients = _parse_ingredients(row)

        # 5. Per-ingredient delta_log_visc, summed for multi-ingredient additivity
        ingredient_estimates = []
        combined_delta = np.zeros(len(SHEAR_RATE_VALS))

        for ing_name, ing_conc in active_ingredients:
            ie = self._estimate_ingredient(
                protein, buf_key, ing_name, ing_conc, conc_level, warn_msgs
            )
            ingredient_estimates.append(ie)
            delta_arr = np.array(
                [ie.delta_log_visc.get(str(sr), 0.0) for sr in SHEAR_RATE_VALS]
            )
            combined_delta += delta_arr

        # 6. Reconstruct absolute viscosity
        baseline_arr = np.array(
            [baseline_log_visc.get(str(sr), np.nan) for sr in SHEAR_RATE_VALS]
        )
        total_log_visc = baseline_arr + combined_delta
        predicted_viscosity = {
            str(sr): (float(np.exp(lv)) if np.isfinite(lv) else None)
            for sr, lv in zip(SHEAR_RATE_VALS, total_log_visc)
        }

        return ViscosityEstimate(
            row_id=row_id,
            protein=protein,
            conc_level=conc_level,
            buffer_key=buf_key,
            baseline_log_visc=baseline_log_visc,
            ingredient_estimates=ingredient_estimates,
            predicted_viscosity=predicted_viscosity,
            warnings=warn_msgs,
        )

    def estimate_csv(self, csv_path) -> pd.DataFrame:
        """
        Estimate viscosity profiles for every row in a CSV file.

        Returns a DataFrame with one row per input row, containing the
        predicted viscosity at each shear rate plus provenance columns.
        """
        df_in = pd.read_csv(csv_path, dtype=str)
        records = []
        for _, row in df_in.iterrows():
            est = self.estimate(row.to_dict())
            records.append(est.to_flat_dict())
        return pd.DataFrame(records)

    def estimate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate viscosity profiles for every row in an existing DataFrame.
        """
        records = []
        for _, row in df.iterrows():
            est = self.estimate(row.to_dict())
            records.append(est.to_flat_dict())
        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _resolve_conc_level(self, protein, prot_conc, warn_msgs):
        """
        Map a numeric protein concentration (mg/mL) to Low / Medium / High
        using the stored per-protein bin boundaries.

        Falls back to 'Medium' with a warning if boundaries are unavailable
        or the concentration is missing.
        """
        if np.isnan(prot_conc):
            warn_msgs.append(
                f"Protein_conc missing for {protein!r}; defaulting to 'Medium'."
            )
            return "Medium"

        prot_bounds = self._boundaries.get(protein)
        if prot_bounds is None:
            warn_msgs.append(
                f"No concentration boundaries found for {protein!r}; "
                "defaulting to 'Medium'."
            )
            return "Medium"

        # Check for exact bin membership first, then fall back to closest bin
        # by midpoint distance for values between or outside observed ranges.
        best_level = None
        best_dist = np.inf

        for level in ["Low", "Medium", "High"]:
            b = prot_bounds.get(level)
            if b is None:
                continue
            lo, hi = b["min_mgml"], b["max_mgml"]
            if lo <= prot_conc <= hi:
                return level
            mid = (lo + hi) / 2.0
            dist = abs(prot_conc - mid)
            if dist < best_dist:
                best_dist = dist
                best_level = level

        if best_level and best_dist > 0:
            warn_msgs.append(
                f"Protein_conc {prot_conc} mg/mL for {protein!r} falls outside "
                f"all observed bins; assigned to closest bin '{best_level}'."
            )
        return best_level or "Medium"

    def _get_baseline_log_visc(self, protein, buf_key, conc_level, warn_msgs):
        """
        Retrieve the protein+buffer baseline log-viscosity at each shear rate.

        All model records for a given (protein, buffer, conc_level) share the
        same baseline_logK / baseline_n, so we just grab the first valid one.
        """
        buf_models = self._models.get(protein, {}).get(buf_key, {})

        for ing_models in buf_models.values():
            record = ing_models.get(conc_level)
            if record and record.get("baseline_logK") is not None:
                bl_logK = record["baseline_logK"]
                bl_n = record["baseline_n"]
                return {
                    str(sr): float(bl_logK + (bl_n - 1) * np.log(sr))
                    for sr in SHEAR_RATE_VALS
                }

        warn_msgs.append(
            f"No baseline found for protein={protein!r}, buffer={buf_key!r}, "
            f"conc_level={conc_level!r}.  Baseline set to 0 (1 cP); "
            "estimates will reflect ingredient effects only."
        )
        return {str(sr): 0.0 for sr in SHEAR_RATE_VALS}

    def _estimate_ingredient(
        self, protein, buf_key, ing_name, ing_conc, conc_level, warn_msgs
    ):
        """
        Compute delta_log_visc for a single ingredient at a given concentration
        by interpolating the stored prediction grid.

            delta_log_visc(gamma_dot) =
                log eta_ingredient(conc, gamma_dot) - log eta_baseline(gamma_dot)
        """
        record = (
            self._models.get(protein, {})
            .get(buf_key, {})
            .get(ing_name, {})
            .get(conc_level)
        )

        model_source = record.get("model_source", "no_model") if record else "no_model"

        if record is None or model_source == "no_model":
            warn_msgs.append(
                f"No model for ingredient={ing_name!r}, protein={protein!r}, "
                f"buffer={buf_key!r}, conc_level={conc_level!r}.  "
                "Ingredient effect assumed zero."
            )
            return IngredientEstimate(
                ingredient=ing_name,
                conc=ing_conc,
                model_source="no_model",
                conc_in_range=False,
                delta_log_visc={str(sr): 0.0 for sr in SHEAR_RATE_VALS},
            )

        conc_range = record["ingredient_conc_range"]
        conc_min = conc_range["min"]
        conc_max = conc_range["max"]
        conc_in_range = conc_min <= ing_conc <= conc_max

        if not conc_in_range:
            clamp = max(conc_min, min(conc_max, ing_conc))
            msg = (
                f"Ingredient {ing_name!r} conc {ing_conc} outside trained range "
                f"[{conc_min}, {conc_max}]. "
            )
            if self.extrapolate:
                msg += "Extrapolating (extrapolate=True)."
            else:
                msg += f"Clamping to {clamp}."
                ing_conc = clamp
            warn_msgs.append(msg)

        bl_logK = record["baseline_logK"]
        bl_n = record["baseline_n"]
        predictions = record["predictions"]  # {sr_str: [{added_conc, viscosity_cP}]}

        delta_log_visc = {}
        for sr in SHEAR_RATE_VALS:
            sr_str = str(sr)
            grid = predictions.get(sr_str, [])
            if not grid:
                delta_log_visc[sr_str] = 0.0
                continue

            grid_concs = np.array([pt["added_conc"] for pt in grid])
            grid_visc = np.array([pt["viscosity_cP"] for pt in grid])

            interp_visc = float(np.interp(ing_conc, grid_concs, grid_visc))
            log_baseline = bl_logK + (bl_n - 1) * np.log(sr)
            delta_log_visc[sr_str] = np.log(max(interp_visc, 1e-10)) - log_baseline

        return IngredientEstimate(
            ingredient=ing_name,
            conc=ing_conc,
            model_source=model_source,
            conc_in_range=conc_in_range,
            delta_log_visc=delta_log_visc,
        )


# ============================================================
# Module-level helpers
# ============================================================


def _make_buf_key(buf_type: str, buf_conc: float) -> str:
    """Reconstruct the buffer key used as JSON dict keys, e.g. 'Histidine_20mM'."""
    if isinstance(buf_conc, float) and np.isnan(buf_conc):
        return f"{buf_type}_UnknownmM"
    return f"{buf_type}_{int(buf_conc)}mM"


def _parse_ingredients(row: dict) -> list:
    """
    Extract active (ingredient_name, concentration) pairs from an input row.
    Skips any ingredient whose type is in NONE_VALUES or whose concentration
    is zero or missing.
    """
    active = []
    for type_col, conc_col in INGREDIENT_PAIRS:
        ing_type = str(row.get(type_col, "none")).strip()
        if ing_type in NONE_VALUES:
            continue
        try:
            ing_conc = float(row.get(conc_col, 0.0))
        except (ValueError, TypeError):
            ing_conc = 0.0
        if ing_conc > 0:
            active.append((ing_type, ing_conc))
    return active


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    models = "viscosity_models.json"
    boundaries = "viscosity_boundaries.json"
    extrapolate = True
    input_csv = "data/raw/formulation_data_02162026.csv"
    output_csv = "data/processed/estimations.csv"

    print(f"Loading models from {models!r} ...")
    estimator = ViscosityEstimator(models, boundaries, extrapolate=extrapolate)

    print(f"Estimating viscosity profiles from {input_csv!r} ...")
    results = estimator.estimate_csv(input_csv)
    results.to_csv(output_csv, index=False)

    n_warn = (results["Warnings"] != "").sum()
    print(f"Done. {len(results)} rows written to {output_csv!r}.")
    print(f"Rows with warnings: {n_warn} / {len(results)}")
    if n_warn:
        print("\nSample warnings:")
        for w in results.loc[results["Warnings"] != "", "Warnings"].head(5):
            print(f"  {w}")
