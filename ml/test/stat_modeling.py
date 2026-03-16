"""
viscosity_model_grid.py
=======================
Builds a complete statistical model for every combination of:
  Protein x Buffer x Ingredient x Conc_Level

Fallback hierarchy when a cell has insufficient data:
  1. direct         -- enough real observations for this exact cell
  2. class_inferred -- pool proteins sharing the same Protein_class_type
  3. global_average -- pool the entire dataset for this ingredient / conc_level
  4. no_model       -- ingredient has zero observations anywhere in the dataset

Output files:
  viscosity_models.json     -- all model parameters + prediction grids, nested by
                               protein > buffer > ingredient > conc_level
  viscosity_boundaries.json -- per-protein Low/Medium/High concentration cutoffs
"""

import io
import itertools
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
N_WORKERS = 32
POLY_DEGREE = 2
RIDGE_ALPHA = 0.1
N_BOOTSTRAP = 300
MIN_SAMPLES = 4
MIN_UNIQUE_CONC = 2
N_PRED_POINTS = 40

PROTEINS = [
    "Adalimumab",
    "BSA",
    "Nivolumab",
    "Pembrolizumab",
    "Vudalimab",
    "poly-hIgG",
    "Etanercept",
    "Belatacept",
    "Trastuzumab",
    "Ibalizumab",
    "Bevacizumab",
    "BGG",
]
BUFFERS = ["Histidine", "Acetate", "PBS"]
STABILIZERS = ["Sucrose", "Trehalose"]
SURFACTANTS = ["Tween-20", "Tween-80"]
EXCIPIENTS = ["Lysine", "Proline", "Arginine"]
SALTS = ["NaCl"]
ALL_INGREDIENTS = STABILIZERS + SURFACTANTS + EXCIPIENTS + SALTS
CONC_LEVELS = ["Low", "Medium", "High"]

SHEAR_RATE_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]
SHEAR_RATE_VALS = [100, 1_000, 10_000, 100_000, 15_000_000]
LOG_SHEAR = np.log(np.array(SHEAR_RATE_VALS, dtype=float))


# ============================================================
# 1. Data loading & preprocessing
# ============================================================


def _fit_power_law(log_visc_row):
    mask = np.isfinite(log_visc_row)
    if mask.sum() < 2:
        return np.nan, np.nan
    coeffs = np.polyfit(LOG_SHEAR[mask], log_visc_row[mask], 1)
    return np.exp(coeffs[1]), coeffs[0] + 1  # K, n


def load_and_preprocess(path="data/raw/formulation_data_02162026.csv"):
    df = pd.read_csv(path)
    df = df[df["Protein_type"] != "none"].copy()

    def categorize_conc(x):
        if x.nunique() < 2:
            return pd.Series(["Medium"] * len(x), index=x.index)
        try:
            p33, p66 = np.percentile(x, 33.33), np.percentile(x, 66.67)
            if p33 == p66:
                return pd.cut(x, bins=3, labels=["Low", "Medium", "High"])
            return x.apply(
                lambda v: "Low" if v <= p33 else ("Medium" if v <= p66 else "High")
            )
        except Exception:
            return pd.Series(["Medium"] * len(x), index=x.index)

    df["Conc_Level"] = df.groupby("Protein_type")["Protein_conc"].transform(
        categorize_conc
    )

    def is_baseline(row):
        return (
            (row["Salt_type"] == "none" or row["Salt_conc"] == 0)
            and (row["Stabilizer_type"] == "none" or row["Stabilizer_conc"] == 0)
            and (row["Surfactant_type"] == "none" or row["Surfactant_conc"] == 0)
            and (row["Excipient_type"] == "none" or row["Excipient_conc"] == 0)
        )

    df["is_baseline"] = df.apply(is_baseline, axis=1)

    def get_added_ingredient(row):
        if row["is_baseline"]:
            return ("none", 0.0)
        added = []
        for typ, conc in [
            ("Salt_type", "Salt_conc"),
            ("Stabilizer_type", "Stabilizer_conc"),
            ("Surfactant_type", "Surfactant_conc"),
            ("Excipient_type", "Excipient_conc"),
        ]:
            if row[typ] != "none" and row[conc] > 0:
                added.append((row[typ], float(row[conc])))
        return added[0] if len(added) == 1 else ("multiple", 0.0)

    df[["added_ingredient", "added_conc"]] = df.apply(
        lambda r: pd.Series(get_added_ingredient(r)), axis=1
    )
    df = df[df["added_ingredient"] != "multiple"].copy()

    lv = np.log(df[SHEAR_RATE_COLS].clip(lower=1e-5).values)
    K_arr, n_arr = zip(*[_fit_power_law(r) for r in lv])
    df["K"] = K_arr
    df["n"] = n_arr
    return df


def compute_concentration_boundaries(df):
    """
    For each protein, compute the actual min/max mg/mL for each Conc_Level bin.
    Returns a nested dict:
      { protein: { "Low": {"min_mgml": x, "max_mgml": x}, "Medium": {...}, "High": {...} } }
    """
    boundaries = {}
    for protein, grp in df.groupby("Protein_type"):
        boundaries[protein] = {}
        for level, level_grp in grp.groupby("Conc_Level"):
            concs = level_grp["Protein_conc"].dropna()
            boundaries[protein][level] = {
                "min_mgml": round(float(concs.min()), 4),
                "max_mgml": round(float(concs.max()), 4),
                "mean_mgml": round(float(concs.mean()), 4),
                "n_observations": int(len(concs)),
            }
    return boundaries


# ============================================================
# 2. Modelling utilities (module-level for pickling)
# ============================================================


def _make_pipe():
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def _fit_with_bootstrap(X, y):
    pipe = _make_pipe()
    pipe.fit(X, y)
    y_hat = pipe.predict(X)
    r2 = float(r2_score(y, y_hat)) if len(y) > 2 else None
    residual_std = float(np.std(y - y_hat))

    boot_preds, n = [], len(X)
    rng = np.random.default_rng()
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        Xb, yb = X[idx], y[idx]
        if np.unique(Xb).size < 2:
            continue
        pb = _make_pipe()
        pb.fit(Xb, yb)
        boot_preds.append(pb.predict(X))

    mean_boot_std = (
        float(np.mean(np.std(boot_preds, axis=0))) if boot_preds else residual_std
    )
    return pipe, r2, residual_std, mean_boot_std


def _local_derivative(pipe, c):
    eps = max(c * 0.01, 1e-6)
    return float((pipe.predict([[c + eps]]) - pipe.predict([[c - eps]])) / (2 * eps))


def _predict_viscosity(conc_vals, pipe_K, pipe_n, bl_logK, bl_n):
    """Returns (len(conc_vals), len(SHEAR_RATE_VALS)) array of predicted viscosities."""
    X = np.array(conc_vals).reshape(-1, 1)
    abs_logK = bl_logK + pipe_K.predict(X)
    abs_n = bl_n + pipe_n.predict(X)
    log_eta = abs_logK[:, None] + (abs_n[:, None] - 1) * LOG_SHEAR[None, :]
    return np.exp(log_eta)


def _fit_group(modeling_data):
    mdata = modeling_data.dropna(subset=SHEAR_RATE_COLS + ["K", "n"]).copy()
    if len(mdata) < MIN_SAMPLES or mdata["added_conc"].nunique() < MIN_UNIQUE_CONC:
        return None

    bl = mdata[mdata["is_baseline"]]
    bl_logK_mean = (
        float(np.log(bl["K"].clip(lower=1e-10)).mean())
        if not bl.empty
        else float(np.log(mdata["K"].clip(lower=1e-10)).mean())
    )
    bl_n_mean = float(bl["n"].mean()) if not bl.empty else float(mdata["n"].mean())

    delta_logK = np.log(mdata["K"].clip(lower=1e-10).values) - bl_logK_mean
    delta_n = mdata["n"].values - bl_n_mean
    X_feat = mdata[["added_conc"]].values

    pipe_K, r2_K, res_K, bstd_K = _fit_with_bootstrap(X_feat, delta_logK)
    pipe_n, r2_n, res_n, bstd_n = _fit_with_bootstrap(X_feat, delta_n)

    median_conc = float(np.median(mdata["added_conc"].values))

    return {
        # sklearn pipelines — used for prediction only, stripped before JSON serialization
        "pipe_K": pipe_K,
        "pipe_n": pipe_n,
        # Baseline anchors: absolute log K = bl_logK_mean + model_prediction(conc)
        "bl_logK_mean": bl_logK_mean,
        "bl_n_mean": bl_n_mean,
        # Fit quality
        "r2_K": r2_K,
        "r2_n": r2_n,
        "res_std_K": res_K,
        "res_std_n": res_n,
        "boot_std_K": bstd_K,
        "boot_std_n": bstd_n,
        # Augmentation parameters
        # dlogK_dc > 0  → ingredient raises viscosity magnitude
        # dn_dc    < 0  → ingredient increases shear-thinning
        "dlogK_dc": _local_derivative(pipe_K, median_conc),
        "dn_dc": _local_derivative(pipe_n, median_conc),
        # Data provenance
        "median_conc": median_conc,
        "conc_min": float(mdata["added_conc"].min()),
        "conc_max": float(mdata["added_conc"].max()),
        "n_samples": int(len(mdata)),
        "n_baseline": int(len(bl)),
        "n_ingredient": int(len(mdata) - len(bl)),
        "mean_prot_conc": float(mdata["Protein_conc"].mean()),
    }


# ============================================================
# 3. Per-cell worker
# ============================================================


def process_cell(task):
    (protein, buf_type, buf_conc, ingredient, conc_lvl, class_lookup, df_bytes) = task

    df_single = pd.read_parquet(io.BytesIO(df_bytes))

    def _slice(mask):
        return df_single[mask]

    p_mask = df_single["Protein_type"] == protein
    bt_mask = df_single["Buffer_type"] == buf_type
    bc_mask = df_single["Buffer_conc"] == buf_conc
    i_mask = df_single["added_ingredient"] == ingredient
    cl_mask = df_single["Conc_Level"] == conc_lvl
    bl_mask = df_single["is_baseline"]

    ingredient_rows = _slice(p_mask & bt_mask & bc_mask & i_mask & cl_mask)
    baseline_rows = _slice(p_mask & bt_mask & bc_mask & bl_mask & cl_mask)

    fit = _fit_group(pd.concat([baseline_rows, ingredient_rows]))
    model_source = "direct"
    source_class = source_proteins = ""

    # ── CLASS fallback ────────────────────────────────────────────────────────
    if fit is None:
        protein_class = class_lookup.get(protein)
        if protein_class:
            class_proteins = [p for p, c in class_lookup.items() if c == protein_class]
            cp_mask = df_single["Protein_type"].isin(class_proteins)
            fit = _fit_group(
                pd.concat(
                    [
                        _slice(cp_mask & bt_mask & bc_mask & bl_mask & cl_mask),
                        _slice(cp_mask & bt_mask & bc_mask & i_mask & cl_mask),
                    ]
                )
            )
            if fit is not None:
                model_source = "class_inferred"
                source_class = protein_class
                source_proteins = "|".join(class_proteins)
                if not baseline_rows.empty:
                    fit["mean_prot_conc"] = float(baseline_rows["Protein_conc"].mean())

    # ── GLOBAL AVERAGE fallback ───────────────────────────────────────────────
    if fit is None:
        fit = _fit_group(
            pd.concat(
                [
                    _slice(bl_mask & cl_mask),
                    _slice(i_mask & cl_mask),
                ]
            )
        )
        if fit is not None:
            model_source = "global_average"
            source_class = "all"
            source_proteins = "all"
            if not baseline_rows.empty:
                fit["mean_prot_conc"] = float(baseline_rows["Protein_conc"].mean())
                fit["bl_logK_mean"] = float(
                    np.log(baseline_rows["K"].clip(lower=1e-10)).mean()
                )
                fit["bl_n_mean"] = float(baseline_rows["n"].mean())

    if fit is None:
        model_source = "no_model"

    key = (protein, buf_type, buf_conc, ingredient, conc_lvl)
    return key, fit, model_source, source_class, source_proteins


# ============================================================
# 4. JSON serialization helpers
# ============================================================


def _build_prediction_grid(fit):
    """
    Evaluate the model on N_PRED_POINTS across the observed concentration range.
    Returns a dict keyed by shear rate (as string) → list of
    {"added_conc": float, "viscosity_cP": float} dicts.
    """
    conc_grid = np.linspace(fit["conc_min"], fit["conc_max"], N_PRED_POINTS)
    visc_array = _predict_viscosity(
        conc_grid,
        fit["pipe_K"],
        fit["pipe_n"],
        fit["bl_logK_mean"],
        fit["bl_n_mean"],
    )
    predictions = {}
    for si, sr_val in enumerate(SHEAR_RATE_VALS):
        predictions[str(sr_val)] = [
            {"added_conc": round(float(c), 6), "viscosity_cP": round(float(v), 6)}
            for c, v in zip(conc_grid, visc_array[:, si])
        ]
    return predictions


def _fit_to_record(
    fit,
    model_source,
    source_class,
    source_proteins,
    protein_conc_boundaries,
    protein,
    conc_lvl,
):
    """
    Convert an in-memory fit dict into a fully JSON-serializable record.
    sklearn Pipeline objects are consumed here to generate predictions and
    then excluded from the output.
    """
    if fit is None:
        return {
            "model_source": model_source,
            "source_class": source_class,
            "source_proteins": source_proteins,
        }

    prot_bounds = protein_conc_boundaries.get(protein, {}).get(conc_lvl, {})

    return {
        "model_source": model_source,
        "source_class": source_class,
        "source_proteins": source_proteins,
        # ── Data provenance ────────────────────────────────────────────────
        "n_samples": fit["n_samples"],
        "n_baseline": fit["n_baseline"],
        "n_ingredient": fit["n_ingredient"],
        "mean_protein_conc_mgml": round(fit["mean_prot_conc"], 4),
        # ── Protein concentration bin boundaries (mg/mL) ──────────────────
        # These define what Low/Medium/High means for this protein so the
        # augmentation pipeline can map a target concentration to a bin.
        "protein_conc_bin": {
            "level": conc_lvl,
            "min_mgml": prot_bounds.get("min_mgml"),
            "max_mgml": prot_bounds.get("max_mgml"),
            "mean_mgml": prot_bounds.get("mean_mgml"),
        },
        # ── Ingredient concentration range ────────────────────────────────
        "ingredient_conc_range": {
            "min": round(fit["conc_min"], 6),
            "max": round(fit["conc_max"], 6),
            "median": round(fit["median_conc"], 6),
        },
        # ── Baseline power-law anchors ────────────────────────────────────
        # Reconstruct absolute viscosity:
        #   log K_abs = baseline_logK + model_pred(conc)
        #   n_abs     = baseline_n    + model_pred(conc)
        #   eta(gamma_dot) = exp(log K_abs) * gamma_dot^(n_abs - 1)
        "baseline_logK": round(fit["bl_logK_mean"], 6),
        "baseline_n": round(fit["bl_n_mean"], 6),
        # ── Model quality ─────────────────────────────────────────────────
        "model_quality": {
            "r2_logK": round(fit["r2_K"], 4) if fit["r2_K"] is not None else None,
            "r2_n": round(fit["r2_n"], 4) if fit["r2_n"] is not None else None,
            "residual_std_logK": round(fit["res_std_K"], 6),
            "residual_std_n": round(fit["res_std_n"], 6),
            "boot_pred_std_logK": round(fit["boot_std_K"], 6),
            "boot_pred_std_n": round(fit["boot_std_n"], 6),
        },
        # ── PRIMARY AUGMENTATION PARAMETERS ──────────────────────────────
        # dlogK_dc_at_median:
        #   Rate of change of delta_log(K) per unit ingredient concentration,
        #   evaluated at the median observed concentration.
        #   Positive → ingredient increases overall viscosity magnitude.
        # dn_dc_at_median:
        #   Rate of change of delta_n per unit ingredient concentration.
        #   Negative → ingredient increases shear-thinning (flow index decreases).
        "augmentation_params": {
            "dlogK_dc_at_median": round(fit["dlogK_dc"], 8),
            "dn_dc_at_median": round(fit["dn_dc"], 8),
        },
        # ── Prediction grid ───────────────────────────────────────────────
        # Viscosity evaluated at N_PRED_POINTS across the observed conc range,
        # for all 5 shear rates.  Keys are shear rate in s^-1 (as strings).
        "predictions": _build_prediction_grid(fit),
    }


# ============================================================
# 5. Entry point
# ============================================================

if __name__ == "__main__":
    print("Loading and preprocessing data ...")
    df_single = load_and_preprocess("data/raw/formulation_data_02162026.csv")

    # Per-protein concentration bin boundaries
    protein_conc_boundaries = compute_concentration_boundaries(df_single)

    class_lookup = (
        df_single[["Protein_type", "Protein_class_type"]]
        .drop_duplicates()
        .set_index("Protein_type")["Protein_class_type"]
        .to_dict()
    )

    buffer_combos = sorted(
        {
            (row["Buffer_type"], row["Buffer_conc"])
            for _, row in df_single[df_single["Buffer_type"].isin(BUFFERS)].iterrows()
        }
    )
    print(f"Buffer combos found: {buffer_combos}")

    # Serialise dataframe once; workers each deserialise from bytes
    _buf = io.BytesIO()
    df_single.to_parquet(_buf, index=False)
    df_bytes = _buf.getvalue()

    tasks = [
        (protein, buf_type, buf_conc, ingredient, conc_lvl, class_lookup, df_bytes)
        for protein, (buf_type, buf_conc), ingredient, conc_lvl in itertools.product(
            PROTEINS, buffer_combos, ALL_INGREDIENTS, CONC_LEVELS
        )
    ]
    print(f"\nDispatching {len(tasks)} cells across {N_WORKERS} workers ...\n")

    # ── Parallel model fitting ────────────────────────────────────────────────
    raw_results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_cell, t): t for t in tasks}
        with tqdm(
            total=len(tasks), desc="Fitting models", unit="cell", dynamic_ncols=True
        ) as pbar:
            for future in as_completed(futures):
                key, fit, model_source, source_class, source_proteins = future.result()
                raw_results[key] = (fit, model_source, source_class, source_proteins)
                pbar.update(1)

    # ── Assemble nested JSON structure ────────────────────────────────────────
    # Structure: models[protein][buffer_key][ingredient][conc_level] = record
    # buffer_key is "{buffer_type}_{buffer_conc}mM" for readability
    print("\nAssembling JSON output ...")
    models = {}

    for protein, (buf_type, buf_conc), ingredient, conc_lvl in tqdm(
        itertools.product(PROTEINS, buffer_combos, ALL_INGREDIENTS, CONC_LEVELS),
        total=len(tasks),
        desc="Serialising",
        unit="cell",
        dynamic_ncols=True,
    ):
        key = (protein, buf_type, buf_conc, ingredient, conc_lvl)
        fit, model_source, source_class, source_proteins = raw_results[key]

        buf_key = f"{buf_type}_{int(buf_conc)}mM"
        record = _fit_to_record(
            fit,
            model_source,
            source_class,
            source_proteins,
            protein_conc_boundaries,
            protein,
            conc_lvl,
        )

        (
            models.setdefault(protein, {})
            .setdefault(buf_key, {})
            .setdefault(ingredient, {})[conc_lvl]
        ) = record

    # ── Build summary counts ──────────────────────────────────────────────────
    all_sources = [raw_results[k][1] for k in raw_results]
    total = len(all_sources)
    counts = {
        s: all_sources.count(s)
        for s in ["direct", "class_inferred", "global_average", "no_model"]
    }

    print(f"\nGrid summary ({total} total combinations):")
    for src, n in counts.items():
        print(f"  {src:<18} {n:>5}  ({100*n/total:.1f}%)")

    # ── Write viscosity_models.json ───────────────────────────────────────────
    output = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "n_combinations": total,
            "model_counts": counts,
            "config": {
                "poly_degree": POLY_DEGREE,
                "ridge_alpha": RIDGE_ALPHA,
                "n_bootstrap": N_BOOTSTRAP,
                "min_samples": MIN_SAMPLES,
                "min_unique_conc": MIN_UNIQUE_CONC,
                "n_pred_points": N_PRED_POINTS,
                "shear_rates": SHEAR_RATE_VALS,
            },
        },
        "models": models,
    }

    with open("viscosity_models.json", "w") as f:
        json.dump(output, f, indent=2, allow_nan=False)

    # ── Write viscosity_boundaries.json ──────────────────────────────────────
    # Standalone file so downstream code can quickly look up what concentration
    # range a given Conc_Level label corresponds to for each protein.
    boundaries_output = {
        "metadata": {
            "description": (
                "Per-protein Low/Medium/High concentration bin boundaries. "
                "Bins are derived from tertile splits of each protein's observed "
                "Protein_conc values. Use min_mgml/max_mgml to determine which "
                "Conc_Level a target protein concentration falls into."
            ),
            "units": "mg/mL",
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "boundaries": protein_conc_boundaries,
    }

    with open("viscosity_boundaries.json", "w") as f:
        json.dump(boundaries_output, f, indent=2)

    print(f"\nSaved: viscosity_models.json")
    print(f"Saved: viscosity_boundaries.json")
    print("\nAll done.")
