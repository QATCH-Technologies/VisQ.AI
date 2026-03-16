"""
benchmark_o_net.py

Evaluates the O-Net CNP model using the updated inference behavior:
  - No gradient updates during learn() — encode-only, multi-draw averaging
  - Diverse context selection (concentration-stratified) before encoding
  - Held-out evaluation: context and query sets are disjoint per group
  - Metrics computed in log10 space (model-native) and linear cP
  - Per-group uncertainty (std_log10) reported alongside RMSE

Two evaluation modes are run back-to-back:
  1. Leave-One-Out (LOO): For each protein group, learn from all OTHER groups
     and predict the held-out group. Measures true generalization.
  2. In-Group Held-Out (IGHO): Within each group, learn from a 70% context
     split and predict the remaining 30%. Measures within-group fit quality.
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from inference_o_net import ViscosityPredictorCNP
except ImportError:
    print("Error: Could not import ViscosityPredictorCNP from inference_o_net.py.")
    print("Make sure benchmark_o_net.py is in the same folder as inference_o_net.py")
    sys.exit(1)

# ==========================================
# Configuration
# ==========================================
DATA_PATH = "data/raw/formulation_data_02162026.csv"
MODEL_DIR = "models/experiments/o_net_v3"

SHEAR_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]

# Context encoding parameters (must match inference_o_net defaults)
N_DRAWS = 20  # random subsets to average for a stable memory vector
K_CONTEXT = 8  # subset size per draw (few-shot elbow from evaluation)
MAX_CTX_POOL = 15  # max context samples after diversity filtering
IGHO_CTX_FRAC = 0.70  # fraction of each group used as context in IGHO mode
N_UNC_SAMPLES = 30  # draws for uncertainty estimation (lower for speed)


# ==========================================
# Data Loading
# ==========================================
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    int_cols = df.select_dtypes(include=["int", "int64", "int32"]).columns
    for col in int_cols:
        if col != "ID":
            df[col] = df[col].astype(float)
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)
    return df


# ==========================================
# Metrics
# ==========================================
def _log10_safe(arr: np.ndarray) -> np.ndarray:
    """Convert linear cP to log10, clamping non-positive values."""
    return np.log10(np.clip(arr, 1e-6, None))


def calculate_metrics(true_vals: np.ndarray, pred_vals: np.ndarray):
    """
    Returns (rmse_log10, mae_log10, r2_log10, rmse_linear, n).
    All error metrics are in log10 space to match the model's training objective.
    rmse_linear is included as a secondary human-readable metric.
    """
    if len(true_vals) < 2:
        return np.nan, np.nan, np.nan, np.nan, len(true_vals)

    t_log = _log10_safe(true_vals)
    p_log = _log10_safe(pred_vals)

    rmse_log = np.sqrt(mean_squared_error(t_log, p_log))
    mae_log = mean_absolute_error(t_log, p_log)
    r2_log = r2_score(t_log, p_log)
    rmse_lin = np.sqrt(mean_squared_error(true_vals, pred_vals))

    return rmse_log, mae_log, r2_log, rmse_lin, len(true_vals)


def print_metrics_table(title: str, results: dict):
    """Prints a formatted per-shear-rate metrics table."""
    print(f"\n{title}")
    print("=" * 90)
    print(
        f"{'Shear Rate':<22} | {'RMSE (log10)':>12} | {'MAE (log10)':>11} | "
        f"{'R² (log10)':>10} | {'RMSE (cP)':>10} | {'N':>6}"
    )
    print("-" * 90)

    overall_true, overall_pred = [], []

    for col in SHEAR_COLS:
        t_list = results[col]["true"]
        p_list = results[col]["pred"]

        if not t_list:
            print(
                f"{col:<22} | {'N/A':>12} | {'N/A':>11} | {'N/A':>10} | {'N/A':>10} | {0:>6}"
            )
            continue

        t_arr = np.array(t_list)
        p_arr = np.array(p_list)
        rmse_log, mae_log, r2_log, rmse_lin, n = calculate_metrics(t_arr, p_arr)

        print(
            f"{col:<22} | {rmse_log:>12.4f} | {mae_log:>11.4f} | "
            f"{r2_log:>10.4f} | {rmse_lin:>10.2f} | {n:>6}"
        )
        overall_true.extend(t_list)
        overall_pred.extend(p_list)

    print("-" * 90)
    if overall_true:
        g_rmse_log, g_mae_log, g_r2_log, g_rmse_lin, g_n = calculate_metrics(
            np.array(overall_true), np.array(overall_pred)
        )
        print(
            f"{'OVERALL':<22} | {g_rmse_log:>12.4f} | {g_mae_log:>11.4f} | "
            f"{g_r2_log:>10.4f} | {g_rmse_lin:>10.2f} | {g_n:>6}"
        )
    print("=" * 90)


def print_group_uncertainty_table(group_unc: dict):
    """Prints per-group average uncertainty (std_log10) across all shear rates."""
    if not group_unc:
        return
    print("\nPer-Group Context Uncertainty (mean std_log10 across shear rates)")
    print("  [std_log10 interpretation: 0.10 ≈ ±26% factor | 0.20 ≈ ±58% factor]")
    print("-" * 50)
    for protein, std_val in sorted(group_unc.items(), key=lambda x: -x[1]):
        bar = "█" * int(std_val * 50)
        print(f"  {protein:<25} {std_val:.4f}  {bar}")
    print("-" * 50)


# ==========================================
# Encode helper (resets state, applies diverse selection, multi-draw encode)
# ==========================================
def encode_context(predictor: ViscosityPredictorCNP, context_df: pd.DataFrame):
    """
    Resets the predictor state and encodes a context pool using the new
    encode-only learn() with multi-draw averaging and diverse selection.
    """
    predictor.memory_vector = None
    predictor.context_t = None

    if context_df.empty:
        return  # zero-shot fallback

    # Apply diverse context selection if the engine supports it
    if hasattr(predictor, "_select_diverse_context"):
        context_df = predictor._select_diverse_context(context_df, max_k=MAX_CTX_POOL)

    predictor.learn(context_df, n_draws=N_DRAWS, k=K_CONTEXT)


# ==========================================
# Mode 1: Leave-One-Out (LOO)
# ==========================================
def run_loo(predictor: ViscosityPredictorCNP, df: pd.DataFrame) -> dict:
    """
    For each protein group, encodes all OTHER groups as context and predicts
    the held-out group. This is the strictest generalization test.
    """
    groups = df.groupby("Protein_type")
    results = {col: {"true": [], "pred": []} for col in SHEAR_COLS}
    group_unc = {}

    print(f"\n[LOO] Evaluating {len(groups)} protein groups...")

    for protein_name, held_out_df in groups:
        # Context = everything except the held-out protein
        context_df = df[df["Protein_type"] != protein_name].copy()

        print(
            f"  LOO [{protein_name}]: "
            f"context={len(context_df)} samples, query={len(held_out_df)} samples",
            end=" ... ",
        )

        encode_context(predictor, context_df)

        # Prepare query (mask viscosity columns so predict() uses placeholder)
        query_df = held_out_df.drop(columns=SHEAR_COLS, errors="ignore").copy()
        preds_df = predictor.predict(query_df)

        # Uncertainty for this group
        try:
            _, unc_stats = predictor.predict_with_uncertainty(
                query_df, n_samples=N_UNC_SAMPLES, k=K_CONTEXT
            )
            group_unc[protein_name] = float(
                np.mean(unc_stats.get("std_log10", [np.nan]))
            )
        except Exception:
            group_unc[protein_name] = np.nan

        # Collect true vs pred pairs
        for col in SHEAR_COLS:
            pred_col = f"Pred_{col}"
            if col in held_out_df.columns and pred_col in preds_df.columns:
                combined = pd.concat(
                    [
                        held_out_df[col].reset_index(drop=True),
                        preds_df[pred_col].reset_index(drop=True),
                    ],
                    axis=1,
                ).dropna()
                if not combined.empty:
                    results[col]["true"].extend(combined[col].values)
                    results[col]["pred"].extend(combined[pred_col].values)

        print("done.")

    return results, group_unc


# ==========================================
# Mode 2: In-Group Held-Out (IGHO)
# ==========================================
def run_igho(predictor: ViscosityPredictorCNP, df: pd.DataFrame) -> dict:
    """
    Within each protein group, encodes a 70% context split and predicts the
    remaining 30%. Tests within-group fit quality without data leakage.
    """
    groups = df.groupby("Protein_type")
    results = {col: {"true": [], "pred": []} for col in SHEAR_COLS}
    group_unc = {}

    print(f"\n[IGHO] Evaluating {len(groups)} protein groups...")

    for protein_name, group_df in groups:
        n_ctx = max(1, int(len(group_df) * IGHO_CTX_FRAC))

        # Reproducible split: first n_ctx rows as context, rest as query
        ctx_df = group_df.iloc[:n_ctx].copy()
        query_df = group_df.iloc[n_ctx:].copy()

        if query_df.empty:
            print(
                f"  IGHO [{protein_name}]: only {len(group_df)} samples, skipping (need ≥2)."
            )
            continue

        print(
            f"  IGHO [{protein_name}]: "
            f"context={len(ctx_df)}, query={len(query_df)}",
            end=" ... ",
        )

        encode_context(predictor, ctx_df)

        # Mask viscosity for prediction
        query_input = query_df.drop(columns=SHEAR_COLS, errors="ignore").copy()
        preds_df = predictor.predict(query_input)

        # Uncertainty
        try:
            _, unc_stats = predictor.predict_with_uncertainty(
                query_input, n_samples=N_UNC_SAMPLES, k=K_CONTEXT
            )
            group_unc[protein_name] = float(
                np.mean(unc_stats.get("std_log10", [np.nan]))
            )
        except Exception:
            group_unc[protein_name] = np.nan

        for col in SHEAR_COLS:
            pred_col = f"Pred_{col}"
            if col in query_df.columns and pred_col in preds_df.columns:
                combined = pd.concat(
                    [
                        query_df[col].reset_index(drop=True),
                        preds_df[pred_col].reset_index(drop=True),
                    ],
                    axis=1,
                ).dropna()
                if not combined.empty:
                    results[col]["true"].extend(combined[col].values)
                    results[col]["pred"].extend(combined[pred_col].values)

        print("done.")

    return results, group_unc


# ==========================================
# Main
# ==========================================
def main():
    print("=" * 60)
    print("O-Net CNP Benchmark")
    print("=" * 60)
    print(f"Model Dir : {MODEL_DIR}")
    print(f"Data Path : {DATA_PATH}")
    print(f"n_draws={N_DRAWS}, k={K_CONTEXT}, max_ctx_pool={MAX_CTX_POOL}")

    # 1. Initialize predictor
    try:
        predictor = ViscosityPredictorCNP(MODEL_DIR)
        print("Predictor initialized.\n")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return

    # 2. Load data
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        print(str(e))
        return

    groups = df.groupby("Protein_type")
    print(f"Loaded {len(df)} samples across {len(groups)} protein groups.")
    for pname, gdf in groups:
        print(f"  {pname:<30} n={len(gdf)}")

    # 3. Run LOO evaluation
    loo_results, loo_unc = run_loo(predictor, df)
    print_metrics_table(
        "MODE 1 — Leave-One-Out (LOO) | True Generalization", loo_results
    )
    print_group_uncertainty_table(loo_unc)

    # 4. Run IGHO evaluation
    igho_results, igho_unc = run_igho(predictor, df)
    print_metrics_table(
        f"MODE 2 — In-Group Held-Out (IGHO, {int(IGHO_CTX_FRAC*100)}/{int((1-IGHO_CTX_FRAC)*100)} split) | Within-Group Fit",
        igho_results,
    )
    print_group_uncertainty_table(igho_unc)

    # 5. Summary comparison
    def _overall_rmse_log(results):
        all_t, all_p = [], []
        for col in SHEAR_COLS:
            all_t.extend(results[col]["true"])
            all_p.extend(results[col]["pred"])
        if not all_t:
            return np.nan
        return np.sqrt(
            mean_squared_error(
                _log10_safe(np.array(all_t)), _log10_safe(np.array(all_p))
            )
        )

    loo_rmse = _overall_rmse_log(loo_results)
    igho_rmse = _overall_rmse_log(igho_results)

    print("\n--- Summary ---")
    print(f"  LOO  overall RMSE (log10):  {loo_rmse:.4f}")
    print(f"  IGHO overall RMSE (log10):  {igho_rmse:.4f}")
    if not np.isnan(loo_rmse) and not np.isnan(igho_rmse):
        gap = igho_rmse - loo_rmse
        note = (
            "IGHO < LOO — model fits within-group better than it generalizes (expected)"
            if gap < 0
            else (
                "LOO ≈ IGHO — strong cross-group generalization"
                if abs(gap) < 0.02
                else "IGHO > LOO — unexpected; check for data leakage or group imbalance"
            )
        )
        print(f"  Gap (IGHO - LOO):           {gap:+.4f}  [{note}]")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
