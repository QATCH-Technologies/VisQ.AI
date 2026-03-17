"""
find_representative_ibalizumab.py
==================================
Find the 10 ibalizumab samples that best generalise to the full ibalizumab
population when used as CNP context, via greedy forward selection.

Algorithm
---------
At each of 10 steps, evaluate every candidate not yet selected:
  1. Build context tensor from (selected ∪ {candidate}).
  2. Encode → memory vector via model.encode_memory().
  3. Predict the 5 shear-rate viscosities for every *held-out* ibalizumab
     sample using model.decode_from_memory().
  4. Score = mean RMSE over all held-out samples (in physics-scaled space,
     equivalent to RMSE in log10(viscosity)).
Add the candidate with the lowest held-out RMSE and repeat.

Usage
-----
  python find_representative_ibalizumab.py \
      --model_dir  models/experiments/o_net_v3 \
      --data       data/raw/formulation_data_03042026.csv \
      --n_select   10 \
      --out_csv    ibalizumab_top10.csv
"""

import argparse
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Suppress the module-level logging and log-file creation in inference_cnp.py
# by monkey-patching logging.basicConfig before the import.
# ──────────────────────────────────────────────────────────────────────────────
import logging

_orig_basicConfig = logging.basicConfig


def _noop_basicConfig(**kwargs):
    pass  # suppress file handler setup inside inference_cnp on import


logging.basicConfig = _noop_basicConfig  # type: ignore
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the architecture + preprocessor logic from the project script.
from inference_cnp import (  # noqa: E402  (import after path surgery)
    CONC_THRESHOLDS,
    PRIOR_TABLE,
    CrossSampleCNP,
    ViscosityPredictorCNP,
)

logging.basicConfig = _orig_basicConfig  # restore

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PROTEIN_KEY = "Ibalizumab"  # exact case as it appears in the CSV
SHEAR_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]
N_SHEARS = len(SHEAR_COLS)


def load_model(model_dir: str, device: torch.device) -> ViscosityPredictorCNP:
    """Load the predictor (silences its per-instance verbose logging)."""
    predictor = ViscosityPredictorCNP(model_dir, verbose=False)
    return predictor


def preprocess_ibalizumab(predictor: ViscosityPredictorCNP, iba_df: pd.DataFrame):
    """
    Preprocess all ibalizumab samples once.

    Returns
    -------
    all_context_t : Tensor [1, N*5, 2+static_dim]
        Full context tensor (shear, visc, static) for all N samples.
    all_static_t  : Tensor [1, N*5, static_dim]
        Static features only — used as query_static during decoding.
    all_shear_t   : Tensor [1, N*5, 1]
        Scaled log-shear rates — used as query_shear during decoding.
    all_visc_t    : Tensor [1, N*5, 1]
        Scaled log-viscosity — ground truth for scoring.
    """
    static_t, shear_t, visc_t = predictor._preprocess(iba_df)
    # context tensor expected by the encoder: (shear || visc || static)
    context_t = torch.cat([shear_t, visc_t, static_t], dim=-1)
    return context_t, static_t, shear_t, visc_t


def compute_held_out_rmse(
    model: CrossSampleCNP,
    ctx_t: torch.Tensor,
    held_out_indices: list,
    all_static_t: torch.Tensor,
    all_shear_t: torch.Tensor,
    all_visc_t: torch.Tensor,
) -> float:
    """
    Encode ctx_t, then predict every held-out sample and return mean RMSE
    (in physics-scaled space, i.e. proportional to log10-viscosity error).
    """
    with torch.no_grad():
        memory = model.encode_memory(ctx_t)  # [1, latent_dim]
        errors = []
        for j in held_out_indices:
            lo, hi = j * N_SHEARS, (j + 1) * N_SHEARS
            q_shear = all_shear_t[:, lo:hi, :]  # [1, 5, 1]
            q_static = all_static_t[:, lo:hi, :]  # [1, 5, static_dim]
            q_visc = all_visc_t[:, lo:hi, :]  # [1, 5, 1]

            pred = model.decode_from_memory(memory, q_shear, q_static)
            rmse = torch.sqrt(((pred - q_visc) ** 2).mean()).item()
            errors.append(rmse)
    return float(np.mean(errors)) if errors else 0.0


def greedy_select(
    predictor: ViscosityPredictorCNP,
    iba_df: pd.DataFrame,
    n_select: int = 10,
    verbose: bool = True,
):
    """
    Run greedy forward selection and return the ordered list of selected
    row indices (within iba_df) plus a per-step score log.
    """
    device = predictor.device
    model = predictor.model
    model.eval()

    n = len(iba_df)
    if n_select > n:
        raise ValueError(f"n_select={n_select} > n_ibalizumab={n}")

    # ── 1. Preprocess everything once ──────────────────────────────────────
    if verbose:
        print(f"\n[Preprocessing] {n} ibalizumab samples …", flush=True)
    t0 = time.perf_counter()
    all_ctx_t, all_static_t, all_shear_t, all_visc_t = preprocess_ibalizumab(
        predictor, iba_df
    )
    if verbose:
        print(
            f"  Done in {time.perf_counter()-t0:.1f}s  "
            f"(ctx shape: {all_ctx_t.shape})",
            flush=True,
        )

    # ── 2. Greedy forward loop ──────────────────────────────────────────────
    selected = []  # row indices added so far
    step_log = []  # [(step, sample_id, score_after, improvement)]
    prev_score = None

    for step in range(n_select):
        remaining = [i for i in range(n) if i not in selected]

        if verbose:
            print(
                f"\n[Step {step+1}/{n_select}] Evaluating "
                f"{len(remaining)} candidates …",
                flush=True,
            )

        best_idx = None
        best_score = float("inf")

        for cand in remaining:
            candidate_set = selected + [cand]

            # Build candidate context tensor by indexing the pre-processed pool
            ctx_indices = []
            for s in candidate_set:
                ctx_indices.extend(range(s * N_SHEARS, (s + 1) * N_SHEARS))
            ctx_t = all_ctx_t[:, ctx_indices, :]

            # Held-out = everything not in candidate_set
            held_out = [i for i in range(n) if i not in candidate_set]
            if not held_out:
                # All samples selected: no held-out data → score = 0
                best_idx, best_score = cand, 0.0
                continue

            score = compute_held_out_rmse(
                model, ctx_t, held_out, all_static_t, all_shear_t, all_visc_t
            )
            if score < best_score:
                best_score = score
                best_idx = cand

        # Commit best candidate
        selected.append(best_idx)
        improvement = (prev_score - best_score) if prev_score is not None else None
        prev_score = best_score

        sample_id = iba_df.iloc[best_idx]["ID"]
        step_log.append(
            {
                "step": step + 1,
                "sample_idx": best_idx,
                "sample_id": sample_id,
                "held_out_rmse": best_score,
                "improvement": improvement if improvement is not None else float("nan"),
            }
        )

        if verbose:
            imp_str = f"  Δ={improvement:+.5f}" if improvement is not None else ""
            print(
                f"  → Added {sample_id:>6} | "
                f"held-out RMSE = {best_score:.5f}{imp_str}",
                flush=True,
            )

    return selected, step_log


def build_output(iba_df: pd.DataFrame, selected: list, step_log: list) -> pd.DataFrame:
    """Assemble a ranked output DataFrame."""
    rows = []
    for entry in step_log:
        row_in_iba = iba_df.iloc[entry["sample_idx"]]
        rows.append(
            {
                "rank": entry["step"],
                "ID": entry["sample_id"],
                "Protein_conc": row_in_iba.get("Protein_conc", np.nan),
                "Buffer_pH": row_in_iba.get("Buffer_pH", np.nan),
                "Salt_type": row_in_iba.get("Salt_type", np.nan),
                "Salt_conc": row_in_iba.get("Salt_conc", np.nan),
                "Stabilizer_type": row_in_iba.get("Stabilizer_type", np.nan),
                "Stabilizer_conc": row_in_iba.get("Stabilizer_conc", np.nan),
                "Surfactant_type": row_in_iba.get("Surfactant_type", np.nan),
                "Viscosity_100": row_in_iba.get("Viscosity_100", np.nan),
                "Viscosity_1000": row_in_iba.get("Viscosity_1000", np.nan),
                "held_out_rmse": entry["held_out_rmse"],
                "delta_rmse": entry["improvement"],
            }
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Select the N most representative Ibalizumab CNP context samples."
    )
    parser.add_argument(
        "--model_dir",
        default="models/experiments/o_net_v3_debug_aug",
        help="Path to directory with best_model.pth, "
        "preprocessor.pkl, physics_scaler.pkl",
    )
    parser.add_argument(
        "--data",
        default="data/raw/formulation_data_03042026.csv",
        help="Path to formulation CSV (e.g. " "formulation_data_03042026.csv)",
    )
    parser.add_argument(
        "--n_select",
        type=int,
        default=10,
        help="How many representative samples to select (default 10)",
    )
    parser.add_argument(
        "--out_csv",
        default="ibalizumab_top10.csv",
        help="Output CSV path for ranked results",
    )
    parser.add_argument(
        "--protein_key",
        default=PROTEIN_KEY,
        help=f"Protein_type string to filter on (default: {PROTEIN_KEY})",
    )
    args = parser.parse_args()

    # ── Validate paths ───────────────────────────────────────────────────────
    for path in [args.model_dir, args.data]:
        if not os.path.exists(path):
            sys.exit(f"ERROR: Path not found — {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading model from: {args.model_dir}")
    predictor = load_model(args.model_dir, device)
    print(
        f"  static_dim={predictor.static_dim}, "
        f"hidden_dim={predictor.config['hidden_dim']}, "
        f"latent_dim={predictor.config['latent_dim']}"
    )

    # ── Load & filter data ───────────────────────────────────────────────────
    print(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)

    # Normalise types (mirrors inference_cnp __main__ block)
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        if col != "ID":
            df[col] = df[col].astype(float)
    df["ID"] = df["ID"].astype(str)

    iba_df = df[df["Protein_type"].str.lower() == args.protein_key.lower()].copy()
    iba_df = iba_df.reset_index(drop=True)
    print(f"Found {len(iba_df)} '{args.protein_key}' samples.")

    if len(iba_df) < args.n_select:
        sys.exit(
            f"ERROR: Only {len(iba_df)} samples available, "
            f"cannot select {args.n_select}."
        )

    # ── Greedy selection ─────────────────────────────────────────────────────
    t_start = time.perf_counter()
    selected_indices, step_log = greedy_select(
        predictor, iba_df, n_select=args.n_select, verbose=True
    )
    elapsed = time.perf_counter() - t_start

    # ── Results ──────────────────────────────────────────────────────────────
    out_df = build_output(iba_df, selected_indices, step_log)

    print("\n" + "=" * 72)
    print(f"TOP {args.n_select} REPRESENTATIVE IBALIZUMAB SAMPLES (greedy CNP)")
    print("=" * 72)
    print(out_df.to_string(index=False))
    print(f"\nTotal selection time: {elapsed:.1f}s")

    # ── Export ───────────────────────────────────────────────────────────────
    # Full rows from the original CSV for the selected samples
    selected_ids = [entry["sample_id"] for entry in step_log]
    full_selected = df[df["ID"].isin(selected_ids)].copy()
    # Attach rank column
    rank_map = {entry["sample_id"]: entry["step"] for entry in step_log}
    full_selected["cnp_rank"] = full_selected["ID"].map(rank_map)
    full_selected = full_selected.sort_values("cnp_rank")

    full_selected.to_csv(args.out_csv, index=False)
    print(f"\nFull rows saved to: {args.out_csv}")

    # Also save the compact ranking table
    summary_path = args.out_csv.replace(".csv", "_summary.csv")
    out_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")

    # ── Score baselines for context ──────────────────────────────────────────
    # Report how well the final 10-sample set predicts all 34 samples
    print("\n" + "─" * 72)
    print(
        "COVERAGE CHECK: predicting ALL ibalizumab samples with "
        f"the top-{args.n_select} context set"
    )

    predictor.memory_vector = None
    predictor.context_t = None
    predictor.learn(full_selected.drop(columns=["cnp_rank"]))
    results_df = predictor.predict(iba_df)

    log_errors = []
    for _, row in results_df.iterrows():
        for sc in SHEAR_COLS:
            act = row.get(sc, np.nan)
            pred_col = f"Pred_{sc}"
            prd = row.get(pred_col, np.nan)
            if pd.notna(act) and pd.notna(prd) and act > 0 and prd > 0:
                log_errors.append(abs(np.log10(prd) - np.log10(act)))

    if log_errors:
        mae_log = np.mean(log_errors)
        rmse_log = np.sqrt(np.mean(np.array(log_errors) ** 2))
        print(
            f"  MAE  (log10 cP): {mae_log:.4f}   "
            f"({10**mae_log - 1:.0%} median fold error)"
        )
        print(f"  RMSE (log10 cP): {rmse_log:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
