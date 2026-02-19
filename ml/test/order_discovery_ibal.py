"""
ibal_sample_selection.py
========================
Determines the optimal order to add Ibalizumab (ibal) samples to the
CrossSampleCNP ICL learner to achieve fastest convergence with lowest
prediction error on unseen data.

Strategy: Greedy Forward Selection
  - At each step, evaluate every remaining (unseen) candidate by temporarily
    adding it to the context, calling learn(), and measuring prediction error
    (log-RMSE) on the rest of the unseen samples.
  - Permanently add the candidate that minimises that error.
  - Repeat until all samples have been selected.

Usage:
    python ibal_sample_selection.py \\
        --model_dir  models/experiments/o_net \\
        --ibal_csv   ibal_eval.csv \\
        --output_dir results/ibal_ordering

Optional:
    --learn_steps   50      (fine-tune steps per candidate evaluation)
    --learn_lr      1e-3    (learning rate for fine-tuning)
    --seed          42      (random seed for reproducibility)
    --no_pretrain           (skip pre-loading non-ibal training data)
    --pretrain_csv  formulation_data_no_ibal.csv
"""

import argparse
import copy
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

try:
    from inference_o_net import ViscosityPredictorCNP
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from inference_o_net import ViscosityPredictorCNP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("IbalSelector")

VISC_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]
PRED_COLS = [f"Pred_{c}" for c in VISC_COLS]


def log_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    log_act = np.log10(np.clip(actual, 1e-6, None))
    log_pred = np.log10(np.clip(predicted, 1e-6, None))
    return float(np.sqrt(np.mean((log_act - log_pred) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    eps = 1e-6
    return float(np.mean(np.abs((actual - predicted) / (actual + eps))) * 100)


def compute_error(results_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    pred_vals, true_vals = [], []
    for pc, vc in zip(PRED_COLS, VISC_COLS):
        if pc in results_df.columns and vc in truth_df.columns:
            pred_vals.append(results_df[pc].values)
            true_vals.append(truth_df[vc].values)
    if not pred_vals:
        return {"log_rmse": np.inf, "mape": np.inf}
    pred_arr = np.concatenate(pred_vals)
    true_arr = np.concatenate(true_vals)
    return {"log_rmse": log_rmse(true_arr, pred_arr), "mape": mape(true_arr, pred_arr)}


def save_model_state(predictor: ViscosityPredictorCNP):
    return {
        "state_dict": copy.deepcopy(predictor.model.state_dict()),
        "memory_vector": (
            predictor.memory_vector.clone()
            if predictor.memory_vector is not None
            else None
        ),
    }


def restore_model_state(predictor: ViscosityPredictorCNP, snapshot: dict):
    predictor.model.load_state_dict(snapshot["state_dict"])
    predictor.model.eval()
    predictor.memory_vector = (
        snapshot["memory_vector"].clone()
        if snapshot["memory_vector"] is not None
        else None
    )


def prepare_df(df: pd.DataFrame, drop_bad_rows: bool = False) -> pd.DataFrame:
    """Normalise dtypes and optionally remove rows that would produce NaN gradients.

    A row is considered 'bad' if:
      - Any viscosity target column is NaN, zero, or negative (log10 undefined), OR
      - Any critical numeric feature (MW, Protein_conc, kP) is NaN.
    These rows cause NaN loss which permanently corrupts model weights.
    """
    df = df.copy()
    int_cols = df.select_dtypes(include=["int", "int64", "int32"]).columns
    for col in int_cols:
        if col != "ID":
            df[col] = df[col].astype(float)
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)

    if drop_bad_rows:
        n_before = len(df)
        # Drop rows where any viscosity column is NaN / <= 0
        visc_mask = pd.Series(True, index=df.index)
        for vc in VISC_COLS:
            if vc in df.columns:
                visc_mask &= df[vc].notna() & (df[vc] > 0)

        # Drop rows where critical numeric features are NaN
        critical_num = [c for c in ["MW", "Protein_conc", "kP"] if c in df.columns]
        num_mask = (
            df[critical_num].notna().all(axis=1)
            if critical_num
            else pd.Series(True, index=df.index)
        )

        df = df[visc_mask & num_mask].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.warning(
                f"prepare_df: dropped {n_dropped} bad rows "
                f"(NaN/zero viscosity or missing MW/Protein_conc/kP). "
                f"{len(df)} rows remain."
            )
    return df


def greedy_forward_selection(
    predictor: ViscosityPredictorCNP,
    ibal_df: pd.DataFrame,
    learn_steps: int = 50,
    learn_lr: float = 1e-3,
) -> list[dict]:
    all_indices = list(ibal_df.index)
    selected_indices = []
    remaining_indices = all_indices.copy()

    selection_log = []
    baseline_snapshot = save_model_state(predictor)

    logger.info(
        f"Starting greedy forward selection over {len(all_indices)} ibal samples."
    )

    step = 0
    while remaining_indices:
        step += 1
        n_remaining = len(remaining_indices)
        n_selected = len(selected_indices)

        logger.info(
            f"\n{'='*60}\n"
            f"Step {step}/{len(all_indices)} | "
            f"Context size: {n_selected} | "
            f"Candidates to evaluate: {n_remaining}"
        )

        t0 = time.time()
        best_candidate_idx = None
        best_error = np.inf
        candidate_errors = {}

        for i, cand_idx in enumerate(remaining_indices):
            trial_indices = selected_indices + [cand_idx]
            context_df = ibal_df.loc[trial_indices].copy()
            holdout_indices = [j for j in remaining_indices if j != cand_idx]
            if not holdout_indices:
                # This is the very last sample; no holdout possible.
                # Assign it a perfect (0) conceptual error so it's selected.
                candidate_errors[cand_idx] = 0.0
                if best_candidate_idx is None:
                    best_candidate_idx = cand_idx
                    best_error = 0.0
                continue

            holdout_df = ibal_df.loc[holdout_indices].copy()
            restore_model_state(predictor, baseline_snapshot)
            predictor.learn(context_df, steps=learn_steps, lr=learn_lr)

            # --- Guard: skip candidate if learning produced NaN weights ---
            params = list(predictor.model.parameters())
            if any(torch.isnan(p).any() for p in params):
                logger.warning(
                    f"  NaN weights after learning candidate "
                    f"{ibal_df.loc[cand_idx, 'ID']} — skipping."
                )
                candidate_errors[cand_idx] = np.inf
                continue

            try:
                results_df = predictor.predict(holdout_df)
                err = compute_error(results_df, holdout_df)["log_rmse"]
                # If predictions are NaN (e.g. memory vector is NaN), treat as inf
                if np.isnan(err):
                    err = np.inf
            except Exception as exc:
                logger.warning(
                    f"  Prediction failed for candidate {ibal_df.loc[cand_idx, 'ID']}: {exc}"
                )
                err = np.inf

            candidate_errors[cand_idx] = err

            if err < best_error:
                best_error = err
                best_candidate_idx = cand_idx

            if (i + 1) % 5 == 0 or (i + 1) == n_remaining:
                elapsed = time.time() - t0
                best_label = (
                    ibal_df.loc[best_candidate_idx, "ID"]
                    if best_candidate_idx is not None
                    else "none yet"
                )
                logger.info(
                    f"  Evaluated {i+1}/{n_remaining} candidates "
                    f"({elapsed:.1f}s elapsed) | "
                    f"Best so far: {best_label} "
                    f"(log-RMSE={best_error:.4f})"
                )
        # --- If every candidate produced inf (all NaN), fall back to first remaining ---
        if best_candidate_idx is None:
            best_candidate_idx = remaining_indices[0]
            best_error = np.inf
            logger.warning(
                f"  Step {step}: all candidates produced NaN/inf — "
                f"falling back to first remaining ({ibal_df.loc[best_candidate_idx, 'ID']}). "
                "Check that the model and pretrain data are valid."
            )

        selected_indices.append(best_candidate_idx)
        remaining_indices.remove(best_candidate_idx)
        restore_model_state(predictor, baseline_snapshot)
        final_context_df = ibal_df.loc[selected_indices].copy()
        predictor.learn(final_context_df, steps=learn_steps, lr=learn_lr)

        # Only advance the baseline if the new learn didn't produce NaN weights
        params = list(predictor.model.parameters())
        if any(torch.isnan(p).any() for p in params):
            logger.warning(
                f"  Step {step}: NaN weights after baseline update — "
                "keeping previous snapshot."
            )
            restore_model_state(predictor, baseline_snapshot)
        else:
            baseline_snapshot = save_model_state(predictor)

        selected_id = ibal_df.loc[best_candidate_idx, "ID"]
        logger.info(
            f"\n>>> Step {step} SELECTED: {selected_id} | "
            f"Holdout log-RMSE = {best_error:.4f}"
        )
        step_record = {
            "step": step,
            "selected_id": selected_id,
            "selected_index": best_candidate_idx,
            "holdout_log_rmse": best_error,
            "context_size": len(selected_indices),
            "candidate_errors": {
                ibal_df.loc[k, "ID"]: v for k, v in candidate_errors.items()
            },
        }
        selection_log.append(step_record)

    logger.info("\nGreedy selection complete.")
    return selection_log


def build_summary_df(selection_log: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in selection_log:
        rows.append(
            {
                "Step": entry["step"],
                "Sample_ID": entry["selected_id"],
                "Context_Size": entry["context_size"],
                "Holdout_LogRMSE": round(entry["holdout_log_rmse"], 5),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_error_df(selection_log: list[dict]) -> pd.DataFrame:
    all_ids = set()
    for entry in selection_log:
        all_ids.update(entry["candidate_errors"].keys())
    all_ids = sorted(all_ids)

    rows = []
    for entry in selection_log:
        row = {"Step": entry["step"], "Selected": entry["selected_id"]}
        for sid in all_ids:
            row[sid] = entry["candidate_errors"].get(sid, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def print_report(summary_df: pd.DataFrame):
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMAL SAMPLE ADDITION ORDER")
    logger.info("=" * 60)
    logger.info(summary_df.to_string(index=False))
    errors = summary_df["Holdout_LogRMSE"].values
    if len(errors) > 3:
        deltas = np.abs(np.diff(errors)) / (errors[:-1] + 1e-9)
        for i, d in enumerate(deltas):
            if d < 0.01:
                logger.info(
                    f"\n>>> Convergence hint: error plateau starts around "
                    f"step {i+2} (sample '{summary_df.iloc[i+1]['Sample_ID']}')"
                )
                break


def main():
    seed = 42
    learn_steps = 50
    learn_lr = 1e-3
    output_dir = r"models\experiments\o_net_no_ibal\benchmarks"
    model_dir = r"models\experiments\o_net_no_ibal"
    no_pretrain = False
    pretrain_csv = r"data/processed/formulation_data_no_ibal.csv"
    ibal_csv = r"data/processed/ibal_eval.csv"
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading model from: {model_dir}")
    predictor = ViscosityPredictorCNP(model_dir)
    logger.info("Model loaded successfully.")
    if not no_pretrain and pretrain_csv and os.path.exists(pretrain_csv):
        logger.info(f"Pre-training on non-ibal data: {pretrain_csv}")
        pretrain_df = prepare_df(pd.read_csv(pretrain_csv), drop_bad_rows=True)
        predictor.learn(pretrain_df, steps=learn_steps, lr=learn_lr)

        # Verify the model didn't go NaN during pre-training
        params = list(predictor.model.parameters())
        if any(torch.isnan(p).any() for p in params):
            logger.error(
                "Pre-training produced NaN model weights! "
                "The pretrain CSV likely still contains problematic rows. "
                "Reinitialising model weights and proceeding zero-shot."
            )
            # Reload a clean model
            predictor = ViscosityPredictorCNP(model_dir)
        else:
            logger.info(f"  Pre-trained on {len(pretrain_df)} samples successfully.")
    else:
        if not no_pretrain and pretrain_csv:
            logger.warning(
                f"pretrain_csv not found at '{pretrain_csv}'. " "Proceeding zero-shot."
            )
        else:
            logger.info("Starting from zero-shot (no pre-training).")

    logger.info(f"Loading ibal eval data: {ibal_csv}")
    ibal_df = prepare_df(pd.read_csv(ibal_csv), drop_bad_rows=True)
    logger.info(f"  Loaded {len(ibal_df)} ibal samples: " f"{ibal_df['ID'].tolist()}")
    logger.info(
        f"\nRunning greedy forward selection | "
        f"learn_steps={learn_steps} | learn_lr={learn_lr}"
    )

    t_start = time.time()
    selection_log = greedy_forward_selection(
        predictor,
        ibal_df,
        learn_steps=learn_steps,
        learn_lr=learn_lr,
    )
    elapsed = time.time() - t_start
    logger.info(f"\nTotal greedy selection time: {elapsed/60:.1f} min")
    summary_df = build_summary_df(selection_log)
    candidate_df = build_candidate_error_df(selection_log)

    summary_path = os.path.join(output_dir, "optimal_order_summary.csv")
    candidate_path = os.path.join(output_dir, "candidate_errors_by_step.csv")

    summary_df.to_csv(summary_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)

    print_report(summary_df)

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  -> {summary_path}")
    logger.info(f"  -> {candidate_path}")
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDED SAMPLE ADDITION ORDER (copy-paste ready):")
    logger.info("=" * 60)
    for row in summary_df.itertuples():
        logger.info(
            f"  {row.Step:>2}. {row.Sample_ID:<8}  "
            f"(holdout log-RMSE after adding = {row.Holdout_LogRMSE:.4f})"
        )


if __name__ == "__main__":
    main()
