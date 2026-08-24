"""
zero_shot_bench.py
===================
Standalone true zero-shot benchmark (NO context, no learning-curve replay) on
data/processed/Zero-shot testdata.csv -- unusual/novel proteins entirely
absent from training: 38 AB-001/R1-*/R2-* charge-engineered variants plus 8
named clinical antibodies (TGN1412, Basiliximab, Natalizumab, Tremelimumab,
Ipilimumab, Atezolizumab, Ganitumab, Vesencumab), none of which appear in
data/processed/no_ibal.csv. Just "how does the production model do on
unusual proteins."

Two things recorded, per the experiment brief:
  1. High/low bin assessment at a configurable threshold (default 30 cP):
     does the model at least get the coarse regime right, even when the
     exact cP value is off?
  2. The standard MAE/RMSE/MAPE metric set, linear AND log10 space, computed
     per-sample (the per-sample CSV) and aggregated (the summary CSV).

Only Viscosity_1000 is compared against -- it's the only shear rate this
dataset provides (visqai.eval.constants.VISC_COLS is already scoped to just
that column, so eval.metrics.compute_metrics needs no changes here).

The raw CSV calls its net-charge column "Whole_Charge", which
visqai.features.charge.normalize_charge_columns recognizes natively -- no
renaming needed before handing rows to the model.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from visqai.eval.data_prep import prepare_df
from visqai.eval.metrics import compute_metrics
from visqai.eval.plotting.zero_shot_bench import plot_metrics_bars, plot_parity_bin
from visqai.inference.predictor import ViscosityPredictorCNP
from visqai.logging_config import configure_logging

warnings.filterwarnings("ignore")

logger = logging.getLogger("ZeroShotBench")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Standalone external zero-shot benchmark (no context) on unusual proteins."
    )
    ap.add_argument("--data_csv", default=Path("data/processed/Zero-shot testdata 2.csv"))
    ap.add_argument(
        "--model_dir",
        default=Path("models/experiments/final_whole_charge_run"),
        help="Trained checkpoint directory to benchmark.",
    )
    ap.add_argument("--output_dir", default=Path("models/experiments/zero_shot_unusual_proteins"))
    ap.add_argument(
        "--bin_threshold",
        type=float,
        default=30.0,
        help="High/low bin cutoff on Viscosity_1000, in cP.",
    )
    return ap.parse_args(argv)


def _load_zero_shot_df(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Whole_Charge" in df.columns and "Charge" not in df.columns:
        df = df.rename(columns={"Whole_Charge": "Charge"})
    df = prepare_df(df, drop_bad_rows=False)

    # prepare_df's own drop_bad_rows requires EVERY VISC_COLS entry present
    # (wrong here -- this dataset only ever fills Viscosity_1000, so that
    # would drop every row) -- filter by hand on just the column this
    # benchmark actually scores, plus the same numeric-sanity columns
    # prepare_df's drop_bad_rows checks.
    visc_mask = df["Viscosity_1000"].notna() & (df["Viscosity_1000"] > 0)
    crit = [c for c in ["MW", "Protein_conc", "kP"] if c in df.columns]
    num_mask = df[crit].notna().all(axis=1) if crit else pd.Series(True, index=df.index)
    return df[visc_mask & num_mask].reset_index(drop=True)


def _drop_targets(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith("Viscosity_")], errors="ignore")


def _log10_safe(arr: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(arr, 1e-6, None))


def _run_benchmark(
    predictor: ViscosityPredictorCNP, df: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, dict]:
    query_df = _drop_targets(df)
    results = predictor.predict(query_df)

    actual = df["Viscosity_1000"].astype(float).values
    pred = pd.to_numeric(results["Pred_Viscosity_1000"], errors="coerce").astype(float).values

    abs_error = np.abs(actual - pred)
    sq_error = (actual - pred) ** 2
    pct_error = np.abs((actual - pred) / np.clip(actual, 1e-6, None)) * 100.0

    log_actual = _log10_safe(actual)
    log_pred = _log10_safe(pred)
    log_abs_error = np.abs(log_actual - log_pred)
    log_sq_error = (log_actual - log_pred) ** 2

    actual_bin = np.where(actual >= threshold, "high", "low")
    pred_bin = np.where(pred >= threshold, "high", "low")
    bin_correct = actual_bin == pred_bin

    per_sample = pd.DataFrame(
        {
            "ID": df["ID"].values,
            "Protein_type": df["Protein_type"].values if "Protein_type" in df.columns else None,
            "actual": actual,
            "pred": pred,
            "abs_error": abs_error,
            "sq_error": sq_error,
            "pct_error": pct_error,
            "log_abs_error": log_abs_error,
            "log_sq_error": log_sq_error,
            "actual_bin": actual_bin,
            "pred_bin": pred_bin,
            "bin_correct": bin_correct,
        }
    )

    agg = compute_metrics(results, df)
    n = len(df)
    n_high_actual = int((actual_bin == "high").sum())
    n_low_actual = n - n_high_actual
    tp = int(((actual_bin == "high") & (pred_bin == "high")).sum())  # high correctly called high
    tn = int(((actual_bin == "low") & (pred_bin == "low")).sum())  # low correctly called low
    fp = int(((actual_bin == "low") & (pred_bin == "high")).sum())  # low called high
    fn = int(((actual_bin == "high") & (pred_bin == "low")).sum())  # high called low
    summary = {
        **agg,
        "n_samples": n,
        "n_high_actual": n_high_actual,
        "n_low_actual": n_low_actual,
        "bin_accuracy_pct": 100.0 * (tp + tn) / n if n else float("nan"),
        "bin_tp_high_as_high": tp,
        "bin_tn_low_as_low": tn,
        "bin_fp_low_as_high": fp,
        "bin_fn_high_as_low": fn,
    }
    return per_sample, summary


def main(argv=None):
    configure_logging()
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading zero-shot benchmark data: {args.data_csv}")
    df = _load_zero_shot_df(args.data_csv)
    logger.info(
        f"  {len(df)} valid samples (Protein_type values: {sorted(df['Protein_type'].unique())})."
    )

    logger.info(f"Initializing model: {args.model_dir}")
    predictor = ViscosityPredictorCNP(args.model_dir)
    per_sample, summary = _run_benchmark(predictor, df, args.bin_threshold)

    logger.info(
        f"  MAE={summary['mae']:.3f} cP  RMSE={summary['rmse']:.3f} cP  MAPE={summary['mape']:.2f}%  "
        f"MAE(log10)={summary['mae_log10']:.4f}  RMSE(log10)={summary['rmse_log10']:.4f}"
    )
    logger.info(
        f"  Bin accuracy: {summary['bin_accuracy_pct']:.1f}%  "
        f"(high-as-high={summary['bin_tp_high_as_high']}, low-as-low={summary['bin_tn_low_as_low']}, "
        f"low-as-high={summary['bin_fp_low_as_high']}, high-as-low={summary['bin_fn_high_as_low']}, "
        f"n_high={summary['n_high_actual']}, n_low={summary['n_low_actual']})"
    )

    per_sample_path = os.path.join(args.output_dir, "zero_shot_per_sample.csv")
    per_sample.to_csv(per_sample_path, index=False)
    logger.info(f"\nSaved per-sample CSV: {per_sample_path}")

    summary_path = os.path.join(args.output_dir, "zero_shot_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    logger.info(f"Saved summary CSV: {summary_path}")

    parity_path = plot_parity_bin(
        per_sample, args.output_dir, prefix="", threshold=args.bin_threshold
    )
    logger.info(f"Saved parity/bin plot: {parity_path}")

    metrics_path = plot_metrics_bars(summary, args.output_dir, prefix="")
    logger.info(f"Saved metrics plot: {metrics_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
