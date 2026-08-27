from __future__ import annotations

import argparse
import inspect
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from visqai.eval.metrics import compute_metrics
from visqai.eval.style import mpl, apply_style, style_axis
from visqai import constants, paths
from visqai.constants import C_DEEP_BLUE, C_TEXT, C_MUTED, C_BORDER, C_WHITE
from visqai.features.dataprocessor import prepare_df
from visqai.inference.predictor import ViscosityPredictorCNP
from visqai.logging_config import configure_logging

warnings.filterwarnings("ignore")

logger = logging.getLogger("ZeroShotEval")


ACCESS_LOG = constants.MODELS_ROOT / "heldout_access.log"


@dataclass(frozen=True)
class HeldoutPanel:
    name: str
    path: Path
    contaminated: bool
    note: str


HELDOUT_PANELS: dict[str, HeldoutPanel] = {
    "zero_shot_panel": HeldoutPanel(
        name="zero_shot_panel",
        path=constants.DATA_ROOT / "processed" / "zero-shot-data.csv",
        contaminated=True,
        note=(
            "46-row molecule-discrimination panel: two engineered variant "
            "families (AB-*/R1-* = R1_family, R2-* = R2_family) plus 8 clinical "
            "mAbs. Includes Tremelimumab (Protein_class_type=mAb_IgG2), an "
            "ISOTYPE CLASS absent from training (train classes: IgG1, IgG4, "
            "Bispecific, Fc-Fusion, Polyclonal, Other) -- flag as unseen-class, "
            "not merely unseen-protein, when reporting results that include it. "
            "Already the subject of extensive EDA (the r=-0.84 pooled Fv-charge "
            "correlation finding, and Task 1's checks 4-5 reproducing it as a "
            "3-block contrast) before this registry existed -- CONTAMINATED. "
            "Any metric computed from this panel is an in-sample estimate of "
            "that EDA, not a held-out validation number, until a fresh, "
            "never-inspected panel exists to replace it."
        ),
    ),
}


class HeldoutPanelAccessError(RuntimeError):
    """Raised when code tries to load a registered held-out panel without an
    explicit purpose="final_eval"."""


def _log_access(name: str, purpose: str | None, granted: bool, caller: str) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{ts}\t{name}\tpurpose={purpose!r}\tgranted={granted}\tcaller={caller}\n"
    with open(ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def _calling_module() -> str:
    for frame in inspect.stack()[2:]:
        mod = frame.frame.f_globals.get("__name__", "?")
        if mod != __name__:
            return f"{mod}:{frame.function}:{frame.lineno}"
    return "?"


def load_heldout_panel(name: str, purpose: str | None = None) -> tuple[pd.DataFrame, dict]:
    caller = _calling_module()

    if name not in HELDOUT_PANELS:
        _log_access(name, purpose, False, caller)
        raise HeldoutPanelAccessError(
            f"Unknown held-out panel {name!r}. Registered: {sorted(HELDOUT_PANELS)}"
        )

    panel = HELDOUT_PANELS[name]

    if purpose != "final_eval":
        _log_access(name, purpose, False, caller)
        raise HeldoutPanelAccessError(
            f"Refusing to load held-out panel {name!r} without purpose='final_eval' "
            f"(got purpose={purpose!r}). This panel is quarantined: {panel.note}"
        )

    _log_access(name, purpose, True, caller)
    df = paths.load_table(panel.path)
    meta = {
        "name": panel.name,
        "contaminated": panel.contaminated,
        "note": panel.note,
        "path": str(panel.path),
    }
    if panel.contaminated:
        logger.warning(
            "Loaded held-out panel %r for purpose=%r -- this panel is CONTAMINATED. "
            "Tag contaminated=True on every metric computed from it and do not "
            "report it as validation evidence. %s",
            name,
            purpose,
            panel.note,
        )
    return df, meta


CLR_MODEL = C_DEEP_BLUE


def plot_parity_bin(
    per_sample: pd.DataFrame, save_dir: str, prefix: str = "", threshold: float = 30.0
) -> str:
    plt, ticker, _ = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

    actual = per_sample["actual"].values
    pred = per_sample["pred"].values

    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    style_axis(ax, ticker)

    lo = max(min(actual.min(), pred.min()) * 0.7, 1e-2)
    hi = max(actual.max(), pred.max()) * 1.4

    ax.plot(
        [lo, hi],
        [lo, hi],
        color=C_MUTED,
        lw=3.6,
        ls="--",
        alpha=0.7,
        zorder=1,
        label="Perfect prediction",
    )

    ax.scatter(
        actual,
        pred,
        color=CLR_MODEL,
        s=52,
        alpha=0.85,
        edgecolors=C_WHITE,
        linewidths=0.8,
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xlabel(r"Actual Viscosity @ 1000 s$^{-1}$  (cP)", fontsize=21, labelpad=10, color=C_TEXT)
    ax.set_ylabel(
        r"Predicted Viscosity @ 1000 s$^{-1}$  (cP)", fontsize=21, labelpad=10, color=C_TEXT
    )
    ax.set_title(
        "Zero-shot predictions for PFDG38 set",
        fontsize=20,
        fontweight="bold",
        pad=14,
        color=C_TEXT,
        loc="left",
    )
    ax.legend(loc="upper left", fontsize=19, framealpha=0.95, borderpad=0.9, edgecolor=C_BORDER)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}zero_shot_parity_bins.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    return path


def plot_metrics_bars(summary: dict, save_dir: str, prefix: str = "") -> str:
    plt, ticker, _ = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

    fig, (axLin, axLog) = plt.subplots(1, 2, figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(C_WHITE)

    lin_metrics = [("mae", "MAE (cP)"), ("rmse", "RMSE (cP)"), ("mape", "MAPE (%)")]
    log_metrics = [("mae_log10", "MAE (log₁₀)"), ("rmse_log10", "RMSE (log₁₀)")]

    for ax, metrics, title in (
        (axLin, lin_metrics, "Linear-Space Metrics"),
        (axLog, log_metrics, "Log-Space Metrics"),
    ):
        style_axis(ax, ticker)
        x = np.arange(len(metrics))
        vals = [summary[k] for k, _ in metrics]
        ax.bar(x, vals, width=0.5, color=CLR_MODEL, edgecolor=C_WHITE, linewidth=0.6, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=10.5, color=CLR_MODEL)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=12, color=C_TEXT)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10, color=C_TEXT, loc="left")

    fig.suptitle(
        "Zero-Shot Benchmark · Standard Metric Set",
        fontsize=16,
        fontweight="bold",
        y=1.02,
        color=C_TEXT,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}zero_shot_metrics.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    return path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Standalone external zero-shot benchmark (no context) on unusual proteins."
    )
    ap.add_argument(
        "--data_csv",
        required=True,
        help=(
            "Zero-shot benchmark panel (proteins entirely absent from training) -- "
            "no default; this repo no longer ships a curated copy of it."
        ),
    )
    ap.add_argument(
        "--model_dir",
        default=None,
        help="Trained checkpoint directory to benchmark. Defaults to the most recently produced checkpoint.",
    )
    ap.add_argument(
        "--output_dir",
        default=None,
        help="Where to write results. Defaults to <model_dir>/zero_shot.",
    )
    ap.add_argument(
        "--bin_threshold",
        type=float,
        default=30.0,
        help="High/low bin cutoff on Viscosity_1000, in cP.",
    )
    return ap.parse_args(argv)


def _load_zero_shot_df(csv_path) -> pd.DataFrame:
    df = paths.load_table(csv_path)
    if "Whole_Charge" in df.columns and "Charge" not in df.columns:
        df = df.rename(columns={"Whole_Charge": "Charge"})
    df = prepare_df(df, drop_bad_rows=False)

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


def run(
    data_csv,
    model_dir=None,
    output_dir=None,
    bin_threshold=30.0,
):
    if model_dir is None:
        model_dir = paths.latest_checkpoint_dir(constants.CHECKPOINTS_DIR)
    if output_dir is None:
        output_dir = os.path.join(model_dir, "zero_shot")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading zero-shot benchmark data: {data_csv}")
    df = _load_zero_shot_df(data_csv)
    logger.info(
        f"  {len(df)} valid samples (Protein_type values: {sorted(df['Protein_type'].unique())})."
    )

    logger.info(f"Initializing model: {model_dir}")
    predictor = ViscosityPredictorCNP(model_dir)
    per_sample, summary = _run_benchmark(predictor, df, bin_threshold)

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

    per_sample_path = os.path.join(output_dir, "zero_shot_per_sample.csv")
    per_sample.to_csv(per_sample_path, index=False)
    logger.info(f"\nSaved per-sample CSV: {per_sample_path}")

    summary_path = os.path.join(output_dir, "zero_shot_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    logger.info(f"Saved summary CSV: {summary_path}")

    parity_path = plot_parity_bin(per_sample, output_dir, prefix="", threshold=bin_threshold)
    logger.info(f"Saved parity/bin plot: {parity_path}")

    metrics_path = plot_metrics_bars(summary, output_dir, prefix="")
    logger.info(f"Saved metrics plot: {metrics_path}")

    logger.info("Done.")
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.model_dir is None:
        args.model_dir = paths.latest_checkpoint_dir(constants.CHECKPOINTS_DIR)
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, "zero_shot")
    configure_logging(log_dir=args.output_dir)
    return run(**vars(args))


if __name__ == "__main__":
    main()
