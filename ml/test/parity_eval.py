"""
parity_plot.py
==============
Leave-one-protein-out parity plot for Viscosity_1000.

For each named protein type, the model learns on every OTHER protein
(plus the buffer-blank "none" rows), then predicts that held-out
protein. All predictions are collected and shown on a single
log-scale parity plot, coloured by protein type.

Edit the CONFIG block, then run:
    python parity_plot.py
"""

import copy
import logging
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")

try:
    from inference_o_net import ViscosityPredictorCNP
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inference_o_net import ViscosityPredictorCNP

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = r"models\experiments\o_net"
DATA_CSV = r"data/raw/formulation_data_02162026.csv"
OUTPUT_DIR = r"models\experiments\o_net\benchmarks"
LEARN_STEPS = 50
LEARN_LR = 1e-3
SEED = 42
TARGET_VISC = "Viscosity_1000"  # shear rate column to show on parity plot
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ParityPlot")

VISC_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]
PRED_TARGET = f"Pred_{TARGET_VISC}"

# ── Brand palette ─────────────────────────────────────────────────────────────
C_DEEP_BLUE = "#2596be"
C_BRIGHT_BLUE = "#13B5F0"
C_CYAN_MED = "#4EC4EB"
C_CYAN_PALE = "#8DD9F7"
C_GREEN = "#4caf50"
C_ORANGE = "#ff9800"
C_RED = "#f44336"
C_TEXT = "#24292f"
C_MUTED = "#6b7280"
C_BORDER = "#d1d5db"
C_BORDER_LT = "#e5e7eb"
C_BG_LIGHTEST = "#f9fafb"
C_WHITE = "#ffffff"

# Distinct colours for up to 12 protein types, drawn from brand palette +
# accessible categorical additions
PROTEIN_COLOURS = [
    "#2596be",
    "#4caf50",
    "#ff9800",
    "#f44336",
    "#9c27b0",
    "#13B5F0",
    "#795548",
    "#607d8b",
    "#e91e63",
    "#009688",
    "#ff5722",
    "#3f51b5",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def prepare_df(df: pd.DataFrame, drop_bad: bool = True) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        if col != "ID":
            df[col] = df[col].astype(float)
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)
    if drop_bad:
        mask = pd.Series(True, index=df.index)
        for vc in VISC_COLS:
            if vc in df.columns:
                mask &= df[vc].notna() & (df[vc] > 0)
        crit = [c for c in ["Protein_conc", "kP"] if c in df.columns]
        if crit:
            mask &= df[crit].notna().all(axis=1)
        df = df[mask].reset_index(drop=True)
    return df


def save_state(p):
    return {
        "state_dict": copy.deepcopy(p.model.state_dict()),
        "memory_vector": (
            p.memory_vector.clone() if p.memory_vector is not None else None
        ),
    }


def restore_state(p, snap):
    p.model.load_state_dict(snap["state_dict"])
    p.model.eval()
    p.memory_vector = (
        snap["memory_vector"].clone() if snap["memory_vector"] is not None else None
    )


def has_nan(p):
    return any(torch.isnan(param).any() for param in p.model.parameters())


def metrics(true, pred):
    """Return MAE, MAPE, log-RMSE, and R² on Viscosity_1000."""
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mae = float(np.mean(np.abs(t - p)))
    mape = float(np.mean(np.abs((t - p) / np.clip(t, 1e-6, None))) * 100)
    log_rmse = float(
        np.sqrt(
            np.mean(
                (np.log10(np.clip(t, 1e-6, None)) - np.log10(np.clip(p, 1e-6, None)))
                ** 2
            )
        )
    )
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    return {"mae": mae, "mape": mape, "log_rmse": log_rmse, "r2": r2}


# ──────────────────────────────────────────────────────────────────────────────
# Leave-one-protein-out loop
# ──────────────────────────────────────────────────────────────────────────────


def safe_log(msg: str):
    """Logger wrapper that strips non-ASCII so Windows CP1252 consoles don't crash."""
    try:
        logger.info(msg)
    except UnicodeEncodeError:
        logger.info(msg.encode("ascii", errors="replace").decode("ascii"))


def run_per_protein(predictor, df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-protein self-calibration evaluation.

    For each named protein type:
      1. Restore the base (zero-shot / pretrained) model state.
      2. Calibrate (learn) on ALL samples of that protein.
      3. Predict those same samples.
      4. Record true vs predicted Viscosity_1000.

    This tests how well the model adapts to and explains each protein's
    formulation-viscosity landscape once given its complete dataset.

    Returns a DataFrame with columns:
        Protein_type, ID, True_Visc1000, Pred_Visc1000
    """
    protein_types = sorted(
        [
            pt
            for pt in df["Protein_type"].unique()
            if str(pt).lower() not in ("none", "nan", "")
        ]
    )

    # Snapshot the base model state once — restored fresh for every protein
    base_snap = save_state(predictor)
    all_records = []

    logger.info(
        f"Running per-protein calibration over {len(protein_types)} protein types."
    )

    for pt in protein_types:
        safe_log(f"\n  Protein: {pt}")

        protein_df = df[df["Protein_type"] == pt].copy()
        logger.info(
            f"     Calibrating on {len(protein_df)} samples, then predicting the same {len(protein_df)} samples."
        )

        # Restore clean base weights, then calibrate on this protein's data
        restore_state(predictor, base_snap)
        predictor.learn(protein_df, steps=LEARN_STEPS, lr=LEARN_LR)

        if has_nan(predictor):
            logger.warning(f"     NaN weights after learning {pt} — skipping.")
            restore_state(predictor, base_snap)
            continue

        # Predict the same protein samples (tests calibration quality)
        try:
            results = predictor.predict(protein_df)
        except Exception as e:
            logger.warning(f"     Predict failed for {pt}: {e}")
            continue

        if PRED_TARGET not in results.columns:
            logger.warning(f"     {PRED_TARGET} not in results for {pt}.")
            continue

        for _, row in results.iterrows():
            true_val = protein_df.loc[
                protein_df["ID"] == str(row["ID"]), TARGET_VISC
            ].values
            if len(true_val) == 0:
                continue
            all_records.append(
                {
                    "Protein_type": pt,
                    "ID": str(row["ID"]),
                    "True_Visc1000": float(true_val[0]),
                    "Pred_Visc1000": float(row[PRED_TARGET]),
                }
            )

        # Per-protein summary
        sub_records = [r for r in all_records if r["Protein_type"] == pt]
        if sub_records:
            sub = pd.DataFrame(sub_records)
            m = metrics(sub["True_Visc1000"], sub["Pred_Visc1000"])
            logger.info(
                f"     MAE={m['mae']:.3f} cP  |  MAPE={m['mape']:.1f}%  |  "
                f"log-RMSE={m['log_rmse']:.4f}  |  R2={m['r2']:.4f}"
            )

    return pd.DataFrame(all_records)


# ──────────────────────────────────────────────────────────────────────────────
# Parity plot
# ──────────────────────────────────────────────────────────────────────────────


def plot_parity(results_df: pd.DataFrame, save_dir: str) -> str:
    protein_types = sorted(results_df["Protein_type"].unique())
    colour_map = {
        pt: PROTEIN_COLOURS[i % len(PROTEIN_COLOURS)]
        for i, pt in enumerate(protein_types)
    }

    true_all = results_df["True_Visc1000"].values
    pred_all = results_df["Pred_Visc1000"].values
    overall = metrics(true_all, pred_all)

    # ── Style ────────────────────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "figure.facecolor": C_WHITE,
            "axes.facecolor": C_BG_LIGHTEST,
            "axes.edgecolor": C_BORDER,
            "axes.labelcolor": C_TEXT,
            "axes.titlecolor": C_TEXT,
            "xtick.color": C_MUTED,
            "ytick.color": C_MUTED,
            "xtick.labelcolor": C_MUTED,
            "ytick.labelcolor": C_MUTED,
            "grid.color": C_BORDER_LT,
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    ax.set_facecolor(C_BG_LIGHTEST)

    # ── Data range for axis limits — tight to actual data with small padding ──
    all_vals = np.concatenate([true_all, pred_all])
    all_vals = all_vals[all_vals > 0]
    log_min = np.log10(all_vals.min())
    log_max = np.log10(all_vals.max())
    log_pad = (log_max - log_min) * 0.04  # 4% padding in log space
    lo = 10 ** (log_min - log_pad)
    hi = 10 ** (log_max + log_pad)

    # ── Perfect parity line ───────────────────────────────────────────────────
    parity_x = np.logspace(np.log10(lo), np.log10(hi), 300)
    ax.plot(
        parity_x,
        parity_x,
        color=C_DEEP_BLUE,
        lw=1.6,
        ls="--",
        zorder=2,
        label="Perfect parity",
    )

    # ── ±20% and ±50% error bands ─────────────────────────────────────────────
    for factor, alpha, label in [(1.50, 0.07, "±50%"), (1.20, 0.12, "±20%")]:
        ax.fill_between(
            parity_x,
            parity_x / factor,
            parity_x * factor,
            color=C_CYAN_PALE,
            alpha=alpha,
            zorder=1,
        )
    # Band edge lines
    for factor, ls in [(1.20, (0, (4, 3))), (1.50, (0, (2, 4)))]:
        ax.plot(
            parity_x,
            parity_x * factor,
            color=C_CYAN_MED,
            lw=0.8,
            ls=ls,
            zorder=2,
            alpha=0.7,
        )
        ax.plot(
            parity_x,
            parity_x / factor,
            color=C_CYAN_MED,
            lw=0.8,
            ls=ls,
            zorder=2,
            alpha=0.7,
        )
    # Band labels (right edge)
    for factor, txt in [(1.20, "±20%"), (1.50, "±50%")]:
        y_pos = hi / factor * 0.92
        ax.text(
            hi * 0.78,
            y_pos,
            txt,
            fontsize=7.5,
            color=C_CYAN_MED,
            va="center",
            alpha=0.9,
        )

    # ── Scatter per protein type ──────────────────────────────────────────────
    for pt in protein_types:
        sub = results_df[results_df["Protein_type"] == pt]
        clr = colour_map[pt]
        ax.scatter(
            sub["True_Visc1000"],
            sub["Pred_Visc1000"],
            color=clr,
            s=52,
            zorder=5,
            alpha=0.85,
            edgecolors=C_WHITE,
            linewidths=0.7,
        )

    # ── Axes: log scale ───────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:g}" if v >= 1 else f"{v:.2f}")
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:g}" if v >= 1 else f"{v:.2f}")
    )
    ax.tick_params(which="both", length=4, width=0.8)
    ax.grid(True, which="major", zorder=0)
    ax.grid(True, which="minor", zorder=0, alpha=0.4, linewidth=0.4)

    for spine in ax.spines.values():
        spine.set_edgecolor(C_BORDER)
        spine.set_linewidth(0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax.set_xlabel(
        "Measured  Viscosity @ 1000 s⁻¹  (cP)", fontsize=12, labelpad=10, color=C_TEXT
    )
    ax.set_ylabel(
        "Predicted  Viscosity @ 1000 s⁻¹  (cP)", fontsize=12, labelpad=10, color=C_TEXT
    )

    # ── Metrics box (top-left) ────────────────────────────────────────────────
    metrics_text = (
        f"Overall  (N={len(results_df)})\n"
        f"MAE        {overall['mae']:.2f} cP\n"
        f"MAPE       {overall['mape']:.1f}%\n"
        f"log-RMSE   {overall['log_rmse']:.4f}\n"
        f"R²         {overall['r2']:.4f}"
    )
    ax.text(
        0.03,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        color=C_TEXT,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=C_WHITE,
            edgecolor=C_BORDER,
            linewidth=0.9,
            alpha=0.92,
        ),
        family="monospace",
    )

    # ── Legend — parity reference lines only, no protein names ──────────────
    parity_handle = Line2D(
        [0], [0], color=C_DEEP_BLUE, lw=1.6, ls="--", label="Perfect parity"
    )
    band_handle = Line2D(
        [0],
        [0],
        color=C_CYAN_MED,
        lw=1.2,
        ls=(0, (4, 3)),
        label="\u00b120% / \u00b150% bands",
    )
    ax.legend(
        handles=[parity_handle, band_handle],
        labels=["Perfect parity", "\u00b120% / \u00b150% bands"],
        loc="lower right",
        fontsize=9,
        framealpha=0.95,
        edgecolor=C_BORDER,
        borderpad=0.8,
        handlelength=1.8,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "Viscosity @ 1000 s\u207b\u00b9 \u2014 Calibration Parity",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color=C_TEXT,
        loc="left",
    )
    fig.text(
        0.0,
        -0.02,
        "Each protein type: model calibrated on all its samples, then predicted on those same samples.",
        ha="left",
        fontsize=8,
        color=C_MUTED,
        style="italic",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    out_path = os.path.join(save_dir, "parity_viscosity_1000_per_protein.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    logger.info(f"Parity plot saved: {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load model
    logger.info(f"Loading model from: {MODEL_DIR}")
    predictor = ViscosityPredictorCNP(MODEL_DIR)
    logger.info("Model loaded.")

    # 2. Load & clean data
    logger.info(f"Loading data: {DATA_CSV}")
    df = prepare_df(pd.read_csv(DATA_CSV), drop_bad=True)
    logger.info(
        f"  {len(df)} valid rows across "
        f"{df['Protein_type'].nunique()} protein types."
    )

    # 3. Per-protein calibration + prediction
    results_df = run_per_protein(predictor, df)

    if results_df.empty:
        logger.error("No predictions were produced — check model paths and data.")
        return

    # 4. Save predictions table
    pred_path = os.path.join(OUTPUT_DIR, "per_protein_predictions_visc1000.csv")
    results_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved: {pred_path}")

    # 5. Overall metrics summary
    m = metrics(results_df["True_Visc1000"], results_df["Pred_Visc1000"])
    logger.info(
        f"\n{'='*55}\n"
        f"  OVERALL  (N={len(results_df)} samples across "
        f"{results_df['Protein_type'].nunique()} proteins)\n"
        f"  MAE        : {m['mae']:.3f} cP\n"
        f"  MAPE       : {m['mape']:.1f}%\n"
        f"  log-RMSE   : {m['log_rmse']:.5f}\n"
        f"  R2         : {m['r2']:.5f}\n"
        f"{'='*55}"
    )

    # Per-protein breakdown
    logger.info("\n  Per-protein breakdown:")
    for pt, grp in results_df.groupby("Protein_type"):
        pm = metrics(grp["True_Visc1000"], grp["Pred_Visc1000"])
        logger.info(
            f"    {pt:<18}  n={len(grp):>3}  "
            f"MAE={pm['mae']:6.2f} cP  MAPE={pm['mape']:5.1f}%  "
            f"R2={pm['r2']:.3f}"
        )

    # 6. Parity plot
    logger.info("\nGenerating parity plot...")
    plot_path = plot_parity(results_df, OUTPUT_DIR)
    logger.info(f"\nDone.  Outputs:\n  {pred_path}\n  {plot_path}")


if __name__ == "__main__":
    main()
