"""
convergence_plot_ibal.py
========================
Replays the optimal ibal sample addition order discovered by
order_discovery_ibal.py, recording MAE and log-RMSE at every step,
then produces publication-quality convergence plots.

Usage:
    python convergence_plot_ibal.py

Edit the CONFIG block below to match your paths.
"""

import copy
import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings("ignore")

try:
    from inference_o_net import ViscosityPredictorCNP
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from inference_o_net import ViscosityPredictorCNP

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit these paths
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = r"models\experiments\o_net_no_ibal"
IBAL_CSV = r"data/processed/ibal_eval.csv"
ORDER_CSV = r"models\experiments\o_net_no_ibal\benchmarks\optimal_order_summary.csv"
PRETRAIN_CSV = r"data/processed/formulation_data_no_ibal.csv"
OUTPUT_DIR = r"models\experiments\o_net_no_ibal\benchmarks"
LEARN_STEPS = 50
LEARN_LR = 1e-3
SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ConvergencePlot")

VISC_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]
PRED_COLS = [f"Pred_{c}" for c in VISC_COLS]


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────


def prepare_df(df: pd.DataFrame, drop_bad_rows: bool = False) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        if col != "ID":
            df[col] = df[col].astype(float)
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)
    if drop_bad_rows:
        visc_mask = pd.Series(True, index=df.index)
        for vc in VISC_COLS:
            if vc in df.columns:
                visc_mask &= df[vc].notna() & (df[vc] > 0)
        crit = [c for c in ["MW", "Protein_conc", "kP"] if c in df.columns]
        num_mask = (
            df[crit].notna().all(axis=1) if crit else pd.Series(True, index=df.index)
        )
        df = df[visc_mask & num_mask].reset_index(drop=True)
    return df


def save_state(predictor):
    return {
        "state_dict": copy.deepcopy(predictor.model.state_dict()),
        "memory_vector": (
            predictor.memory_vector.clone()
            if predictor.memory_vector is not None
            else None
        ),
    }


def restore_state(predictor, snap):
    predictor.model.load_state_dict(snap["state_dict"])
    predictor.model.eval()
    predictor.memory_vector = (
        snap["memory_vector"].clone() if snap["memory_vector"] is not None else None
    )


def has_nan_weights(predictor):
    return any(torch.isnan(p).any() for p in predictor.model.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(results_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    pred_all, true_all = [], []
    for pc, vc in zip(PRED_COLS, VISC_COLS):
        if pc in results_df.columns and vc in truth_df.columns:
            pred_all.append(results_df[pc].values)
            true_all.append(truth_df[vc].values)
    if not pred_all:
        return {"mae": np.nan, "log_rmse": np.nan}

    pred = np.concatenate(pred_all)
    true = np.concatenate(true_all)

    # MAE in linear cP
    mae = float(np.mean(np.abs(true - pred)))

    # log-RMSE (scale-invariant)
    log_pred = np.log10(np.clip(pred, 1e-6, None))
    log_true = np.log10(np.clip(true, 1e-6, None))
    log_rmse = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))

    return {"mae": mae, "log_rmse": log_rmse}


# ──────────────────────────────────────────────────────────────────────────────
# Core replay loop
# ──────────────────────────────────────────────────────────────────────────────


def run_convergence_replay(
    predictor,
    ibal_df: pd.DataFrame,
    ordered_ids: list[str],
    learn_steps: int = 50,
    learn_lr: float = 1e-3,
) -> pd.DataFrame:
    """
    Adds samples one-by-one in `ordered_ids` order.
    At each step: learn on context, predict on holdout, record metrics.
    Returns a DataFrame with columns: step, sample_id, n_context, mae, log_rmse.
    """
    id_to_idx = {str(row["ID"]): idx for idx, row in ibal_df.iterrows()}
    ordered_ids = [sid for sid in ordered_ids if sid in id_to_idx]

    baseline_snap = save_state(predictor)
    records = []

    logger.info(f"Replaying {len(ordered_ids)} samples in optimal order...")

    for step, sample_id in enumerate(ordered_ids, start=1):
        context_ids = ordered_ids[:step]
        holdout_ids = ordered_ids[step:]

        context_idx = [id_to_idx[s] for s in context_ids]
        context_df = ibal_df.loc[context_idx].copy()

        restore_state(predictor, baseline_snap)
        predictor.learn(context_df, steps=learn_steps, lr=learn_lr)

        metrics = {"mae": np.nan, "log_rmse": np.nan}

        if holdout_ids and not has_nan_weights(predictor):
            holdout_idx = [id_to_idx[s] for s in holdout_ids]
            holdout_df = ibal_df.loc[holdout_idx].copy()
            try:
                results_df = predictor.predict(holdout_df)
                metrics = compute_metrics(results_df, holdout_df)
            except Exception as e:
                logger.warning(f"  Step {step}: predict failed — {e}")

        # Advance baseline if weights are clean
        if not has_nan_weights(predictor):
            baseline_snap = save_state(predictor)
        else:
            logger.warning(f"  Step {step}: NaN weights, keeping previous baseline.")
            restore_state(predictor, baseline_snap)

        records.append(
            {
                "step": step,
                "sample_id": sample_id,
                "n_context": step,
                "mae": metrics["mae"],
                "log_rmse": metrics["log_rmse"],
            }
        )

        logger.info(
            f"  [{step:>2}/{len(ordered_ids)}] Added {sample_id:>6} | "
            f"Holdout MAE={metrics['mae']:.3f} cP  "
            f"log-RMSE={metrics['log_rmse']:.5f}"
        )

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

# How many context-sample steps to show in the plots
PLOT_MAX_STEPS = 10

# VisQ brand palette
C_DEEP_BLUE = "#2596be"  # MAE line / primary axis label
C_BRIGHT_BLUE = "#13B5F0"  # log-RMSE line / secondary axis label
C_CYAN_MED = "#4EC4EB"  # tick marks, spine highlights
C_CYAN_PALE = "#8DD9F7"  # fill / shading under curves
C_GREEN = "#4caf50"  # best-point marker
C_ORANGE = "#ff9800"  # convergence / plateau annotation
C_TEXT = "#24292f"  # axis labels, titles
C_MUTED = "#6b7280"  # subtitles, captions
C_BORDER = "#d1d5db"  # grid lines, spines
C_BORDER_LT = "#e5e7eb"  # inner grid
C_BG_LIGHT = "#f3f4f6"  # panel fill
C_BG_LIGHTEST = "#f9fafb"  # figure fill
C_WHITE = "#ffffff"  # page background

FONT_MAIN = "DejaVu Sans"


def apply_base_style():
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
            "grid.alpha": 1.0,
            "text.color": C_TEXT,
            "legend.facecolor": C_WHITE,
            "legend.edgecolor": C_BORDER,
            "legend.labelcolor": C_TEXT,
            "font.family": FONT_MAIN,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def find_convergence_step(
    values: np.ndarray, window: int = 3, threshold: float = 0.005
) -> int | None:
    """Return 0-based index of first step where the error plateau begins."""
    for i in range(window, len(values)):
        if np.all(np.abs(np.diff(values[i - window : i])) < threshold):
            return i - window
    return None


def _style_axis(ax, spine_color=C_BORDER):
    """Apply consistent spine / grid styling to an axis."""
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(0.9)
    ax.tick_params(length=3, width=0.8)
    ax.grid(True, which="major", axis="both", zorder=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_facecolor(C_BG_LIGHTEST)


def _annotate_best(ax, x_arr, y_arr, fmt, color, offset_frac=0.06):
    """Mark and label the minimum value in a data series."""
    b = np.argmin(y_arr)
    ax.scatter(
        [x_arr[b]],
        [y_arr[b]],
        color=C_GREEN,
        s=100,
        zorder=7,
        edgecolors=C_WHITE,
        linewidths=1.4,
    )
    rng = y_arr.max() - y_arr.min() or 1e-6
    ax.annotate(
        fmt.format(y_arr[b]),
        xy=(x_arr[b], y_arr[b]),
        xytext=(x_arr[b] + 0.25, y_arr[b] - rng * offset_frac),
        fontsize=8.5,
        color=C_GREEN,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=C_GREEN, lw=0.6),
    )


def _annotate_convergence(ax, x_arr, y_arr, c_idx, color=C_ORANGE):
    """Draw a vertical marker and label where the plateau begins."""
    if c_idx is None or c_idx >= len(x_arr):
        return
    cx = x_arr[c_idx]
    ax.axvline(cx, color=color, lw=1.1, ls="--", alpha=0.7, zorder=3)
    ax.text(
        cx + 0.15,
        y_arr.max() - (y_arr.max() - y_arr.min()) * 0.04,
        f"plateau  n={cx}",
        fontsize=7.5,
        color=color,
        va="top",
        style="italic",
    )


def plot_convergence(df: pd.DataFrame, save_dir: str):
    """Produce two figures trimmed to PLOT_MAX_STEPS:
       1. Combined dual-axis (MAE left, log-RMSE right).
       2. Side-by-side individual panels.
    Both show only the first PLOT_MAX_STEPS context steps, where generalisation
    on the remaining unseen samples is most diagnostic.
    """
    apply_base_style()
    os.makedirs(save_dir, exist_ok=True)

    # ── Filter to valid rows then trim to first PLOT_MAX_STEPS ───────────────
    valid_mask = ~(df["mae"].isna() | df["log_rmse"].isna())
    valid_mask.iloc[-1] = False  # last step has no holdout
    plot_df = df[valid_mask].head(PLOT_MAX_STEPS)

    sx = plot_df["n_context"].values
    smae = plot_df["mae"].values
    slrmse = plot_df["log_rmse"].values
    labels = plot_df["sample_id"].values  # sample IDs for x-tick labels

    n_remaining_at_start = int(df["n_context"].max()) - 1  # total unseen at step 1

    conv_mae = find_convergence_step(smae)
    conv_lrmse = find_convergence_step(slrmse)

    # ── Figure 1: Combined dual-axis ─────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    _style_axis(ax1)

    # — MAE (left, deep blue) —
    ax1.plot(sx, smae, color=C_DEEP_BLUE, lw=2.4, zorder=4, solid_capstyle="round")
    ax1.scatter(
        sx, smae, color=C_DEEP_BLUE, s=38, zorder=5, edgecolors=C_WHITE, linewidths=1.1
    )
    ax1.fill_between(sx, smae, alpha=0.10, color=C_CYAN_PALE, zorder=1)
    ax1.set_xlabel(
        "Samples added to context  (n)", fontsize=11, labelpad=8, color=C_TEXT
    )
    ax1.set_ylabel("MAE  (cP)", fontsize=11, labelpad=8, color=C_DEEP_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_DEEP_BLUE, colors=C_DEEP_BLUE)
    ax1.spines["left"].set_edgecolor(C_DEEP_BLUE)
    ax1.spines["left"].set_linewidth(1.2)
    ax1.set_xlim(sx[0] - 0.4, sx[-1] + 0.6)

    # — log-RMSE (right, bright blue) —
    ax2 = ax1.twinx()
    ax2.set_facecolor(C_BG_LIGHTEST)
    ax2.plot(
        sx,
        slrmse,
        color=C_BRIGHT_BLUE,
        lw=2.4,
        zorder=4,
        solid_capstyle="round",
        ls=(0, (6, 2)),
    )
    ax2.scatter(
        sx,
        slrmse,
        color=C_BRIGHT_BLUE,
        s=38,
        zorder=5,
        edgecolors=C_WHITE,
        linewidths=1.1,
        marker="D",
    )
    ax2.fill_between(sx, slrmse, alpha=0.06, color=C_BRIGHT_BLUE, zorder=1)
    ax2.set_ylabel("log-RMSE  (log₁₀ cP)", fontsize=11, labelpad=8, color=C_BRIGHT_BLUE)
    ax2.tick_params(axis="y", labelcolor=C_BRIGHT_BLUE, colors=C_BRIGHT_BLUE)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor(C_BRIGHT_BLUE)
    ax2.spines["right"].set_linewidth(1.2)
    ax2.spines["top"].set_visible(False)

    # — Best-point markers —
    _annotate_best(ax1, sx, smae, "  {:.3f} cP", C_DEEP_BLUE)
    _annotate_best(ax2, sx, slrmse, "  {:.4f}", C_BRIGHT_BLUE, offset_frac=0.04)

    # — Convergence lines —
    _annotate_convergence(ax1, sx, smae, conv_mae)
    _annotate_convergence(ax2, sx, slrmse, conv_lrmse)

    # — Integer x-ticks with sample ID sub-labels —
    ax1.set_xticks(sx)
    ax1.set_xticklabels(
        [f"{n}" for n, sid in zip(sx, labels)],
        fontsize=8.5,
        color=C_MUTED,
    )

    # — Top info strip —
    holdout_start = int(df["n_context"].max()) - len(plot_df)
    fig.text(
        0.5,
        0.97,
        f"Evaluating generalisation to {n_remaining_at_start}→"
        f"{n_remaining_at_start - PLOT_MAX_STEPS + 1} unseen samples",
        ha="center",
        fontsize=8.5,
        color=C_MUTED,
        style="italic",
    )

    # — Legend —
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=C_DEEP_BLUE,
            lw=2.4,
            marker="o",
            markersize=6,
            markerfacecolor=C_DEEP_BLUE,
            markeredgecolor=C_WHITE,
            label="MAE (cP)",
        ),
        Line2D(
            [0],
            [0],
            color=C_BRIGHT_BLUE,
            lw=2.4,
            marker="D",
            markersize=5,
            markerfacecolor=C_BRIGHT_BLUE,
            markeredgecolor=C_WHITE,
            ls=(0, (6, 2)),
            label="log-RMSE (log₁₀ cP)",
        ),
        Line2D(
            [0],
            [0],
            color=C_GREEN,
            lw=0,
            marker="o",
            markersize=8,
            markerfacecolor=C_GREEN,
            markeredgecolor=C_WHITE,
            label="Best value",
        ),
        Line2D([0], [0], color=C_ORANGE, lw=1.2, ls="--", label="Plateau onset"),
    ]
    ax1.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
        borderpad=0.9,
        edgecolor=C_BORDER,
    )

    ax1.set_title(
        "Ibalizumab · ICL Convergence",
        fontsize=13,
        fontweight="bold",
        pad=12,
        color=C_TEXT,
        loc="left",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = os.path.join(save_dir, "convergence_combined.png")
    fig.savefig(combined_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    logger.info(f"Saved: {combined_path}")

    # ── Figure 2: Side-by-side panels ────────────────────────────────────────
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    fig.patch.set_facecolor(C_WHITE)

    panel_cfg = [
        # (ax,  vals,    metric_label,  unit,         line_clr,       c_idx)
        (axA, smae, "MAE", "cP", C_DEEP_BLUE, conv_mae),
        (axB, slrmse, "log-RMSE", "log₁₀ cP", C_BRIGHT_BLUE, conv_lrmse),
    ]

    for ax, vals, mlabel, unit, clr, c_idx in panel_cfg:
        _style_axis(ax)

        # Dashed reference at minimum
        ax.axhline(np.nanmin(vals), color=clr, lw=0.8, ls="--", alpha=0.35, zorder=2)

        # Fill + line
        ax.fill_between(sx, vals, alpha=0.11, color=C_CYAN_PALE, zorder=1)
        ax.plot(sx, vals, color=clr, lw=2.5, zorder=4, solid_capstyle="round")
        ax.scatter(
            sx, vals, color=clr, s=42, zorder=5, edgecolors=C_WHITE, linewidths=1.2
        )

        # Best point
        fmt = "  {:.3f} cP" if mlabel == "MAE" else "  {:.4f}"
        _annotate_best(ax, sx, vals, fmt, clr)

        # Convergence
        _annotate_convergence(ax, sx, vals, c_idx)

        # Axes labels
        ax.set_xlabel(
            "Samples added to context  (n)", fontsize=10.5, labelpad=7, color=C_TEXT
        )
        ax.set_ylabel(f"{mlabel}  ({unit})", fontsize=10.5, labelpad=7, color=clr)
        ax.tick_params(axis="y", labelcolor=clr)
        ax.spines["left"].set_edgecolor(clr)
        ax.spines["left"].set_linewidth(1.2)
        ax.set_title(
            f"{mlabel} vs. Context Size",
            fontsize=12,
            fontweight="bold",
            pad=9,
            color=C_TEXT,
            loc="left",
        )

        # x-ticks: integer + sample ID
        ax.set_xticks(sx)
        ax.set_xticklabels(
            [f"{n}" for n, sid in zip(sx, labels)],
            fontsize=8,
            color=C_MUTED,
        )
        ax.set_xlim(sx[0] - 0.4, sx[-1] + 0.6)

        # Thin top border accent in brand colour
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_edgecolor(clr)
        ax.spines["top"].set_linewidth(2.5)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Ibalizumab · ICL Convergence Curves",
        fontsize=13,
        fontweight="bold",
        y=1.02,
        color=C_TEXT,
        x=0.0,
        ha="left",
        transform=fig.transFigure,
    )
    fig.text(
        0.0,
        -0.04,
        f"Model learned on n samples, evaluated on the remaining "
        f"{n_remaining_at_start}→{n_remaining_at_start - PLOT_MAX_STEPS + 1} "
        "unseen Ibalizumab formulations",
        ha="left",
        fontsize=8,
        color=C_MUTED,
        style="italic",
    )

    plt.tight_layout()
    panels_path = os.path.join(save_dir, "convergence_panels.png")
    fig.savefig(panels_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    logger.info(f"Saved: {panels_path}")

    return combined_path, panels_path


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

    # 2. Optional pretrain on non-ibal data
    if PRETRAIN_CSV and os.path.exists(PRETRAIN_CSV):
        logger.info(f"Pre-training on: {PRETRAIN_CSV}")
        pretrain_df = prepare_df(pd.read_csv(PRETRAIN_CSV), drop_bad_rows=True)
        predictor.learn(pretrain_df, steps=LEARN_STEPS, lr=LEARN_LR)
        if has_nan_weights(predictor):
            logger.error("NaN weights after pretrain — reloading clean model.")
            predictor = ViscosityPredictorCNP(MODEL_DIR)
        else:
            logger.info(f"  Pre-trained on {len(pretrain_df)} samples.")

    # 3. Load ibal data
    logger.info(f"Loading ibal data: {IBAL_CSV}")
    ibal_df = prepare_df(pd.read_csv(IBAL_CSV), drop_bad_rows=True)
    logger.info(f"  {len(ibal_df)} valid ibal samples.")

    # 4. Load optimal order
    logger.info(f"Loading optimal order: {ORDER_CSV}")
    order_df = pd.read_csv(ORDER_CSV)
    ordered_ids = order_df["Sample_ID"].tolist()
    logger.info(f"  Order: {ordered_ids}")

    # 5. Replay
    results_df = run_convergence_replay(
        predictor,
        ibal_df,
        ordered_ids,
        learn_steps=LEARN_STEPS,
        learn_lr=LEARN_LR,
    )

    # 6. Save metrics table
    metrics_path = os.path.join(OUTPUT_DIR, "convergence_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved: {metrics_path}")

    # 7. Print summary
    valid = results_df.dropna(subset=["mae", "log_rmse"])
    best_lrmse = valid.loc[valid["log_rmse"].idxmin()]
    best_mae = valid.loc[valid["mae"].idxmin()]
    logger.info(
        f"\n{'='*55}\n"
        f"  Best log-RMSE : {best_lrmse['log_rmse']:.5f}  @ n={best_lrmse['n_context']}  ({best_lrmse['sample_id']} added)\n"
        f"  Best MAE      : {best_mae['mae']:.3f} cP  @ n={best_mae['n_context']}  ({best_mae['sample_id']} added)\n"
        f"{'='*55}"
    )

    # 8. Plot
    logger.info("Generating plots...")
    combined, panels = plot_convergence(results_df, OUTPUT_DIR)
    logger.info(f"\nAll done. Outputs in: {OUTPUT_DIR}")
    logger.info(f"  {combined}")
    logger.info(f"  {panels}")
    logger.info(f"  {metrics_path}")


if __name__ == "__main__":
    main()
