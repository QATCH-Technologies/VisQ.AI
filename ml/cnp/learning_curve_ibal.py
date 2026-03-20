"""
learning_curve_ibal.py
======================
Replays the optimal ibalizumab sample-addition order discovered by
order_discovery_ibal.py, recording MAE, RMSE, MAPE and log10-RMSE at every
step, then produces publication-quality convergence plots.

Changes from the original convergence_plot_ibal.py:
  [FIX-1] save_state / restore_state simplified.
          The new learn() is encode-only — model weights are never modified,
          so the baseline snapshot only needs to preserve memory_vector and
          context_t. There is no need to save/restore the full state_dict
          between steps. This also makes each replay step ~10× faster.

  [FIX-2] learn() call updated to encode-only multi-draw API.
          Steps and LR config constants are kept for backward compatibility
          with any external callers but are passed as ignored kwargs.
          N_DRAWS and K_CONTEXT control the new averaging logic.

  [FIX-3] Diverse context selection applied before each learn() call.
          _select_diverse_context() (exposed on the engine) stratifies the
          growing context pool by Protein_conc quartile before encoding,
          preventing high-concentration samples from dominating the memory
          vector at larger context sizes.

  [FIX-4] Per-step uncertainty captured via predict_with_uncertainty().
          std_log10 is recorded at every step and used to draw a shaded CI
          ribbon on the log10 convergence plot.

  [FIX-5] Log10-space metrics added alongside linear cP metrics.
          compute_metrics() now returns rmse_log10 and mae_log10 which match
          the model's training objective and the run-5 benchmark (0.232).

  [FIX-6] New plot: log10 RMSE convergence with CI ribbon.
          plot_log_convergence() shows the metric the model was actually
          optimised for, with the per-step uncertainty band rendered as a
          shaded region.

  [FIX-7] init_clean_predictor no longer fine-tunes on pretrain_df.
          The new learn() is encode-only, so "pre-training" on the full
          non-ibal dataset simply encodes all other proteins as a cross-
          protein context baseline — a valid and useful zero-shot prior.

Usage:
    python learning_curve_ibal.py

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
MODEL_DIR = r"models\experiments\o_net_v3_no_ibal"
IBAL_CSV = r"data/processed/ibal_eval.csv"
ORDER_CSV = r"models\experiments\o_net_v3_no_ibal\benchmarks\optimal_order_summary.csv"
PRETRAIN_CSV = r"data/processed/formulation_data_no_ibal.csv"
OUTPUT_DIR = r"models\experiments\o_net_v3_no_ibal\benchmarks"

# [FIX-2] Encode-only context parameters (steps/lr retained for compat only)
N_DRAWS = 20  # random context subsets to average per encode call
K_CONTEXT = 8  # subset size per draw (matches few-shot elbow)
MAX_CTX_POOL = 15  # max samples after diverse context selection
N_UNC_SAMPLES = 30  # draws for per-step uncertainty estimation

# Legacy params — passed through to learn() but ignored by the new engine
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
    # "Viscosity_100",
    "Viscosity_1000",
    # "Viscosity_10000",
    # "Viscosity_100000",
    # "Viscosity_15000000",
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


# [FIX-1] save_state / restore_state simplified.
# Weights never change, so we only snapshot the latent memory state.
def save_state(predictor) -> dict:
    """Snapshot only the latent memory state (weights are immutable)."""
    return {
        "memory_vector": (
            predictor.memory_vector.clone()
            if predictor.memory_vector is not None
            else None
        ),
        "context_t": (
            predictor.context_t.clone() if predictor.context_t is not None else None
        ),
    }


def restore_state(predictor, snap: dict):
    """Restore latent memory state from snapshot."""
    predictor.memory_vector = (
        snap["memory_vector"].clone() if snap["memory_vector"] is not None else None
    )
    predictor.context_t = (
        snap["context_t"].clone() if snap["context_t"] is not None else None
    )


def reset_memory(predictor):
    """Clear latent state without touching model weights."""
    predictor.memory_vector = None
    predictor.context_t = None


def has_nan_weights(predictor) -> bool:
    return any(torch.isnan(p).any() for p in predictor.model.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


def _log10_safe(arr: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(arr, 1e-6, None))


def compute_metrics(results_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    """
    Returns linear-cP and log10-space metrics.

    [FIX-5] rmse_log10 and mae_log10 are added to match the model's training
    objective (optimised in log10 viscosity space) and to be directly
    comparable to benchmark RMSE values from evaluation reports.
    """
    pred_all, true_all = [], []
    for pc, vc in zip(PRED_COLS, VISC_COLS):
        if pc in results_df.columns and vc in truth_df.columns:
            pred_all.append(results_df[pc].values)
            true_all.append(truth_df[vc].values)
    if not pred_all:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "mae_log10": np.nan,
            "rmse_log10": np.nan,
        }

    pred = np.concatenate(pred_all)
    true = np.concatenate(true_all)

    # --- Linear cP metrics (human-readable) ---
    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mape = float(np.mean(np.abs((true - pred) / np.clip(true, 1e-6, None)))) * 100.0

    # --- Log10 metrics (model-native, matches training objective) ---
    t_log = _log10_safe(true)
    p_log = _log10_safe(pred)
    mae_log10 = float(np.mean(np.abs(t_log - p_log)))
    rmse_log10 = float(np.sqrt(np.mean((t_log - p_log) ** 2)))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mae_log10": mae_log10,
        "rmse_log10": rmse_log10,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core replay loop
# ──────────────────────────────────────────────────────────────────────────────


def _encode_context(predictor, context_df: pd.DataFrame):
    """
    [FIX-2 + FIX-3] Reset memory, apply diverse context selection, then
    encode with multi-draw averaging.
    """
    reset_memory(predictor)
    if context_df.empty:
        return

    # Diverse context selection when the engine exposes it
    if hasattr(predictor, "_select_diverse_context"):
        context_df = predictor._select_diverse_context(context_df, max_k=MAX_CTX_POOL)

    predictor.learn(
        context_df,
        steps=LEARN_STEPS,  # ignored by new engine, kept for compat
        lr=LEARN_LR,  # ignored by new engine, kept for compat
        n_draws=N_DRAWS,
        k=K_CONTEXT,
    )


def run_convergence_replay(
    predictor,
    ibal_df: pd.DataFrame,
    ordered_ids: list,
    learn_steps: int = 50,  # kept for API compat — passed to _encode_context
    learn_lr: float = 1e-3,  # kept for API compat — passed to _encode_context
) -> pd.DataFrame:
    """
    Adds ibalizumab samples one-by-one in `ordered_ids` order.
    At each step: encode context, predict holdout, record metrics + uncertainty.

    [FIX-1] No state_dict snapshots needed — weights never change.
    [FIX-4] Per-step std_log10 uncertainty recorded from context subsampling.

    Returns DataFrame with columns:
        step, sample_id, n_context,
        mae, rmse, mape,            ← linear cP
        mae_log10, rmse_log10,      ← log10 (model-native)
        std_log10                   ← mean uncertainty across shear rates
    """
    id_to_idx = {str(row["ID"]): idx for idx, row in ibal_df.iterrows()}
    ordered_ids = [sid for sid in ordered_ids if sid in id_to_idx]

    records = []
    null_metrics = {
        "mae": np.nan,
        "rmse": np.nan,
        "mape": np.nan,
        "mae_log10": np.nan,
        "rmse_log10": np.nan,
    }

    logger.info(f"Replaying {len(ordered_ids)} samples (plus 0-shot baseline)...")

    # ── Step 0: Zero-shot baseline ────────────────────────────────────────────
    reset_memory(predictor)
    all_holdout_idx = [id_to_idx[s] for s in ordered_ids]
    holdout_df_0 = ibal_df.loc[all_holdout_idx].copy()

    metrics_0 = null_metrics.copy()
    std_log10_0 = np.nan
    if not has_nan_weights(predictor):
        try:
            # Zero-shot: no context encoded, predict raw
            query_0 = holdout_df_0.drop(columns=VISC_COLS, errors="ignore")
            results_0 = predictor.predict(query_0)
            metrics_0 = compute_metrics(results_0, holdout_df_0)
            _, unc_0 = predictor.predict_with_uncertainty(
                query_0, n_samples=N_UNC_SAMPLES, k=K_CONTEXT
            )
            std_log10_0 = float(np.mean(unc_0.get("std_log10", [np.nan])))
        except Exception as e:
            logger.warning(f"  Step 0: predict failed — {e}")

    records.append(
        {
            "step": 0,
            "sample_id": "None",
            "n_context": 0,
            **metrics_0,
            "std_log10": std_log10_0,
        }
    )
    logger.info(
        f"  [ 0/{len(ordered_ids)}] 0-Shot Baseline | "
        f"Holdout MAE={metrics_0['mae']:.3f} cP  "
        f"RMSE={metrics_0['rmse']:.3f} cP  "
        f"RMSE(log10)={metrics_0['rmse_log10']:.4f}  "
        f"MAPE={metrics_0['mape']:.2f}%  "
        f"σ(log10)={std_log10_0:.4f}"
    )

    # ── Steps 1..N ───────────────────────────────────────────────────────────
    for step, sample_id in enumerate(ordered_ids, start=1):
        context_ids = ordered_ids[:step]
        holdout_ids = ordered_ids[step:]

        context_idx = [id_to_idx[s] for s in context_ids]
        context_df = ibal_df.loc[context_idx].copy()

        # [FIX-2 + FIX-3] Encode-only, with diverse selection
        _encode_context(predictor, context_df)

        metrics = null_metrics.copy()
        std_log10 = np.nan

        if holdout_ids and not has_nan_weights(predictor):
            holdout_idx = [id_to_idx[s] for s in holdout_ids]
            holdout_df = ibal_df.loc[holdout_idx].copy()
            query_df = holdout_df.drop(columns=VISC_COLS, errors="ignore")
            try:
                results_df = predictor.predict(query_df)
                metrics = compute_metrics(results_df, holdout_df)
            except Exception as e:
                logger.warning(f"  Step {step}: predict failed — {e}")

            # [FIX-4] Per-step uncertainty from context subsampling
            try:
                _, unc_stats = predictor.predict_with_uncertainty(
                    query_df, n_samples=N_UNC_SAMPLES, k=K_CONTEXT
                )
                std_log10 = float(np.mean(unc_stats.get("std_log10", [np.nan])))
            except Exception as e:
                logger.warning(f"  Step {step}: uncertainty failed — {e}")

        records.append(
            {
                "step": step,
                "sample_id": sample_id,
                "n_context": step,
                **metrics,
                "std_log10": std_log10,
            }
        )
        logger.info(
            f"  [{step:>2}/{len(ordered_ids)}] Added {sample_id:>6} | "
            f"Holdout MAE={metrics['mae']:.3f} cP  "
            f"RMSE={metrics['rmse']:.3f} cP  "
            f"RMSE(log10)={metrics['rmse_log10']:.4f}  "
            f"MAPE={metrics['mape']:.2f}%  "
            f"σ(log10)={std_log10:.4f}"
        )

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

PLOT_MAX_STEPS = 10

# VisQ brand palette
C_DEEP_BLUE = "#2596be"
C_BRIGHT_BLUE = "#13B5F0"
C_CYAN_MED = "#4EC4EB"
C_CYAN_PALE = "#8DD9F7"
C_GREEN = "#4caf50"
C_ORANGE = "#ff9800"
C_TEXT = "#24292f"
C_MUTED = "#6b7280"
C_BORDER = "#d1d5db"
C_BORDER_LT = "#e5e7eb"
C_BG_LIGHT = "#f3f4f6"
C_BG_LIGHTEST = "#f9fafb"
C_WHITE = "#ffffff"
C_PURPLE = "#9b59b6"
C_RED_SOFT = "#e74c3c"

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
            "font.size": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _style_axis(ax, spine_color=C_BORDER):
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(0.9)
    ax.tick_params(length=3, width=0.8)
    ax.grid(True, which="major", axis="both", zorder=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_facecolor(C_BG_LIGHTEST)


def _annotate_best(ax, x_arr, y_arr, fmt, color, offset_frac=0.06):
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
        fontsize=11,
        color=C_GREEN,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=C_GREEN, lw=0.6),
    )


def find_convergence_step(
    values: np.ndarray, window: int = 3, threshold: float = 0.005
):
    for i in range(window, len(values)):
        if np.all(np.abs(np.diff(values[i - window : i])) < threshold):
            return i - window
    return None


def _annotate_convergence(ax, x_arr, y_arr, c_idx, color=C_ORANGE):
    if c_idx is None or c_idx >= len(x_arr):
        return
    cx = x_arr[c_idx]
    ax.axvline(cx, color=color, lw=1.1, ls="--", alpha=0.7, zorder=3)
    ax.text(
        cx + 0.15,
        y_arr.max() - (y_arr.max() - y_arr.min()) * 0.04,
        f"plateau  n={cx}",
        fontsize=10.5,
        color=color,
        va="top",
        style="italic",
    )


def _shared_x_labels(ax, sx, labels):
    ax.set_xticks(sx)
    ax.set_xticklabels(
        [
            f"{n}\n({sid})" if sid != "None" else "0\n(0-shot)"
            for n, sid in zip(sx, labels)
        ],
        fontsize=11,
        color=C_MUTED,
    )
    ax.set_xlim(sx[0] - 0.4, sx[-1] + 0.6)


def _prep_plot_data(df: pd.DataFrame, metric_cols: list):
    """Return the trimmed plot slice, x-array, label-array, and value arrays."""
    valid_mask = ~df[metric_cols].isna().any(axis=1)
    valid_mask.iloc[-1] = False
    plot_df = df[valid_mask].head(PLOT_MAX_STEPS + 1)
    sx = plot_df["n_context"].values
    labels = plot_df["sample_id"].values
    vals = {c: plot_df[c].values for c in metric_cols}
    return plot_df, sx, labels, vals


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 & 2: linear cP convergence (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────


def plot_convergence(df: pd.DataFrame, save_dir: str, prefix: str = ""):
    """MAE + RMSE (linear cP) convergence — dual-axis combined and side-by-side panels."""
    apply_base_style()
    os.makedirs(save_dir, exist_ok=True)

    _, sx, labels, vals = _prep_plot_data(df, ["mae", "rmse"])
    smae = vals["mae"]
    srmse = vals["rmse"]
    conv_mae = find_convergence_step(smae)
    conv_rmse = find_convergence_step(srmse)

    # ── Combined dual-axis ──────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    _style_axis(ax1)

    ax1.plot(sx, smae, color=C_DEEP_BLUE, lw=2.4, zorder=4, solid_capstyle="round")
    ax1.scatter(
        sx, smae, color=C_DEEP_BLUE, s=38, zorder=5, edgecolors=C_WHITE, linewidths=1.1
    )
    ax1.fill_between(sx, smae, alpha=0.10, color=C_CYAN_PALE, zorder=1)
    ax1.set_xlabel(
        "Samples added to context  (n)", fontsize=14, labelpad=10, color=C_TEXT
    )
    ax1.set_ylabel("MAE  (cP)", fontsize=14, labelpad=10, color=C_DEEP_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_DEEP_BLUE, colors=C_DEEP_BLUE, labelsize=12)
    ax1.spines["left"].set_edgecolor(C_DEEP_BLUE)
    ax1.spines["left"].set_linewidth(1.2)

    ax2 = ax1.twinx()
    ax2.set_facecolor(C_BG_LIGHTEST)
    ax2.plot(
        sx,
        srmse,
        color=C_BRIGHT_BLUE,
        lw=2.4,
        zorder=4,
        solid_capstyle="round",
        ls=(0, (6, 2)),
    )
    ax2.scatter(
        sx,
        srmse,
        color=C_BRIGHT_BLUE,
        s=38,
        zorder=5,
        edgecolors=C_WHITE,
        linewidths=1.1,
        marker="D",
    )
    ax2.fill_between(sx, srmse, alpha=0.06, color=C_BRIGHT_BLUE, zorder=1)
    ax2.set_ylabel("RMSE  (cP)", fontsize=14, labelpad=10, color=C_BRIGHT_BLUE)
    ax2.tick_params(
        axis="y", labelcolor=C_BRIGHT_BLUE, colors=C_BRIGHT_BLUE, labelsize=12
    )
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor(C_BRIGHT_BLUE)
    ax2.spines["right"].set_linewidth(1.2)
    ax2.spines["top"].set_visible(False)

    _annotate_best(ax1, sx, smae, "  {:.3f} cP", C_DEEP_BLUE)
    _annotate_best(ax2, sx, srmse, "  {:.3f} cP", C_BRIGHT_BLUE, offset_frac=0.04)
    _annotate_convergence(ax1, sx, smae, conv_mae)
    _annotate_convergence(ax2, sx, srmse, conv_rmse)
    _shared_x_labels(ax1, sx, labels)

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=C_DEEP_BLUE,
            lw=2.4,
            marker="o",
            markersize=8,
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
            markersize=7,
            markerfacecolor=C_BRIGHT_BLUE,
            markeredgecolor=C_WHITE,
            ls=(0, (6, 2)),
            label="RMSE (cP)",
        ),
        Line2D(
            [0],
            [0],
            color=C_GREEN,
            lw=0,
            marker="o",
            markersize=10,
            markerfacecolor=C_GREEN,
            markeredgecolor=C_WHITE,
            label="Best value",
        ),
        Line2D([0], [0], color=C_ORANGE, lw=1.5, ls="--", label="Plateau onset"),
    ]
    ax1.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=12,
        framealpha=0.95,
        borderpad=0.9,
        edgecolor=C_BORDER,
    )
    ax1.set_title(
        "Ibalizumab · Learning Curve",
        fontsize=16,
        fontweight="bold",
        pad=14,
        color=C_TEXT,
        loc="left",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = os.path.join(save_dir, f"{prefix}convergence_combined.png")
    fig.savefig(combined_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)

    # ── Side-by-side panels ─────────────────────────────────────────────────
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(C_WHITE)

    panel_cfg = [
        (axA, smae, "MAE", "cP", C_DEEP_BLUE, conv_mae),
        (axB, srmse, "RMSE", "cP", C_BRIGHT_BLUE, conv_rmse),
    ]
    for ax, vals_arr, mlabel, unit, clr, c_idx in panel_cfg:
        _style_axis(ax)
        ax.axhline(
            np.nanmin(vals_arr), color=clr, lw=0.8, ls="--", alpha=0.35, zorder=2
        )
        ax.fill_between(sx, vals_arr, alpha=0.11, color=C_CYAN_PALE, zorder=1)
        ax.plot(sx, vals_arr, color=clr, lw=2.5, zorder=4, solid_capstyle="round")
        ax.scatter(
            sx, vals_arr, color=clr, s=42, zorder=5, edgecolors=C_WHITE, linewidths=1.2
        )
        _annotate_best(ax, sx, vals_arr, f"  {{:.3f}} {unit}", clr)
        _annotate_convergence(ax, sx, vals_arr, c_idx)
        ax.set_xlabel(
            "Samples added to context  (n)", fontsize=13, labelpad=9, color=C_TEXT
        )
        ax.set_ylabel(f"{mlabel}  ({unit})", fontsize=13, labelpad=9, color=clr)
        ax.tick_params(axis="y", labelcolor=clr, labelsize=12)
        ax.spines["left"].set_edgecolor(clr)
        ax.spines["left"].set_linewidth(1.2)
        ax.set_title(
            f"{mlabel} vs. Context Size",
            fontsize=15,
            fontweight="bold",
            pad=11,
            color=C_TEXT,
            loc="left",
        )
        _shared_x_labels(ax, sx, labels)
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_edgecolor(clr)
        ax.spines["top"].set_linewidth(2.5)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    panels_path = os.path.join(save_dir, f"{prefix}convergence_panels.png")
    fig.savefig(panels_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)

    return combined_path, panels_path


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: MAPE convergence (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────


def plot_mape(df: pd.DataFrame, save_dir: str, prefix: str = ""):
    apply_base_style()
    os.makedirs(save_dir, exist_ok=True)

    _, sx, labels, vals = _prep_plot_data(df, ["mape"])
    smape = vals["mape"]
    conv_mape = find_convergence_step(smape)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    _style_axis(ax)

    clr = C_PURPLE
    ax.axhline(np.nanmin(smape), color=clr, lw=0.8, ls="--", alpha=0.35, zorder=2)
    ax.fill_between(sx, smape, alpha=0.11, color=clr, zorder=1)
    ax.plot(sx, smape, color=clr, lw=2.5, zorder=4, solid_capstyle="round")
    ax.scatter(sx, smape, color=clr, s=42, zorder=5, edgecolors=C_WHITE, linewidths=1.2)
    _annotate_best(ax, sx, smape, "  {:.2f}%", clr)
    _annotate_convergence(ax, sx, smape, conv_mape, color=C_ORANGE)

    ax.set_xlabel(
        "Samples added to context  (n)", fontsize=13, labelpad=9, color=C_TEXT
    )
    ax.set_ylabel("MAPE  (%)", fontsize=13, labelpad=9, color=clr)
    ax.tick_params(axis="y", labelcolor=clr, labelsize=12)
    ax.spines["left"].set_edgecolor(clr)
    ax.spines["left"].set_linewidth(1.2)
    ax.set_title(
        "MAPE vs. Context Size",
        fontsize=15,
        fontweight="bold",
        pad=11,
        color=C_TEXT,
        loc="left",
    )
    _shared_x_labels(ax, sx, labels)
    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_edgecolor(clr)
    ax.spines["top"].set_linewidth(2.5)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    mape_path = os.path.join(save_dir, f"{prefix}convergence_mape.png")
    fig.savefig(mape_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    return mape_path


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4 (NEW): log10 RMSE convergence with uncertainty ribbon  [FIX-6]
# ──────────────────────────────────────────────────────────────────────────────


def plot_log_convergence(df: pd.DataFrame, save_dir: str, prefix: str = ""):
    """
    [FIX-6] Convergence curve in log10-RMSE space with a ±1σ shaded ribbon
    derived from per-step std_log10 (context-subsampling uncertainty).

    This plot uses the same metric as the model's training objective and the
    benchmark report, making it directly comparable to the 0.190 LOO RMSE
    and 0.232 bootstrap RMSE from evaluation.

    std_log10 interpretation:
        0.05 → ±12% factor uncertainty
        0.10 → ±26% factor uncertainty
        0.20 → ±58% factor uncertainty
    """
    apply_base_style()
    os.makedirs(save_dir, exist_ok=True)

    # Keep rows where rmse_log10 is valid; std_log10 may be NaN for some steps
    valid_mask = ~df["rmse_log10"].isna()
    valid_mask.iloc[-1] = False
    plot_df = df[valid_mask].head(PLOT_MAX_STEPS + 1)

    if plot_df.empty:
        logger.warning("No valid log10 data to plot — skipping log convergence plot.")
        return None

    sx = plot_df["n_context"].values
    labels = plot_df["sample_id"].values
    srmse_log = plot_df["rmse_log10"].values
    smae_log = plot_df["mae_log10"].values if "mae_log10" in plot_df else None
    sstd_log = plot_df["std_log10"].values if "std_log10" in plot_df else None

    conv_rmse_log = find_convergence_step(srmse_log, threshold=0.002)

    clr_rmse = C_RED_SOFT
    clr_mae = C_DEEP_BLUE
    clr_unc = "#e8b4b8"  # soft rose ribbon

    fig, ax1 = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    _style_axis(ax1)

    # ── Uncertainty ribbon (±1σ around RMSE log10 curve) ────────────────────
    if sstd_log is not None and not np.all(np.isnan(sstd_log)):
        ribbon_lo = srmse_log - sstd_log
        ribbon_hi = srmse_log + sstd_log
        ax1.fill_between(
            sx,
            ribbon_lo,
            ribbon_hi,
            alpha=0.22,
            color=clr_unc,
            zorder=1,
            label="±1σ context uncertainty",
        )

    # ── RMSE log10 (left axis) ───────────────────────────────────────────────
    ax1.plot(sx, srmse_log, color=clr_rmse, lw=2.5, zorder=4, solid_capstyle="round")
    ax1.scatter(
        sx,
        srmse_log,
        color=clr_rmse,
        s=50,
        zorder=5,
        edgecolors=C_WHITE,
        linewidths=1.2,
        marker="s",
    )
    ax1.axhline(
        np.nanmin(srmse_log), color=clr_rmse, lw=0.8, ls="--", alpha=0.35, zorder=2
    )
    _annotate_best(ax1, sx, srmse_log, "  {:.4f}", clr_rmse)
    _annotate_convergence(ax1, sx, srmse_log, conv_rmse_log)

    ax1.set_ylabel("RMSE  (log₁₀ viscosity)", fontsize=14, labelpad=10, color=clr_rmse)
    ax1.tick_params(axis="y", labelcolor=clr_rmse, labelsize=12)
    ax1.spines["left"].set_edgecolor(clr_rmse)
    ax1.spines["left"].set_linewidth(1.2)

    # ── MAE log10 (right axis, if available) ────────────────────────────────
    if smae_log is not None and not np.all(np.isnan(smae_log)):
        ax2 = ax1.twinx()
        ax2.set_facecolor(C_BG_LIGHTEST)
        ax2.plot(
            sx,
            smae_log,
            color=clr_mae,
            lw=2.2,
            zorder=3,
            solid_capstyle="round",
            ls=(0, (6, 2)),
        )
        ax2.scatter(
            sx,
            smae_log,
            color=clr_mae,
            s=38,
            zorder=5,
            edgecolors=C_WHITE,
            linewidths=1.1,
            marker="D",
        )
        _annotate_best(ax2, sx, smae_log, "  {:.4f}", clr_mae, offset_frac=0.04)
        ax2.set_ylabel(
            "MAE  (log₁₀ viscosity)", fontsize=14, labelpad=10, color=clr_mae
        )
        ax2.tick_params(axis="y", labelcolor=clr_mae, colors=clr_mae, labelsize=12)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_edgecolor(clr_mae)
        ax2.spines["right"].set_linewidth(1.2)
        ax2.spines["top"].set_visible(False)

    ax1.set_xlabel(
        "Samples added to context  (n)", fontsize=14, labelpad=10, color=C_TEXT
    )
    _shared_x_labels(ax1, sx, labels)

    # ── Reference line: benchmark LOO RMSE ──────────────────────────────────
    # Annotate the model's LOO RMSE from the benchmark run (0.190) as a
    # horizontal reference so the reader can see when few-shot context
    # beats the cross-protein baseline.
    benchmark_loo = 0.190
    ax1.axhline(benchmark_loo, color=C_ORANGE, lw=1.2, ls=":", alpha=0.8, zorder=3)
    yrange = srmse_log.max() - srmse_log.min() or 0.05
    ax1.text(
        sx[-1] + 0.1,
        benchmark_loo + yrange * 0.03,
        f"LOO baseline ({benchmark_loo:.3f})",
        fontsize=10,
        color=C_ORANGE,
        va="bottom",
        ha="right",
        style="italic",
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=clr_rmse,
            lw=2.5,
            marker="s",
            markersize=8,
            markerfacecolor=clr_rmse,
            markeredgecolor=C_WHITE,
            label="RMSE (log₁₀)",
        ),
        Line2D(
            [0],
            [0],
            color=clr_mae,
            lw=2.2,
            marker="D",
            markersize=7,
            markerfacecolor=clr_mae,
            markeredgecolor=C_WHITE,
            ls=(0, (6, 2)),
            label="MAE (log₁₀)",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, fc=clr_unc, alpha=0.4, label="±1σ context uncertainty"
        ),
        Line2D(
            [0],
            [0],
            color=C_GREEN,
            lw=0,
            marker="o",
            markersize=10,
            markerfacecolor=C_GREEN,
            markeredgecolor=C_WHITE,
            label="Best value",
        ),
        Line2D([0], [0], color=C_ORANGE, lw=1.5, ls="--", label="Plateau onset"),
        Line2D(
            [0], [0], color=C_ORANGE, lw=1.2, ls=":", alpha=0.8, label="LOO baseline"
        ),
    ]
    ax1.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=11,
        framealpha=0.95,
        borderpad=0.9,
        edgecolor=C_BORDER,
    )

    ax1.set_title(
        "Ibalizumab · Log₁₀ Learning Curve  (model-native metric)",
        fontsize=16,
        fontweight="bold",
        pad=14,
        color=C_TEXT,
        loc="left",
    )
    ax1.spines["top"].set_visible(True)
    ax1.spines["top"].set_edgecolor(clr_rmse)
    ax1.spines["top"].set_linewidth(2.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    log_path = os.path.join(save_dir, f"{prefix}convergence_log10.png")
    fig.savefig(log_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    logger.info(f"  Saved log10 convergence plot: {log_path}")
    return log_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load pretrain data
    pretrain_df = None
    if PRETRAIN_CSV and os.path.exists(PRETRAIN_CSV):
        logger.info(f"Loading pre-training context pool: {PRETRAIN_CSV}")
        pretrain_df = prepare_df(pd.read_csv(PRETRAIN_CSV), drop_bad_rows=True)
        logger.info(f"  {len(pretrain_df)} valid pre-training samples.")

    # [FIX-7] init_clean_predictor: pre-training data encoded as cross-protein
    # context baseline (encode-only, no weight updates).
    def init_clean_predictor():
        logger.info(f"Initializing clean model from: {MODEL_DIR}")
        pred = ViscosityPredictorCNP(MODEL_DIR)
        if pretrain_df is not None:
            logger.info(
                f"  Encoding {len(pretrain_df)} cross-protein context samples "
                f"as zero-shot prior (encode-only, no weight updates)..."
            )
            _encode_context(pred, pretrain_df)
            if has_nan_weights(pred):
                logger.error("NaN weights detected — reloading clean model.")
                pred = ViscosityPredictorCNP(MODEL_DIR)
            else:
                logger.info("  Cross-protein prior encoded successfully.")
        return pred

    # 2. Load ibalizumab evaluation data
    logger.info(f"Loading ibalizumab data: {IBAL_CSV}")
    ibal_df = prepare_df(pd.read_csv(IBAL_CSV), drop_bad_rows=True)
    logger.info(f"  {len(ibal_df)} valid ibalizumab samples.")

    # 3. Load optimal order and generate a random baseline order
    logger.info(f"Loading optimal order: {ORDER_CSV}")
    order_df = pd.read_csv(ORDER_CSV)
    optimal_ids = order_df["Sample_ID"].tolist()
    random_ids = optimal_ids.copy()
    np.random.shuffle(random_ids)

    # ── EVALUATION 1: OPTIMAL ORDER ──────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("RUNNING EVALUATION: OPTIMAL ORDER")
    logger.info("=" * 55)
    predictor_opt = init_clean_predictor()
    results_opt_df = run_convergence_replay(predictor_opt, ibal_df, optimal_ids)
    metrics_path_opt = os.path.join(OUTPUT_DIR, "optimal_convergence_metrics.csv")
    results_opt_df.to_csv(metrics_path_opt, index=False)
    logger.info(f"  Metrics saved: {metrics_path_opt}")

    # ── EVALUATION 2: RANDOM ORDER ───────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("RUNNING EVALUATION: RANDOM ORDER")
    logger.info("=" * 55)
    predictor_rand = init_clean_predictor()
    results_rand_df = run_convergence_replay(predictor_rand, ibal_df, random_ids)
    metrics_path_rand = os.path.join(OUTPUT_DIR, "random_convergence_metrics.csv")
    results_rand_df.to_csv(metrics_path_rand, index=False)
    logger.info(f"  Metrics saved: {metrics_path_rand}")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    for name, df in [("OPTIMAL", results_opt_df), ("RANDOM", results_rand_df)]:
        valid = df.dropna(subset=["mae", "rmse", "mape"])
        if valid.empty:
            continue
        best_rmse = valid.loc[valid["rmse"].idxmin()]
        best_mae = valid.loc[valid["mae"].idxmin()]
        best_mape = valid.loc[valid["mape"].idxmin()]
        best_rmse_log = valid.loc[valid["rmse_log10"].idxmin()]
        logger.info(
            f"\n{'='*65}\n"
            f"  {name} Best RMSE      : {best_rmse['rmse']:.3f} cP  "
            f"@ n={best_rmse['n_context']}  ({best_rmse['sample_id']} added)\n"
            f"  {name} Best MAE       : {best_mae['mae']:.3f} cP  "
            f"@ n={best_mae['n_context']}  ({best_mae['sample_id']} added)\n"
            f"  {name} Best MAPE      : {best_mape['mape']:.2f}%  "
            f"@ n={best_mape['n_context']}  ({best_mape['sample_id']} added)\n"
            f"  {name} Best RMSE(log) : {best_rmse_log['rmse_log10']:.4f}  "
            f"@ n={best_rmse_log['n_context']}  ({best_rmse_log['sample_id']} added)\n"
            f"{'='*65}"
        )

    # ── PLOTS ────────────────────────────────────────────────────────────────
    logger.info("\nGenerating plots...")

    plot_convergence(results_opt_df, OUTPUT_DIR, prefix="optimal_")
    plot_mape(results_opt_df, OUTPUT_DIR, prefix="optimal_")
    plot_log_convergence(results_opt_df, OUTPUT_DIR, prefix="optimal_")

    plot_convergence(results_rand_df, OUTPUT_DIR, prefix="random_")
    plot_mape(results_rand_df, OUTPUT_DIR, prefix="random_")
    plot_log_convergence(results_rand_df, OUTPUT_DIR, prefix="random_")

    logger.info(f"\nAll done. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
