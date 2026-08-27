from __future__ import annotations

import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd
import torch

from visqai.constants import (
    SHEAR_COLS,
    SHORT_SHEAR_LABELS,
    SHEAR_RATES,
    C_DEEP_BLUE,
    C_BRIGHT_BLUE,
    C_CYAN_PALE,
    C_GREEN,
    C_ORANGE,
    C_PURPLE,
    C_RED_SOFT,
    C_TEXT,
    C_MUTED,
    C_BORDER,
    C_BG_LIGHTEST,
    C_WHITE,
)
from visqai.eval.metrics import compute_metrics
from visqai.eval.style import mpl, apply_style, style_axis
from visqai import constants, paths
from visqai.features.dataprocessor import prepare_df
from visqai.inference.predictor import ViscosityPredictorCNP
from visqai.logging_config import configure_logging

warnings.filterwarnings("ignore")

logger = logging.getLogger("LearningCurveEval")


def save_state(predictor) -> dict:
    return {
        "memory_vector": (
            predictor.memory_vector.clone() if predictor.memory_vector is not None else None
        ),
        "context_t": predictor.context_t.clone() if predictor.context_t is not None else None,
        "offset_hat": predictor.offset_hat,
        "conc_hat": predictor.conc_hat,
        "conc_center": predictor.conc_center,
        "slope_hat": predictor.slope_hat,
        "slope_center": predictor.slope_center,
        "n_context_points": predictor.n_context_points,
    }


def restore_state(predictor, snap: dict):
    """Restore latent memory state from snapshot."""
    predictor.memory_vector = (
        snap["memory_vector"].clone() if snap["memory_vector"] is not None else None
    )
    predictor.context_t = snap["context_t"].clone() if snap["context_t"] is not None else None
    predictor.offset_hat = snap.get("offset_hat", 0.0)
    predictor.conc_hat = snap.get("conc_hat", 0.0)
    predictor.conc_center = snap.get("conc_center", 0.0)
    predictor.slope_hat = snap.get("slope_hat", 0.0)
    predictor.slope_center = snap.get("slope_center", 0.0)
    predictor.n_context_points = snap.get("n_context_points", 0)


def reset_memory(predictor):
    """Clear latent state without touching model weights."""
    predictor.memory_vector = None
    predictor.context_t = None


def has_nan_weights(predictor) -> bool:
    return any(torch.isnan(p).any() for p in predictor.model.parameters())


def predict_from_built(predictor, df_built: pd.DataFrame, orig_df: pd.DataFrame) -> pd.DataFrame:
    q_static, q_shear, _ = predictor._preprocess_built(df_built)
    return predictor._predict_from_tensors(q_static, q_shear, orig_df)


SHAPE_FLAT_EPS = 0.005


def _classify_slopes(log_visc: np.ndarray, eps: float = SHAPE_FLAT_EPS) -> np.ndarray:
    """Per-segment direction: -1 thinning, 0 flat, +1 thickening."""
    d = np.diff(log_visc)
    out = np.zeros_like(d)
    out[d < -eps] = -1.0
    out[d > eps] = 1.0
    return out


def compute_shape_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    m = np.isfinite(a) & np.isfinite(p) & (a > 0) & (p > 0)
    nan = {
        "shape_rmse_log10": np.nan,
        "slope_sign_agree": np.nan,
        "slope_n_sloped": 0,
        "plateau_err_log10": np.nan,
        "thin_ratio_log_err": np.nan,
    }
    if m.sum() < 2:
        return nan

    la = np.log10(a[m])
    lp = np.log10(p[m])

    thin_ratio_log_err = float(abs((la[0] - la[-1]) - (lp[0] - lp[-1])))

    if m.sum() < 3:
        return {**nan, "thin_ratio_log_err": thin_ratio_log_err}

    shape_rmse = float(np.sqrt(np.mean(((la - la.mean()) - (lp - lp.mean())) ** 2)))
    da = np.diff(la)
    dp = np.diff(lp)
    sloped = np.abs(da) >= SHAPE_FLAT_EPS
    n_sloped = int(sloped.sum())
    if n_sloped > 0:
        slope_sign_agree = float((np.sign(da[sloped]) == np.sign(dp[sloped])).mean())
    else:
        slope_sign_agree = np.nan
    plateau_err = float(abs((la[0] - la[2]) - (lp[0] - lp[2])))

    return {
        "shape_rmse_log10": shape_rmse,
        "slope_sign_agree": slope_sign_agree,
        "slope_n_sloped": n_sloped,
        "plateau_err_log10": plateau_err,
        "thin_ratio_log_err": thin_ratio_log_err,
    }


def _aggregate_shape(profs: dict) -> dict:
    agg = {
        "shape_rmse_log10": np.nan,
        "slope_sign_agree": np.nan,
        "plateau_err_log10": np.nan,
        "thin_ratio_log_err": np.nan,
    }
    if not profs:
        return agg
    for key in ["shape_rmse_log10", "plateau_err_log10", "thin_ratio_log_err"]:
        vals = [p[key] for p in profs.values() if np.isfinite(p.get(key, np.nan))]
        if vals:
            agg[key] = float(np.mean(vals))
    num = den = 0.0
    for p in profs.values():
        ns = p.get("slope_n_sloped", 0)
        sa = p.get("slope_sign_agree", np.nan)
        if ns > 0 and np.isfinite(sa):
            num += sa * ns
            den += ns
    if den > 0:
        agg["slope_sign_agree"] = float(num / den)
    return agg


PLOT_MAX_STEPS = 10


def annotate_best(ax, x_arr, y_arr, fmt, color, offset_frac=0.06):
    b = np.argmin(y_arr)
    ax.scatter(
        [x_arr[b]], [y_arr[b]], color=C_GREEN, s=100, zorder=7, edgecolors=C_WHITE, linewidths=1.4
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


def find_convergence_step(values: np.ndarray, window: int = 3, threshold: float = 0.005):
    for i in range(window, len(values)):
        if np.all(np.abs(np.diff(values[i - window : i])) < threshold):
            return i - window
    return None


def annotate_convergence(ax, x_arr, y_arr, c_idx, color=C_ORANGE):
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


def shared_x_labels(ax, sx, labels):
    many = len(sx) > 14
    ax.set_xticks(sx)
    ax.set_xticklabels(
        (
            [f"{n} ({sid})" if sid != "None" else "0 (0-shot)" for n, sid in zip(sx, labels)]
            if many
            else [f"{n}\n({sid})" if sid != "None" else "0\n(0-shot)" for n, sid in zip(sx, labels)]
        ),
        fontsize=8.5 if many else 11,
        color=C_MUTED,
        rotation=75 if many else 0,
        ha="right" if many else "center",
    )
    ax.set_xlim(sx[0] - 0.4, sx[-1] + 0.6)


def prep_plot_data(df: pd.DataFrame, metric_cols: list, max_steps: int = PLOT_MAX_STEPS):
    """Return the trimmed plot slice, x-array, label-array, and value arrays."""
    valid_mask = ~df[metric_cols].isna().any(axis=1)
    valid_mask.iloc[-1] = False
    plot_df = df[valid_mask].head(max_steps + 1)
    sx = plot_df["n_context"].values
    labels = plot_df["sample_id"].values
    vals = {c: plot_df[c].values for c in metric_cols}
    return plot_df, sx, labels, vals


def plot_convergence(
    df: pd.DataFrame, save_dir: str, prefix: str = "", max_steps: int = PLOT_MAX_STEPS
):
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

    _, sx, labels, vals = prep_plot_data(df, ["mae", "rmse"], max_steps=max_steps)
    smae = vals["mae"]
    srmse = vals["rmse"]
    conv_mae = find_convergence_step(smae)
    conv_rmse = find_convergence_step(srmse)

    fig, ax1 = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    style_axis(ax1, ticker)

    ax1.plot(sx, smae, color=C_DEEP_BLUE, lw=2.4, zorder=4, solid_capstyle="round")
    ax1.scatter(sx, smae, color=C_DEEP_BLUE, s=38, zorder=5, edgecolors=C_WHITE, linewidths=1.1)
    ax1.fill_between(sx, smae, alpha=0.10, color=C_CYAN_PALE, zorder=1)
    ax1.set_xlabel("Samples added to context  (n)", fontsize=14, labelpad=10, color=C_TEXT)
    ax1.set_ylabel("MAE  (cP)", fontsize=14, labelpad=10, color=C_DEEP_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_DEEP_BLUE, colors=C_DEEP_BLUE, labelsize=12)
    ax1.spines["left"].set_edgecolor(C_DEEP_BLUE)
    ax1.spines["left"].set_linewidth(1.2)

    ax2 = ax1.twinx()
    ax2.set_facecolor(C_BG_LIGHTEST)
    ax2.plot(
        sx, srmse, color=C_BRIGHT_BLUE, lw=2.4, zorder=4, solid_capstyle="round", ls=(0, (6, 2))
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
    ax2.tick_params(axis="y", labelcolor=C_BRIGHT_BLUE, colors=C_BRIGHT_BLUE, labelsize=12)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor(C_BRIGHT_BLUE)
    ax2.spines["right"].set_linewidth(1.2)
    ax2.spines["top"].set_visible(False)

    annotate_best(ax1, sx, smae, "  {:.3f} cP", C_DEEP_BLUE)
    annotate_best(ax2, sx, srmse, "  {:.3f} cP", C_BRIGHT_BLUE, offset_frac=0.04)
    annotate_convergence(ax1, sx, smae, conv_mae)
    annotate_convergence(ax2, sx, srmse, conv_rmse)
    shared_x_labels(ax1, sx, labels)

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

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor(C_WHITE)

    panel_cfg = [
        (axA, smae, "MAE", "cP", C_DEEP_BLUE, conv_mae),
        (axB, srmse, "RMSE", "cP", C_BRIGHT_BLUE, conv_rmse),
    ]
    for ax, vals_arr, mlabel, unit, clr, c_idx in panel_cfg:
        style_axis(ax, ticker)
        ax.axhline(np.nanmin(vals_arr), color=clr, lw=0.8, ls="--", alpha=0.35, zorder=2)
        ax.fill_between(sx, vals_arr, alpha=0.11, color=C_CYAN_PALE, zorder=1)
        ax.plot(sx, vals_arr, color=clr, lw=2.5, zorder=4, solid_capstyle="round")
        ax.scatter(sx, vals_arr, color=clr, s=42, zorder=5, edgecolors=C_WHITE, linewidths=1.2)
        annotate_best(ax, sx, vals_arr, f"  {{:.3f}} {unit}", clr)
        annotate_convergence(ax, sx, vals_arr, c_idx)
        ax.set_xlabel("Samples added to context  (n)", fontsize=13, labelpad=9, color=C_TEXT)
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
        shared_x_labels(ax, sx, labels)
        ax.spines["top"].set_visible(True)
        ax.spines["top"].set_edgecolor(clr)
        ax.spines["top"].set_linewidth(2.5)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    panels_path = os.path.join(save_dir, f"{prefix}convergence_panels.png")
    fig.savefig(panels_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)

    return combined_path, panels_path


def plot_mape(df: pd.DataFrame, save_dir: str, prefix: str = "", max_steps: int = PLOT_MAX_STEPS):
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

    _, sx, labels, vals = prep_plot_data(df, ["mape"], max_steps=max_steps)
    smape = vals["mape"]
    conv_mape = find_convergence_step(smape)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    style_axis(ax, ticker)

    clr = C_PURPLE
    ax.axhline(np.nanmin(smape), color=clr, lw=0.8, ls="--", alpha=0.35, zorder=2)
    ax.fill_between(sx, smape, alpha=0.11, color=clr, zorder=1)
    ax.plot(sx, smape, color=clr, lw=2.5, zorder=4, solid_capstyle="round")
    ax.scatter(sx, smape, color=clr, s=42, zorder=5, edgecolors=C_WHITE, linewidths=1.2)
    annotate_best(ax, sx, smape, "  {:.2f}%", clr)
    annotate_convergence(ax, sx, smape, conv_mape, color=C_ORANGE)

    ax.set_xlabel("Samples added to context  (n)", fontsize=13, labelpad=9, color=C_TEXT)
    ax.set_ylabel("MAPE  (%)", fontsize=13, labelpad=9, color=clr)
    ax.tick_params(axis="y", labelcolor=clr, labelsize=12)
    ax.spines["left"].set_edgecolor(clr)
    ax.spines["left"].set_linewidth(1.2)
    ax.set_title(
        "MAPE vs. Context Size", fontsize=15, fontweight="bold", pad=11, color=C_TEXT, loc="left"
    )
    shared_x_labels(ax, sx, labels)
    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_edgecolor(clr)
    ax.spines["top"].set_linewidth(2.5)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    mape_path = os.path.join(save_dir, f"{prefix}convergence_mape.png")
    fig.savefig(mape_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    return mape_path


def plot_log_convergence(
    df: pd.DataFrame, save_dir: str, prefix: str = "", max_steps: int = PLOT_MAX_STEPS
):
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

    valid_mask = ~df["rmse_log10"].isna()
    valid_mask.iloc[-1] = False
    plot_df = df[valid_mask].head(max_steps + 1)

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
    clr_unc = "#e8b4b8"

    fig, ax1 = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    style_axis(ax1, ticker)

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
    ax1.axhline(np.nanmin(srmse_log), color=clr_rmse, lw=0.8, ls="--", alpha=0.35, zorder=2)
    annotate_best(ax1, sx, srmse_log, "  {:.4f}", clr_rmse)
    annotate_convergence(ax1, sx, srmse_log, conv_rmse_log)

    ax1.set_ylabel("RMSE  (log₁₀ viscosity)", fontsize=14, labelpad=10, color=clr_rmse)
    ax1.tick_params(axis="y", labelcolor=clr_rmse, labelsize=12)
    ax1.spines["left"].set_edgecolor(clr_rmse)
    ax1.spines["left"].set_linewidth(1.2)

    if smae_log is not None and not np.all(np.isnan(smae_log)):
        ax2 = ax1.twinx()
        ax2.set_facecolor(C_BG_LIGHTEST)
        ax2.plot(
            sx, smae_log, color=clr_mae, lw=2.2, zorder=3, solid_capstyle="round", ls=(0, (6, 2))
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
        annotate_best(ax2, sx, smae_log, "  {:.4f}", clr_mae, offset_frac=0.04)
        ax2.set_ylabel("MAE  (log₁₀ viscosity)", fontsize=14, labelpad=10, color=clr_mae)
        ax2.tick_params(axis="y", labelcolor=clr_mae, colors=clr_mae, labelsize=12)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_edgecolor(clr_mae)
        ax2.spines["right"].set_linewidth(1.2)
        ax2.spines["top"].set_visible(False)

    ax1.set_xlabel("Samples added to context  (n)", fontsize=14, labelpad=10, color=C_TEXT)
    shared_x_labels(ax1, sx, labels)

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
        plt.Rectangle((0, 0), 1, 1, fc=clr_unc, alpha=0.4, label="±1σ context uncertainty"),
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
        Line2D([0], [0], color=C_ORANGE, lw=1.2, ls=":", alpha=0.8, label="LOO baseline"),
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


def plot_shape_convergence(
    df: pd.DataFrame, save_dir: str, prefix: str = "", max_steps: int = PLOT_MAX_STEPS
):
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

    need = ["slope_sign_agree", "shape_rmse_log10"]
    if not all(c in df.columns for c in need):
        logger.warning("Shape columns absent — skipping shape convergence plot.")
        return None
    valid = ~df[need].isna().any(axis=1)
    valid.iloc[-1] = False
    plot_df = df[valid].head(max_steps + 1)
    if plot_df.empty:
        logger.warning("No valid shape data to plot — skipping.")
        return None

    sx = plot_df["n_context"].values
    labels = plot_df["sample_id"].values
    s_agree = plot_df["slope_sign_agree"].values * 100.0
    s_shape = plot_df["shape_rmse_log10"].values

    clr_agree = C_GREEN
    clr_shape = C_RED_SOFT

    fig, ax1 = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    style_axis(ax1, ticker)

    ax1.plot(sx, s_agree, color=clr_agree, lw=2.5, zorder=4, solid_capstyle="round")
    ax1.scatter(sx, s_agree, color=clr_agree, s=46, zorder=5, edgecolors=C_WHITE, linewidths=1.2)
    ax1.set_ylabel("Slope-sign agreement  (%)  ↑ better", fontsize=14, labelpad=10, color=clr_agree)
    ax1.tick_params(axis="y", labelcolor=clr_agree, labelsize=12)
    ax1.set_ylim(0, 105)
    ax1.spines["left"].set_edgecolor(clr_agree)
    ax1.spines["left"].set_linewidth(1.2)
    ax1.axhspan(75, 105, color=clr_agree, alpha=0.06, zorder=0)

    ax2 = ax1.twinx()
    ax2.set_facecolor(C_BG_LIGHTEST)
    ax2.plot(sx, s_shape, color=clr_shape, lw=2.4, zorder=4, ls=(0, (6, 2)), solid_capstyle="round")
    ax2.scatter(
        sx, s_shape, color=clr_shape, s=40, zorder=5, marker="D", edgecolors=C_WHITE, linewidths=1.1
    )
    ax2.set_ylabel(
        "Shape RMSE  (log₁₀, level-invariant)  ↓ better", fontsize=14, labelpad=10, color=clr_shape
    )
    ax2.tick_params(axis="y", labelcolor=clr_shape, labelsize=12)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor(clr_shape)
    ax2.spines["right"].set_linewidth(1.2)
    ax2.spines["top"].set_visible(False)

    ax1.set_xlabel("Samples added to context  (n)", fontsize=14, labelpad=10, color=C_TEXT)
    shared_x_labels(ax1, sx, labels)

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=clr_agree,
            lw=2.5,
            marker="o",
            markersize=8,
            markerfacecolor=clr_agree,
            markeredgecolor=C_WHITE,
            label="Slope-sign agreement (↑)",
        ),
        Line2D(
            [0],
            [0],
            color=clr_shape,
            lw=2.4,
            marker="D",
            markersize=7,
            markerfacecolor=clr_shape,
            markeredgecolor=C_WHITE,
            ls=(0, (6, 2)),
            label="Shape RMSE log₁₀ (↓)",
        ),
    ]
    ax1.legend(
        handles=legend_elements,
        loc="center right",
        fontsize=12,
        framealpha=0.95,
        borderpad=0.9,
        edgecolor=C_BORDER,
    )
    ax1.set_title(
        "Ibalizumab · Shape-Fidelity Convergence  (morphology, not level)",
        fontsize=16,
        fontweight="bold",
        pad=14,
        color=C_TEXT,
        loc="left",
    )
    ax1.spines["top"].set_visible(True)
    ax1.spines["top"].set_edgecolor(clr_agree)
    ax1.spines["top"].set_linewidth(2.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, f"{prefix}convergence_shape.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    logger.info(f"  Saved shape convergence plot: {path}")
    return path


def predict_profiles(predictor, samples_df: pd.DataFrame) -> dict:
    if samples_df.empty:
        return {}

    drop_targets = [c for c in samples_df.columns if c.startswith("Viscosity_")]
    query = samples_df.drop(columns=drop_targets, errors="ignore")

    try:
        results = predictor.predict(query)
    except Exception as e:
        logger.warning(f"  [profiles] predict failed — {e}")
        return {}

    pred_cols = [f"Pred_{c}" for c in SHEAR_COLS]
    out = {}
    for i, (_, srow) in enumerate(samples_df.iterrows()):
        sid = str(srow["ID"])
        actual = np.array(
            [float(srow[c]) if c in srow and pd.notnull(srow[c]) else np.nan for c in SHEAR_COLS]
        )
        rrow = results.iloc[i] if i < len(results) else None
        pred = np.full(len(SHEAR_COLS), np.nan)
        if rrow is not None:
            for j, pc in enumerate(pred_cols):
                if pc in results.columns and pd.notnull(rrow[pc]):
                    pred[j] = float(rrow[pc])
        m = ~np.isnan(actual) & ~np.isnan(pred) & (actual > 0) & (pred > 0)
        if m.any():
            rmse_log = float(np.sqrt(np.mean((np.log10(actual[m]) - np.log10(pred[m])) ** 2)))
        else:
            rmse_log = np.nan
        out[sid] = {
            "actual": actual,
            "pred": pred,
            "conc": float(srow.get("Protein_conc", np.nan)),
            "pH": float(srow.get("Buffer_pH", np.nan)),
            "rmse_log10": rmse_log,
            **compute_shape_metrics(actual, pred),
        }
    return out


def plot_sample_profile(sid: str, prof: dict, save_path: str, is_context: bool):
    """Render one sample's predicted-vs-actual shear-thinning profile."""
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    actual = prof["actual"]
    pred = prof["pred"]
    rates = np.array(SHEAR_RATES)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=140)
    fig.patch.set_facecolor(C_WHITE)
    style_axis(ax, ticker)

    frame_clr = C_GREEN if is_context else C_DEEP_BLUE
    role = "CONTEXT" if is_context else "HELD-OUT"

    ma = ~np.isnan(actual) & (actual > 0)
    if ma.any():
        ax.plot(
            rates[ma],
            actual[ma],
            color=C_TEXT,
            lw=2.6,
            marker="o",
            markersize=8,
            markerfacecolor=C_TEXT,
            markeredgecolor=C_WHITE,
            markeredgewidth=1.3,
            zorder=5,
            label="Actual",
            solid_capstyle="round",
        )
    mp = ~np.isnan(pred) & (pred > 0)
    if mp.any():
        ax.plot(
            rates[mp],
            pred[mp],
            color=frame_clr,
            lw=2.4,
            marker="D",
            markersize=7,
            markerfacecolor=frame_clr,
            markeredgecolor=C_WHITE,
            markeredgewidth=1.2,
            ls=(0, (6, 2)),
            zorder=4,
            label="Predicted",
            solid_capstyle="round",
        )
    mb = ma & mp
    if mb.any():
        ax.fill_between(rates[mb], actual[mb], pred[mb], color=frame_clr, alpha=0.12, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Shear rate  (s⁻¹)", fontsize=13, labelpad=9, color=C_TEXT)
    ax.set_ylabel("Viscosity  (cP)", fontsize=13, labelpad=9, color=C_TEXT)
    ax.set_xticks(rates)
    ax.set_xticklabels(SHORT_SHEAR_LABELS, fontsize=11, color=C_MUTED)

    def _thin_ratio(arr, mask):
        v = arr[mask]
        return (v[0] / v[-1]) if len(v) >= 2 and v[-1] > 0 else np.nan

    act_ratio = _thin_ratio(actual, ma)
    pred_ratio = _thin_ratio(pred, mp)
    rmse_log = prof.get("rmse_log10", np.nan)
    shape_rmse = prof.get("shape_rmse_log10", np.nan)
    slope_agree = prof.get("slope_sign_agree", np.nan)
    plateau_err = prof.get("plateau_err_log10", np.nan)

    subtitle = f"conc={prof['conc']:.0f} mg/mL · pH={prof['pH']:.1f}  |  RMSE(log₁₀)={rmse_log:.3f}"
    shape_txt = (
        f"thinning  actual {act_ratio:.2f}×   pred {pred_ratio:.2f}×"
        if not (np.isnan(act_ratio) or np.isnan(pred_ratio))
        else ""
    )
    shape_txt2 = ""
    if np.isfinite(shape_rmse) and np.isfinite(slope_agree):
        shape_txt2 = f"shape-RMSE {shape_rmse:.3f}   slope-match {slope_agree:.0%}   plateauΔ {plateau_err:.3f}"

    ax.set_title(
        f"{role}  ·  Sample {sid}",
        fontsize=14,
        fontweight="bold",
        pad=30,
        color=frame_clr,
        loc="left",
    )
    ax.text(0.0, 1.075, subtitle, transform=ax.transAxes, fontsize=10.5, color=C_MUTED, va="bottom")
    if shape_txt:
        ax.text(
            0.0,
            1.038,
            shape_txt,
            transform=ax.transAxes,
            fontsize=10,
            color=frame_clr,
            va="bottom",
            style="italic",
        )
    if shape_txt2:
        shape_clr = (
            C_RED_SOFT if slope_agree < 0.5 else (C_ORANGE if slope_agree < 0.75 else C_GREEN)
        )
        ax.text(
            0.0,
            1.002,
            shape_txt2,
            transform=ax.transAxes,
            fontsize=9.5,
            color=shape_clr,
            va="bottom",
            fontweight="bold",
        )

    for spine in ["top", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_edgecolor(frame_clr)
        ax.spines[spine].set_linewidth(2.2)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", fontsize=11, framealpha=0.95, borderpad=0.8, edgecolor=C_BORDER)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)


_NULL_SHAPE_AGG = {
    "shape_rmse_log10": np.nan,
    "slope_sign_agree": np.nan,
    "plateau_err_log10": np.nan,
    "thin_ratio_log_err": np.nan,
}


def encode_context(
    predictor,
    context_df: pd.DataFrame,
    learn_steps: int = 50,
    learn_lr: float = 1e-3,
    n_draws: int = 20,
    k_context: int = 8,
    max_ctx_pool: int = 15,
):
    reset_memory(predictor)
    if context_df.empty:
        return

    if hasattr(predictor, "_select_diverse_context"):
        context_df = predictor._select_diverse_context(context_df, max_k=max_ctx_pool)

    predictor.learn(context_df, steps=learn_steps, lr=learn_lr, n_draws=n_draws, k=k_context)


def _shape_index_row(step, sid, role, n_context, prof):
    return {
        "step": step,
        "sample_id": sid,
        "role": role,
        "n_context": n_context,
        "conc": prof.get("conc", np.nan),
        "pH": prof.get("pH", np.nan),
        "rmse_log10": prof.get("rmse_log10", np.nan),
        "shape_rmse_log10": prof.get("shape_rmse_log10", np.nan),
        "slope_sign_agree": prof.get("slope_sign_agree", np.nan),
        "slope_n_sloped": prof.get("slope_n_sloped", 0),
        "plateau_err_log10": prof.get("plateau_err_log10", np.nan),
        "thin_ratio_log_err": prof.get("thin_ratio_log_err", np.nan),
    }


def render_step_profiles(predictor, ibal_df, id_to_idx, context_ids, holdout_ids, step, order_dir):
    step_dir = os.path.join(order_dir, f"step_{step:02d}")
    ctx_dir = os.path.join(step_dir, "context")
    hold_dir = os.path.join(step_dir, "heldout")

    index_rows = []

    if context_ids:
        ctx_idx = [id_to_idx[s] for s in context_ids if s in id_to_idx]
        ctx_df = ibal_df.loc[ctx_idx].copy()
        ctx_profs = predict_profiles(predictor, ctx_df)
        for sid, prof in ctx_profs.items():
            plot_sample_profile(sid, prof, os.path.join(ctx_dir, f"{sid}.png"), is_context=True)
            index_rows.append(_shape_index_row(step, sid, "context", len(context_ids), prof))

    if holdout_ids:
        hold_idx = [id_to_idx[s] for s in holdout_ids if s in id_to_idx]
        hold_df = ibal_df.loc[hold_idx].copy()
        hold_profs = predict_profiles(predictor, hold_df)
        for sid, prof in hold_profs.items():
            plot_sample_profile(sid, prof, os.path.join(hold_dir, f"{sid}.png"), is_context=False)
            index_rows.append(_shape_index_row(step, sid, "heldout", len(context_ids), prof))

    if index_rows:
        os.makedirs(step_dir, exist_ok=True)
        pd.DataFrame(index_rows).to_csv(os.path.join(step_dir, "_index.csv"), index=False)

    n_ctx = len(context_ids) if context_ids else 0
    n_hold = len(holdout_ids) if holdout_ids else 0
    logger.info(
        f"  [profiles] step {step:>2}: {n_ctx} context + {n_hold} held-out profiles -> {step_dir}"
    )


def run_convergence_replay(
    predictor,
    ibal_df: pd.DataFrame,
    ordered_ids: list,
    n_draws: int = 20,
    k_context: int = 8,
    max_ctx_pool: int = 15,
    n_unc_samples: int = 30,
    learn_steps: int = 50,
    learn_lr: float = 1e-3,
    order_dir: str = None,
    plot_step_profiles: bool = True,
    profile_max_steps: int | None = 8,
) -> pd.DataFrame:
    """Adds ibalizumab samples one-by-one in `ordered_ids` order. At each
    step: encode context, predict holdout, record metrics + uncertainty +
    shape fidelity.

    Returns DataFrame with columns: step, sample_id, n_context, mae, rmse,
    mape, mae_log10, rmse_log10, std_log10, plus shape-fidelity columns.
    """
    id_to_idx = {str(row["ID"]): idx for idx, row in ibal_df.iterrows()}
    ordered_ids = [str(sid) for sid in ordered_ids if str(sid) in id_to_idx]
    if not ordered_ids:
        raise ValueError(
            "No `ordered_ids` matched an ID in `ibal_df` — check that the order CSV's "
            "sample IDs line up with ibal_df's ID column."
        )

    records = []
    null_metrics = {
        "mae": np.nan,
        "rmse": np.nan,
        "mape": np.nan,
        "mae_log10": np.nan,
        "rmse_log10": np.nan,
    }

    logger.info(f"Replaying {len(ordered_ids)} samples (plus 0-shot baseline)...")

    # Step 0: zero-shot baseline.
    reset_memory(predictor)
    all_holdout_idx = [id_to_idx[s] for s in ordered_ids]
    holdout_df_0 = ibal_df.loc[all_holdout_idx].copy()

    metrics_0 = null_metrics.copy()
    std_log10_0 = np.nan
    shape_agg_0 = dict(_NULL_SHAPE_AGG)
    if not has_nan_weights(predictor):
        try:
            drop_targets = [c for c in holdout_df_0.columns if c.startswith("Viscosity_")]
            query_0 = holdout_df_0.drop(columns=drop_targets, errors="ignore")
            results_0 = predictor.predict(query_0)
            metrics_0 = compute_metrics(results_0, holdout_df_0)
            _, unc_0 = predictor.predict_with_uncertainty(
                query_0, n_samples=n_unc_samples, k=k_context
            )
            std_log10_0 = float(np.mean(unc_0.get("std_log10", [np.nan])))
            profs_0 = predict_profiles(predictor, holdout_df_0)
            shape_agg_0 = _aggregate_shape(profs_0)
        except Exception as e:
            logger.warning(f"  Step 0: predict failed — {e}")

    records.append(
        {
            "step": 0,
            "sample_id": "None",
            "n_context": 0,
            **metrics_0,
            "std_log10": std_log10_0,
            **shape_agg_0,
        }
    )
    logger.info(
        f"  [ 0/{len(ordered_ids)}] 0-Shot Baseline | Holdout MAE={metrics_0['mae']:.3f} cP  "
        f"RMSE={metrics_0['rmse']:.3f} cP  RMSE(log10)={metrics_0['rmse_log10']:.4f}  "
        f"MAPE={metrics_0['mape']:.2f}%  std(log10)={std_log10_0:.4f}"
    )

    if plot_step_profiles and order_dir is not None:
        try:
            render_step_profiles(
                predictor,
                ibal_df,
                id_to_idx,
                context_ids=[],
                holdout_ids=ordered_ids,
                step=0,
                order_dir=order_dir,
            )
        except Exception as e:
            logger.warning(f"  Step 0: profile render failed — {e}")

    for step, sample_id in enumerate(ordered_ids, start=1):
        context_ids = ordered_ids[:step]
        holdout_ids = ordered_ids[step:]

        context_idx = [id_to_idx[s] for s in context_ids]
        context_df = ibal_df.loc[context_idx].copy()

        encode_context(
            predictor,
            context_df,
            learn_steps=learn_steps,
            learn_lr=learn_lr,
            n_draws=n_draws,
            k_context=k_context,
            max_ctx_pool=max_ctx_pool,
        )

        metrics = null_metrics.copy()
        std_log10 = np.nan

        if holdout_ids and not has_nan_weights(predictor):
            holdout_idx = [id_to_idx[s] for s in holdout_ids]
            holdout_df = ibal_df.loc[holdout_idx].copy()
            drop_targets_n = [c for c in holdout_df.columns if c.startswith("Viscosity_")]
            query_df = holdout_df.drop(columns=drop_targets_n, errors="ignore")
            try:
                results_df = predictor.predict(query_df)
                metrics = compute_metrics(results_df, holdout_df)
            except Exception as e:
                logger.warning(f"  Step {step}: predict failed — {e}")

            try:
                _, unc_stats = predictor.predict_with_uncertainty(
                    query_df, n_samples=n_unc_samples, k=k_context
                )
                std_log10 = float(np.mean(unc_stats.get("std_log10", [np.nan])))
            except Exception as e:
                logger.warning(f"  Step {step}: uncertainty failed — {e}")

        shape_agg = dict(_NULL_SHAPE_AGG)
        if holdout_ids and not has_nan_weights(predictor):
            try:
                holdout_idx = [id_to_idx[s] for s in holdout_ids]
                holdout_df = ibal_df.loc[holdout_idx].copy()
                profs = predict_profiles(predictor, holdout_df)
                shape_agg = _aggregate_shape(profs)
            except Exception as e:
                logger.warning(f"  Step {step}: shape agg failed — {e}")

        records.append(
            {
                "step": step,
                "sample_id": sample_id,
                "n_context": step,
                **metrics,
                "std_log10": std_log10,
                **shape_agg,
            }
        )
        logger.info(
            f"  [{step:>2}/{len(ordered_ids)}] Added {sample_id:>6} | Holdout MAE={metrics['mae']:.3f} cP  "
            f"RMSE={metrics['rmse']:.3f} cP  RMSE(log10)={metrics['rmse_log10']:.4f}  MAPE={metrics['mape']:.2f}%  "
            f"shapeRMSE={shape_agg['shape_rmse_log10']:.3f}  slopeMatch={shape_agg['slope_sign_agree']:.0%}"
        )

        if (
            plot_step_profiles
            and order_dir is not None
            and (profile_max_steps is None or step <= profile_max_steps)
        ):
            try:
                render_step_profiles(
                    predictor,
                    ibal_df,
                    id_to_idx,
                    context_ids=context_ids,
                    holdout_ids=holdout_ids,
                    step=step,
                    order_dir=order_dir,
                )
            except Exception as e:
                logger.warning(f"  Step {step}: profile render failed — {e}")

    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Driver (moved from cli/learning_curve.py; main() split into run() + main()
# so this eval is callable directly from scripts/run.py without argv round-tripping).
# --------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Replay a sample-addition order through ViscosityPredictorCNP and plot the learning curve."
    )
    ap.add_argument(
        "--model_dir",
        default=None,
        help="Trained checkpoint directory to evaluate. Defaults to the most recently produced checkpoint.",
    )
    ap.add_argument(
        "--ibal_csv",
        required=True,
        help=(
            "Held-out evaluation CSV (e.g. ibalizumab samples). No default; this repo no "
            "longer ships a curated split -- point this at a real held-out file."
        ),
    )
    ap.add_argument(
        "--order_csv",
        default=None,
        help=(
            "CSV with a Sample_ID column giving the addition order. Defaults to "
            "<model_dir>/eval_parity/context_selection.csv (wherever the parity eval just wrote it)."
        ),
    )
    ap.add_argument(
        "--pretrain_csv",
        default=None,
        help="Optional cross-protein context pool to encode as a zero-shot prior. Skipped if omitted.",
    )
    ap.add_argument(
        "--output_dir",
        default=None,
        help="Where to write eval results. Defaults to <model_dir>/benchmarks.",
    )
    ap.add_argument("--n_draws", type=int, default=20)
    ap.add_argument("--k_context", type=int, default=8)
    ap.add_argument("--max_ctx_pool", type=int, default=15)
    ap.add_argument("--n_unc_samples", type=int, default=30)
    ap.add_argument("--plot_step_profiles", action="store_true", default=True)
    ap.add_argument("--no-plot-step-profiles", dest="plot_step_profiles", action="store_false")
    ap.add_argument("--profile_max_steps", type=int, default=8, help="-1 for unlimited.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--also-random-baseline",
        dest="also_random_baseline",
        action="store_true",
        default=True,
        help="Also replay a shuffled order as a baseline (default on).",
    )
    ap.add_argument("--no-random-baseline", dest="also_random_baseline", action="store_false")
    return ap.parse_args(argv)


def _load_order_ids(order_csv) -> list:
    """Load the sample-addition order from `order_csv`. Accepts either:

    - a `Sample_ID` column (the original convergence-replay order format,
      e.g. `optimal_order_summary.csv`'s Step/Sample_ID/... columns), used
      in file order; or
    - an `ID` column (the format parity_eval.py's context-selection save
      produces, and what a plain filtered formulation-data CSV has), sorted
      by a `cnp_rank`/`rank` column first if present, otherwise used in
      file order.
    """
    order_df = paths.load_table(order_csv)
    if "Sample_ID" in order_df.columns:
        return order_df["Sample_ID"].tolist()
    if "ID" in order_df.columns:
        rank_col = (
            "cnp_rank"
            if "cnp_rank" in order_df.columns
            else ("rank" if "rank" in order_df.columns else None)
        )
        if rank_col:
            order_df = order_df.sort_values(rank_col)
        return order_df["ID"].tolist()
    raise KeyError(
        f"{order_csv} has neither a 'Sample_ID' nor an 'ID' column (found: {list(order_df.columns)}) "
        "-- can't determine the sample-addition order."
    )


def _init_clean_predictor(model_dir, pretrain_df, n_draws, k_context, max_ctx_pool):
    logger.info(f"Initializing clean model from: {model_dir}")
    pred = ViscosityPredictorCNP(model_dir)
    if pretrain_df is not None:
        logger.info(
            f"  Encoding {len(pretrain_df)} cross-protein context samples as zero-shot prior (encode-only, no weight updates)..."
        )
        encode_context(
            pred, pretrain_df, n_draws=n_draws, k_context=k_context, max_ctx_pool=max_ctx_pool
        )
        if has_nan_weights(pred):
            logger.error("NaN weights detected — reloading clean model.")
            pred = ViscosityPredictorCNP(model_dir)
        else:
            logger.info("  Cross-protein prior encoded successfully.")
    return pred


def _run_and_plot(
    model_dir,
    pretrain_df,
    n_draws,
    k_context,
    max_ctx_pool,
    n_unc_samples,
    plot_step_profiles,
    profile_max_steps,
    output_dir,
    ibal_df,
    ids,
    order_dir,
    csv_path,
    prefix,
    max_steps=None,
):
    predictor = _init_clean_predictor(model_dir, pretrain_df, n_draws, k_context, max_ctx_pool)
    results_df = run_convergence_replay(
        predictor,
        ibal_df,
        ids,
        n_draws=n_draws,
        k_context=k_context,
        max_ctx_pool=max_ctx_pool,
        n_unc_samples=n_unc_samples,
        order_dir=order_dir,
        plot_step_profiles=plot_step_profiles,
        profile_max_steps=profile_max_steps,
    )
    results_df.to_csv(csv_path, index=False)
    logger.info(f"  Metrics saved: {csv_path}")

    plot_kwargs = {} if max_steps is None else {"max_steps": max_steps}
    plot_convergence(results_df, output_dir, prefix=prefix, **plot_kwargs)
    plot_mape(results_df, output_dir, prefix=prefix, **plot_kwargs)
    plot_log_convergence(results_df, output_dir, prefix=prefix, **plot_kwargs)
    plot_shape_convergence(results_df, output_dir, prefix=prefix, **plot_kwargs)
    return results_df


def _log_summary(name, df):
    valid = df.dropna(subset=["mae", "rmse", "mape"])
    if valid.empty:
        return
    best_rmse = valid.loc[valid["rmse"].idxmin()]
    best_mae = valid.loc[valid["mae"].idxmin()]
    best_mape = valid.loc[valid["mape"].idxmin()]
    best_rmse_log = valid.loc[valid["rmse_log10"].idxmin()]
    logger.info(
        f"\n{'='*65}\n"
        f"  {name} Best RMSE      : {best_rmse['rmse']:.3f} cP  @ n={best_rmse['n_context']}  ({best_rmse['sample_id']} added)\n"
        f"  {name} Best MAE       : {best_mae['mae']:.3f} cP  @ n={best_mae['n_context']}  ({best_mae['sample_id']} added)\n"
        f"  {name} Best MAPE      : {best_mape['mape']:.2f}%  @ n={best_mape['n_context']}  ({best_mape['sample_id']} added)\n"
        f"  {name} Best RMSE(log) : {best_rmse_log['rmse_log10']:.4f}  @ n={best_rmse_log['n_context']}  ({best_rmse_log['sample_id']} added)\n"
        f"{'='*65}"
    )


def run(
    ibal_csv,
    model_dir=None,
    order_csv=None,
    pretrain_csv=None,
    output_dir=None,
    n_draws=20,
    k_context=8,
    max_ctx_pool=15,
    n_unc_samples=30,
    plot_step_profiles=True,
    profile_max_steps=8,
    seed=42,
    also_random_baseline=True,
):
    if model_dir is None:
        model_dir = paths.latest_checkpoint_dir(constants.CHECKPOINTS_DIR)
    if output_dir is None:
        output_dir = os.path.join(model_dir, "benchmarks")
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if order_csv is None:
        order_csv = os.path.join(model_dir, "eval_parity", "context_selection.csv")

    pretrain_df = None
    if pretrain_csv and os.path.exists(pretrain_csv):
        logger.info(f"Loading pre-training context pool: {pretrain_csv}")
        pretrain_df = prepare_df(paths.load_table(pretrain_csv), drop_bad_rows=True)
        logger.info(f"  {len(pretrain_df)} valid pre-training samples.")

    logger.info(f"Loading ibalizumab data: {ibal_csv}")
    ibal_df = prepare_df(paths.load_table(ibal_csv, index_col=False), drop_bad_rows=True)
    logger.info(f"  {len(ibal_df)} valid ibalizumab samples.")

    logger.info(f"Loading optimal order: {order_csv}")
    optimal_ids = _load_order_ids(order_csv)

    profile_max = None if profile_max_steps < 0 else profile_max_steps
    common = dict(
        model_dir=model_dir,
        pretrain_df=pretrain_df,
        n_draws=n_draws,
        k_context=k_context,
        max_ctx_pool=max_ctx_pool,
        n_unc_samples=n_unc_samples,
        plot_step_profiles=plot_step_profiles,
        profile_max_steps=profile_max,
        output_dir=output_dir,
    )

    logger.info("\n" + "=" * 55)
    logger.info("RUNNING EVALUATION: OPTIMAL ORDER")
    logger.info("=" * 55)
    results_opt_df = _run_and_plot(
        **common,
        ibal_df=ibal_df,
        ids=optimal_ids,
        order_dir=os.path.join(output_dir, "profiles_optimal"),
        csv_path=os.path.join(output_dir, "optimal_convergence_metrics.csv"),
        prefix="optimal_",
    )

    results_rand_df = None
    if also_random_baseline:
        random_ids = optimal_ids.copy()
        np.random.shuffle(random_ids)
        logger.info("\n" + "=" * 55)
        logger.info("RUNNING EVALUATION: RANDOM ORDER")
        logger.info("=" * 55)
        results_rand_df = _run_and_plot(
            **common,
            ibal_df=ibal_df,
            ids=random_ids,
            order_dir=os.path.join(output_dir, "profiles_random"),
            csv_path=os.path.join(output_dir, "random_convergence_metrics.csv"),
            prefix="random_",
        )

    _log_summary("OPTIMAL", results_opt_df)
    if results_rand_df is not None:
        _log_summary("RANDOM", results_rand_df)

    logger.info("Done.")
    return results_opt_df, results_rand_df


def main(argv=None):
    args = parse_args(argv)
    if args.model_dir is None:
        args.model_dir = paths.latest_checkpoint_dir(constants.CHECKPOINTS_DIR)
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, "benchmarks")
    configure_logging(log_dir=args.output_dir)
    return run(**vars(args))


if __name__ == "__main__":
    main()
