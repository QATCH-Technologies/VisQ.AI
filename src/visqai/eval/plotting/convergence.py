"""
convergence.py
===============
Convergence-replay plots (linear cP, MAPE, log10 with uncertainty ribbon,
shape-fidelity, per-sample shear profile), moved from
ml/cnp_mk2/learning_curve_ibal.py onto the unified house style
(visqai.eval.plotting.style) and shared shear constants
(visqai.eval.constants).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from visqai.eval.constants import SHEAR_COLS, SHORT_SHEAR_LABELS, SHEAR_RATES
from visqai.eval.metrics import calc_metrics
from visqai.eval.shape_metrics import compute_shape_metrics
from visqai.eval.plotting.style import (
    mpl,
    apply_style,
    style_axis,
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
from visqai.eval.plotting.helpers import (
    PLOT_MAX_STEPS,
    annotate_best,
    find_convergence_step,
    annotate_convergence,
    shared_x_labels,
    prep_plot_data,
)

logger = logging.getLogger(__name__)


def plot_convergence(df: pd.DataFrame, save_dir: str, prefix: str = ""):
    """MAE + RMSE (linear cP) convergence — dual-axis combined and side-by-side panels."""
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

    _, sx, labels, vals = prep_plot_data(df, ["mae", "rmse"])
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
    ax2.plot(sx, srmse, color=C_BRIGHT_BLUE, lw=2.4, zorder=4, solid_capstyle="round", ls=(0, (6, 2)))
    ax2.scatter(sx, srmse, color=C_BRIGHT_BLUE, s=38, zorder=5, edgecolors=C_WHITE, linewidths=1.1, marker="D")
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
        Line2D([0], [0], color=C_DEEP_BLUE, lw=2.4, marker="o", markersize=8, markerfacecolor=C_DEEP_BLUE, markeredgecolor=C_WHITE, label="MAE (cP)"),
        Line2D([0], [0], color=C_BRIGHT_BLUE, lw=2.4, marker="D", markersize=7, markerfacecolor=C_BRIGHT_BLUE, markeredgecolor=C_WHITE, ls=(0, (6, 2)), label="RMSE (cP)"),
        Line2D([0], [0], color=C_GREEN, lw=0, marker="o", markersize=10, markerfacecolor=C_GREEN, markeredgecolor=C_WHITE, label="Best value"),
        Line2D([0], [0], color=C_ORANGE, lw=1.5, ls="--", label="Plateau onset"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=12, framealpha=0.95, borderpad=0.9, edgecolor=C_BORDER)
    ax1.set_title("Ibalizumab · Learning Curve", fontsize=16, fontweight="bold", pad=14, color=C_TEXT, loc="left")

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
        ax.set_title(f"{mlabel} vs. Context Size", fontsize=15, fontweight="bold", pad=11, color=C_TEXT, loc="left")
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


def plot_mape(df: pd.DataFrame, save_dir: str, prefix: str = ""):
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

    _, sx, labels, vals = prep_plot_data(df, ["mape"])
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
    ax.set_title("MAPE vs. Context Size", fontsize=15, fontweight="bold", pad=11, color=C_TEXT, loc="left")
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


def plot_log_convergence(df: pd.DataFrame, save_dir: str, prefix: str = ""):
    """Convergence curve in log10-RMSE space with a ±1σ shaded ribbon derived
    from per-step std_log10 (context-subsampling uncertainty)."""
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

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
    clr_unc = "#e8b4b8"

    fig, ax1 = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    style_axis(ax1, ticker)

    if sstd_log is not None and not np.all(np.isnan(sstd_log)):
        ribbon_lo = srmse_log - sstd_log
        ribbon_hi = srmse_log + sstd_log
        ax1.fill_between(sx, ribbon_lo, ribbon_hi, alpha=0.22, color=clr_unc, zorder=1, label="±1σ context uncertainty")

    ax1.plot(sx, srmse_log, color=clr_rmse, lw=2.5, zorder=4, solid_capstyle="round")
    ax1.scatter(sx, srmse_log, color=clr_rmse, s=50, zorder=5, edgecolors=C_WHITE, linewidths=1.2, marker="s")
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
        ax2.plot(sx, smae_log, color=clr_mae, lw=2.2, zorder=3, solid_capstyle="round", ls=(0, (6, 2)))
        ax2.scatter(sx, smae_log, color=clr_mae, s=38, zorder=5, edgecolors=C_WHITE, linewidths=1.1, marker="D")
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
        sx[-1] + 0.1, benchmark_loo + yrange * 0.03, f"LOO baseline ({benchmark_loo:.3f})",
        fontsize=10, color=C_ORANGE, va="bottom", ha="right", style="italic",
    )

    legend_elements = [
        Line2D([0], [0], color=clr_rmse, lw=2.5, marker="s", markersize=8, markerfacecolor=clr_rmse, markeredgecolor=C_WHITE, label="RMSE (log₁₀)"),
        Line2D([0], [0], color=clr_mae, lw=2.2, marker="D", markersize=7, markerfacecolor=clr_mae, markeredgecolor=C_WHITE, ls=(0, (6, 2)), label="MAE (log₁₀)"),
        plt.Rectangle((0, 0), 1, 1, fc=clr_unc, alpha=0.4, label="±1σ context uncertainty"),
        Line2D([0], [0], color=C_GREEN, lw=0, marker="o", markersize=10, markerfacecolor=C_GREEN, markeredgecolor=C_WHITE, label="Best value"),
        Line2D([0], [0], color=C_ORANGE, lw=1.5, ls="--", label="Plateau onset"),
        Line2D([0], [0], color=C_ORANGE, lw=1.2, ls=":", alpha=0.8, label="LOO baseline"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=11, framealpha=0.95, borderpad=0.9, edgecolor=C_BORDER)

    ax1.set_title("Ibalizumab · Log₁₀ Learning Curve  (model-native metric)", fontsize=16, fontweight="bold", pad=14, color=C_TEXT, loc="left")
    ax1.spines["top"].set_visible(True)
    ax1.spines["top"].set_edgecolor(clr_rmse)
    ax1.spines["top"].set_linewidth(2.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    log_path = os.path.join(save_dir, f"{prefix}convergence_log10.png")
    fig.savefig(log_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    logger.info(f"  Saved log10 convergence plot: {log_path}")
    return log_path


def plot_shape_convergence(df: pd.DataFrame, save_dir: str, prefix: str = ""):
    """Does the predicted curve MORPHOLOGY improve as context grows? Plots
    slope-sign agreement (↑ better) and level-invariant shape RMSE (↓ better)."""
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

    need = ["slope_sign_agree", "shape_rmse_log10"]
    if not all(c in df.columns for c in need):
        logger.warning("Shape columns absent — skipping shape convergence plot.")
        return None
    valid = ~df[need].isna().any(axis=1)
    valid.iloc[-1] = False
    plot_df = df[valid].head(PLOT_MAX_STEPS + 1)
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
    ax2.scatter(sx, s_shape, color=clr_shape, s=40, zorder=5, marker="D", edgecolors=C_WHITE, linewidths=1.1)
    ax2.set_ylabel("Shape RMSE  (log₁₀, level-invariant)  ↓ better", fontsize=14, labelpad=10, color=clr_shape)
    ax2.tick_params(axis="y", labelcolor=clr_shape, labelsize=12)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor(clr_shape)
    ax2.spines["right"].set_linewidth(1.2)
    ax2.spines["top"].set_visible(False)

    ax1.set_xlabel("Samples added to context  (n)", fontsize=14, labelpad=10, color=C_TEXT)
    shared_x_labels(ax1, sx, labels)

    legend_elements = [
        Line2D([0], [0], color=clr_agree, lw=2.5, marker="o", markersize=8, markerfacecolor=clr_agree, markeredgecolor=C_WHITE, label="Slope-sign agreement (↑)"),
        Line2D([0], [0], color=clr_shape, lw=2.4, marker="D", markersize=7, markerfacecolor=clr_shape, markeredgecolor=C_WHITE, ls=(0, (6, 2)), label="Shape RMSE log₁₀ (↓)"),
    ]
    ax1.legend(handles=legend_elements, loc="center right", fontsize=12, framealpha=0.95, borderpad=0.9, edgecolor=C_BORDER)
    ax1.set_title("Ibalizumab · Shape-Fidelity Convergence  (morphology, not level)", fontsize=16, fontweight="bold", pad=14, color=C_TEXT, loc="left")
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
    """Predict the full 5-point shear-thinning profile for each sample.
    Returns a dict keyed by sample ID with actual/pred arrays, conc/pH, and
    shape-fidelity metrics. The model's memory is NOT modified here."""
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
        actual = np.array([float(srow[c]) if c in srow and pd.notnull(srow[c]) else np.nan for c in SHEAR_COLS])
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
            rates[ma], actual[ma], color=C_TEXT, lw=2.6, marker="o", markersize=8,
            markerfacecolor=C_TEXT, markeredgecolor=C_WHITE, markeredgewidth=1.3,
            zorder=5, label="Actual", solid_capstyle="round",
        )
    mp = ~np.isnan(pred) & (pred > 0)
    if mp.any():
        ax.plot(
            rates[mp], pred[mp], color=frame_clr, lw=2.4, marker="D", markersize=7,
            markerfacecolor=frame_clr, markeredgecolor=C_WHITE, markeredgewidth=1.2,
            ls=(0, (6, 2)), zorder=4, label="Predicted", solid_capstyle="round",
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

    ax.set_title(f"{role}  ·  Sample {sid}", fontsize=14, fontweight="bold", pad=30, color=frame_clr, loc="left")
    ax.text(0.0, 1.075, subtitle, transform=ax.transAxes, fontsize=10.5, color=C_MUTED, va="bottom")
    if shape_txt:
        ax.text(0.0, 1.038, shape_txt, transform=ax.transAxes, fontsize=10, color=frame_clr, va="bottom", style="italic")
    if shape_txt2:
        shape_clr = C_RED_SOFT if slope_agree < 0.5 else (C_ORANGE if slope_agree < 0.75 else C_GREEN)
        ax.text(0.0, 1.002, shape_txt2, transform=ax.transAxes, fontsize=9.5, color=shape_clr, va="bottom", fontweight="bold")

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
