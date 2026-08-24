"""
zero_shot_bench.py
===================
Plots for visqai.cli.zero_shot_bench: a standalone, single-model true
zero-shot benchmark (no context, no learning-curve replay) on
data/processed/Zero-shot testdata.csv -- unusual/novel proteins entirely
absent from training, scored on Viscosity_1000 only (the only shear rate
this dataset provides).

Two plots, one per experiment the CLI script runs:
  1. plot_parity_bin   -- actual-vs-predicted parity scatter, log-log.
  2. plot_metrics_bars -- bar chart of the standard MAE/RMSE/MAPE (linear +
     log10) metric set for this single model/run.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from visqai.eval.plotting.style import (
    mpl,
    apply_style,
    style_axis,
    C_DEEP_BLUE,
    C_TEXT,
    C_MUTED,
    C_BORDER,
    C_WHITE,
)

CLR_MODEL = C_DEEP_BLUE


def plot_parity_bin(
    per_sample: pd.DataFrame, save_dir: str, prefix: str = "", threshold: float = 30.0
) -> str:
    """Actual-vs-predicted parity scatter, log-log. Expects columns
    actual/pred."""
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

    actual = per_sample["actual"].values
    pred = per_sample["pred"].values

    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    style_axis(ax, ticker)

    lo = max(min(actual.min(), pred.min()) * 0.7, 1e-2)
    hi = max(actual.max(), pred.max()) * 1.4

    ax.plot([lo, hi], [lo, hi], color=C_MUTED, lw=3.6, ls="--", alpha=0.7, zorder=1, label="Perfect prediction")

    ax.scatter(actual, pred, color=CLR_MODEL, s=52, alpha=0.85, edgecolors=C_WHITE,
               linewidths=0.8, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xlabel(r"Actual Viscosity @ 1000 s$^{-1}$  (cP)", fontsize=21, labelpad=10, color=C_TEXT)
    ax.set_ylabel(r"Predicted Viscosity @ 1000 s$^{-1}$  (cP)", fontsize=21, labelpad=10, color=C_TEXT)
    ax.set_title(
        "Zero-shot predictions for PFDG38 set",
        fontsize=20, fontweight="bold", pad=14, color=C_TEXT, loc="left",
    )
    ax.legend(loc="upper left", fontsize=19, framealpha=0.95, borderpad=0.9, edgecolor=C_BORDER)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}zero_shot_parity_bins.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    return path


def plot_metrics_bars(summary: dict, save_dir: str, prefix: str = "") -> str:
    """Bar chart: MAE/RMSE/MAPE (linear, left panel) and MAE/RMSE (log10,
    right panel) for this single model/run. `summary` needs keys
    mae/rmse/mape/mae_log10/rmse_log10."""
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    os.makedirs(save_dir, exist_ok=True)

    fig, (axLin, axLog) = plt.subplots(1, 2, figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(C_WHITE)

    lin_metrics = [("mae", "MAE (cP)"), ("rmse", "RMSE (cP)"), ("mape", "MAPE (%)")]
    log_metrics = [("mae_log10", "MAE (log₁₀)"), ("rmse_log10", "RMSE (log₁₀)")]

    for ax, metrics, title in ((axLin, lin_metrics, "Linear-Space Metrics"), (axLog, log_metrics, "Log-Space Metrics")):
        style_axis(ax, ticker)
        x = np.arange(len(metrics))
        vals = [summary[k] for k, _ in metrics]
        ax.bar(x, vals, width=0.5, color=CLR_MODEL, edgecolor=C_WHITE, linewidth=0.6, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=10.5, color=CLR_MODEL)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=12, color=C_TEXT)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10, color=C_TEXT, loc="left")

    fig.suptitle("Zero-Shot Benchmark · Standard Metric Set", fontsize=16, fontweight="bold", y=1.02, color=C_TEXT)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}zero_shot_metrics.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    return path
