"""
helpers.py
==========
Shared plot-annotation helpers, moved from learning_curve_ibal.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from visqai.eval.plotting.style import C_GREEN, C_ORANGE, C_MUTED, C_WHITE

PLOT_MAX_STEPS = 10


def annotate_best(ax, x_arr, y_arr, fmt, color, offset_frac=0.06):
    b = np.argmin(y_arr)
    ax.scatter([x_arr[b]], [y_arr[b]], color=C_GREEN, s=100, zorder=7, edgecolors=C_WHITE, linewidths=1.4)
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
    ax.set_xticks(sx)
    ax.set_xticklabels(
        [f"{n}\n({sid})" if sid != "None" else "0\n(0-shot)" for n, sid in zip(sx, labels)],
        fontsize=11,
        color=C_MUTED,
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
