"""
parity.py
=========
Parity and per-sample profile plots + the long-form results builder, moved
from ml/cnp_mk2/ibal_parity_test.py. Uses the unified house style
(visqai.eval.plotting.style) instead of its own divergent _mpl/_apply_style.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from visqai.eval.constants import SHEAR_COLS, SHEAR_LABELS, SHEAR_COLORS, SHEAR_RATES, N_SHEARS
from visqai.eval.metrics import calc_metrics
from visqai.eval.plotting.style import mpl, apply_style, C_DEEP_BLUE, C_ACCENT, C_TEXT, C_MUTED, C_BORDER, C_WHITE, C_CONTEXT

logger = logging.getLogger(__name__)


def build_long(pred_df, is_context):
    rows = []
    for _, row in pred_df.iterrows():
        for sc in SHEAR_COLS:
            act = row.get(sc, np.nan)
            prd = row.get(f"Pred_{sc}", np.nan)
            valid = pd.notna(act) and pd.notna(prd) and act > 0 and prd > 0
            rows.append(
                {
                    "ID": row["ID"],
                    "shear_col": sc,
                    "shear_label": SHEAR_LABELS[sc],
                    "actual_cP": act,
                    "pred_cP": prd,
                    "is_context": is_context,
                    "Protein_conc": row.get("Protein_conc", np.nan),
                    "Buffer_pH": row.get("Buffer_pH", np.nan),
                    "log10_error": (np.log10(prd) - np.log10(act)) if valid else np.nan,
                    "fold_error": (prd / act) if valid else np.nan,
                }
            )
    return rows


def make_parity_plot(long_df, shear_subset, title, out_path, single_shear=False, context_ids=None):
    plt, ticker, Line2D = mpl()
    apply_style(plt)

    sub = long_df[long_df["shear_col"].isin(shear_subset)].copy()
    sub = sub[(sub["actual_cP"] > 0) & (sub["pred_cP"] > 0)].dropna(subset=["actual_cP", "pred_cP"])
    if sub.empty:
        logger.warning(f"No valid data for {out_path} — skipping.")
        return

    m = calc_metrics(sub["actual_cP"].values, sub["pred_cP"].values)
    all_vals = np.concatenate([sub["actual_cP"].values, sub["pred_cP"].values])
    all_vals = all_vals[all_vals > 0]
    log_min, log_max = np.log10(all_vals.min()), np.log10(all_vals.max())
    pad = (log_max - log_min) * 0.04
    lo, hi = 10 ** (log_min - pad), 10 ** (log_max + pad)

    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=160)
    ax.plot(np.linspace(lo, hi, 400), np.linspace(lo, hi, 400), color=C_DEEP_BLUE, lw=1.8, ls="--", zorder=3)

    ctx_set = set(context_ids) if context_ids is not None else set()
    for sc in shear_subset:
        mask = sub["shear_col"] == sc
        if not mask.any():
            continue
        held = mask & ~sub["ID"].isin(ctx_set)
        if held.any():
            ax.scatter(
                sub.loc[held, "actual_cP"], sub.loc[held, "pred_cP"], color=SHEAR_COLORS[sc],
                s=62, zorder=5, alpha=0.88, edgecolors=C_WHITE, linewidths=0.9,
            )
        ctx = mask & sub["ID"].isin(ctx_set)
        if ctx.any():
            ax.scatter(
                sub.loc[ctx, "actual_cP"], sub.loc[ctx, "pred_cP"], color=C_CONTEXT,
                s=80, zorder=6, alpha=0.92, edgecolors=C_WHITE, linewidths=0.9, marker="D",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, which="major", zorder=0, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Measured Viscosity (cP)", fontsize=15, labelpad=12, color=C_TEXT)
    ax.set_ylabel("Predicted Viscosity (cP)", fontsize=15, labelpad=12, color=C_TEXT)

    metrics_text = (
        f"MAE   {m['mae']:.2f} cP\nRMSE  {m['rmse']:.2f} cP\n"
        f"R²    {m['r2']:.3f}\n<=2x   {m['within_2x']:.0f}%   (n={m['n']})"
    )
    ax.text(
        0.04, 0.97, metrics_text, transform=ax.transAxes, fontsize=12, va="top", ha="left", color=C_TEXT,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.55", facecolor=C_WHITE, edgecolor=C_BORDER, linewidth=0.8, alpha=0.93),
    )

    parity_handle = Line2D([0], [0], color=C_DEEP_BLUE, lw=1.8, ls="--")
    context_handle = Line2D([0], [0], marker="D", color="w", markerfacecolor=C_CONTEXT, markersize=8)
    if single_shear:
        handles = [parity_handle] + ([context_handle] if ctx_set else [])
        labels = ["Perfect parity"] + (["Context"] if ctx_set else [])
    else:
        shear_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SHEAR_COLORS[sc], markersize=8)
            for sc in shear_subset
        ]
        handles = [parity_handle] + shear_handles + ([context_handle] if ctx_set else [])
        labels = ["Perfect parity"] + [SHEAR_LABELS[sc] for sc in shear_subset] + (["Context"] if ctx_set else [])
    ax.legend(
        handles=handles, labels=labels, loc="lower right", fontsize=11, framealpha=0.92,
        edgecolor=C_BORDER, borderpad=0.9, handlelength=2.0,
    )

    ax.set_title(title, fontsize=16, pad=14, color=C_TEXT, loc="left", fontweight="semibold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    logger.info(f"  saved: {out_path}")


def make_profile_plot(results_df, sample_id, out_path):
    plt, ticker, Line2D = mpl()
    apply_style(plt)
    row = results_df[results_df["ID"] == str(sample_id)]
    if row.empty:
        logger.warning(f"Profile: ID '{sample_id}' not found — skipping.")
        return
    row = row.iloc[0]
    measured = [row.get(sc, np.nan) for sc in SHEAR_COLS]
    predicted = [row.get(f"Pred_{sc}", np.nan) for sc in SHEAR_COLS]

    conc, ph = row.get("Protein_conc", "?"), row.get("Buffer_pH", "?")
    parts = []
    for key in ("Salt_type", "Stabilizer_type", "Surfactant_type"):
        v = row.get(key, "none")
        if str(v).lower() not in ("none", "nan", ""):
            parts.append(str(v))
    subtitle = f"{conc} mg/mL  |  pH {ph}  |  {', '.join(parts) if parts else 'no excipients'}"

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    valid = [i for i in range(N_SHEARS) if pd.notna(measured[i]) and pd.notna(predicted[i])]
    if valid:
        xs = [SHEAR_RATES[i] for i in valid]
        ax.fill_between(
            xs, [measured[i] for i in valid], [predicted[i] for i in valid],
            color=C_DEEP_BLUE, alpha=0.08, linewidth=0, zorder=1,
        )
    ax.plot(
        SHEAR_RATES, measured, color=C_DEEP_BLUE, lw=2.2, marker="o", markersize=7,
        markeredgecolor=C_WHITE, markeredgewidth=0.8, zorder=4, label="Measured",
    )
    ax.plot(
        SHEAR_RATES, predicted, color=C_ACCENT, lw=2.2, ls="--", marker="s", markersize=7,
        markeredgecolor=C_WHITE, markeredgewidth=0.8, zorder=4, label="Predicted",
    )
    ax.set_xscale("log")
    yvals = [v for v in measured + predicted if pd.notna(v)]
    ax.set_ylim(0, max(yvals) * 1.15 if yvals else 1)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(True, which="major", zorder=0, linewidth=0.6)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Shear Rate (s⁻¹)", fontsize=14, labelpad=10, color=C_TEXT)
    ax.set_ylabel("Viscosity (cP)", fontsize=14, labelpad=10, color=C_TEXT)
    ax.set_title(
        f"Ibalizumab — Viscosity Profile  (ID {sample_id})\n{subtitle}",
        fontsize=14, pad=12, color=C_TEXT, loc="left", fontweight="semibold",
    )
    ax.legend(loc="upper right", fontsize=12, framealpha=0.92, edgecolor=C_BORDER, borderpad=0.8, handlelength=2.0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    logger.info(f"  saved: {out_path}")
