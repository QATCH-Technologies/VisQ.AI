"""
ibalizumab_parity_test.py
==========================
Use the top-10 representative Ibalizumab samples (output of
find_representative_ibalizumab.py) as CNP context and evaluate
predictions against the held-out 24 samples.

Edit the CONFIG block, then run:
    python ibalizumab_parity_test.py

Or override any setting via CLI:
    python ibalizumab_parity_test.py --model_dir path/to/model ...

Outputs
-------
  <out_dir>/ibalizumab_parity_results.csv   — full pred/actual table
  <out_dir>/parity_ibal_all_shears.png      — log parity, all 5 shear rates
  <out_dir>/parity_ibal_1000.png            — log parity, 1 000 s-1 only
"""

import argparse
import logging
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── suppress file-handler noise from inference_cnp on import ────────────────
_orig_basicConfig = logging.basicConfig
logging.basicConfig = lambda **kw: None  # type: ignore
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_cnp import ViscosityPredictorCNP  # noqa: E402

logging.basicConfig = _orig_basicConfig

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  <- edit here, or pass CLI flags to override
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = r"models/experiments/o_net_v3_10_ibal_no_aug"
DATA_CSV = r"data/raw/formulation_data_03042026.csv"
TOP10_CSV = r"ibalizumab_top10.csv"  # output of find_representative_ibalizumab.py
OUT_DIR = r"models/experiments/o_net_v3_10_ibal_no_aug/benchmarks"
PROTEIN_KEY = "Ibalizumab"
PLOT_ENABLED = True
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("IbalParityTest")

# ── Shear rate definitions ────────────────────────────────────────────────────
SHEAR_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]
SHEAR_LABELS = {
    "Viscosity_100": "100 s\u207b\u00b9",
    "Viscosity_1000": "1 000 s\u207b\u00b9",
    "Viscosity_10000": "10 000 s\u207b\u00b9",
    "Viscosity_100000": "100 000 s\u207b\u00b9",
    "Viscosity_15000000": "15 000 000 s\u207b\u00b9",
}

# ── Brand palette (mirrors parity_plot.py) ────────────────────────────────────
C_DEEP_BLUE = "#2596be"
C_TEXT = "#24292f"
C_MUTED = "#6b7280"
C_BORDER = "#d1d5db"
C_BORDER_LT = "#e5e7eb"
C_BG_LIGHTEST = "#f9fafb"
C_WHITE = "#ffffff"

# One distinct colour per shear rate (drawn from project's PROTEIN_COLOURS pool)
SHEAR_COLORS = {
    "Viscosity_100": "#2596be",
    "Viscosity_1000": "#4caf50",
    "Viscosity_10000": "#ff9800",
    "Viscosity_100000": "#f44336",
    "Viscosity_15000000": "#9c27b0",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        if col != "ID":
            df[col] = df[col].astype(float)
    if "ID" in df.columns:
        df["ID"] = df["ID"].astype(str)
    return df


def calc_metrics(true, pred):
    """MAE, MAPE, RMSE, R2 in linear cP; log10-MAE, log10-RMSE, bias, within-2x."""
    t = np.asarray(true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mae = float(np.mean(np.abs(t - p)))
    mape = float(np.mean(np.abs((t - p) / np.clip(t, 1e-6, None))) * 100)
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))

    log_t = np.log10(np.clip(t, 1e-9, None))
    log_p = np.log10(np.clip(p, 1e-9, None))
    log_err = log_p - log_t
    log_mae = float(np.mean(np.abs(log_err)))
    log_rmse = float(np.sqrt(np.mean(log_err**2)))
    log_bias = float(np.mean(log_err))
    w2x = float(np.mean(np.abs(log_err) < np.log10(2)) * 100)

    return dict(
        mae=mae,
        mape=mape,
        rmse=rmse,
        r2=r2,
        log_mae=log_mae,
        log_rmse=log_rmse,
        log_bias=log_bias,
        within_2x=w2x,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Parity plot
# ──────────────────────────────────────────────────────────────────────────────


def _apply_style():
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
            "font.size": 20,
        }
    )


def make_parity_plot(long_df, shear_subset, title, out_path, single_shear=False):
    """
    Log-log parity plot matching the project reference style.

    Parameters
    ----------
    long_df      : DataFrame with [ID, shear_col, actual_cP, pred_cP]
    shear_subset : list of Viscosity_* cols to include
    title        : axes title string
    out_path     : PNG save path
    single_shear : if True, suppress the shear-rate colour legend
    """
    _apply_style()

    sub = long_df[long_df["shear_col"].isin(shear_subset)].copy()
    sub = sub[(sub["actual_cP"] > 0) & (sub["pred_cP"] > 0)].dropna(
        subset=["actual_cP", "pred_cP"]
    )

    if sub.empty:
        logger.warning(f"No valid data for {out_path} — skipping.")
        return

    m = calc_metrics(sub["actual_cP"].values, sub["pred_cP"].values)

    # ── Axis limits: tight to data, 4 % log-padding ──────────────────────────
    all_vals = np.concatenate([sub["actual_cP"].values, sub["pred_cP"].values])
    all_vals = all_vals[all_vals > 0]
    log_min = np.log10(all_vals.min())
    log_max = np.log10(all_vals.max())
    log_pad = (log_max - log_min) * 0.04
    lo = 10 ** (log_min - log_pad)
    hi = 10 ** (log_max + log_pad)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=150)
    fig.patch.set_facecolor(C_WHITE)
    ax.set_facecolor(C_BG_LIGHTEST)

    # Perfect parity line
    parity_x = np.logspace(np.log10(lo), np.log10(hi), 300)
    ax.plot(parity_x, parity_x, color=C_DEEP_BLUE, lw=1.6, ls="--", zorder=2)

    # Scatter — colour-coded by shear rate
    for sc in shear_subset:
        mask = sub["shear_col"] == sc
        if not mask.any():
            continue
        ax.scatter(
            sub.loc[mask, "actual_cP"],
            sub.loc[mask, "pred_cP"],
            color=SHEAR_COLORS[sc],
            s=52,
            zorder=5,
            alpha=0.85,
            edgecolors=C_WHITE,
            linewidths=0.7,
        )

    # ── Linear axes ───────────────────────────────────────────────────────────
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.tick_params(which="both", length=4, width=0.8)
    ax.grid(True, which="major", zorder=0)

    for spine in ax.spines.values():
        spine.set_edgecolor(C_BORDER)
        spine.set_linewidth(0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax.set_xlabel("Measured Viscosity (cP)", fontsize=16, labelpad=12, color=C_TEXT)
    ax.set_ylabel("Predicted Viscosity (cP)", fontsize=16, labelpad=12, color=C_TEXT)

    # ── Metrics box — top-left, monospace ────────────────────────────────────
    metrics_text = f"MAE   {m['mae']:.2f} cP\n" f"RMSE  {m['rmse']:.2f} cP"
    ax.text(
        0.03,
        0.97,
        metrics_text,
        transform=ax.transAxes,
        fontsize=14,
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

    # ── Legend ────────────────────────────────────────────────────────────────
    parity_handle = Line2D(
        [0], [0], color=C_DEEP_BLUE, lw=1.6, ls="--", label="Perfect parity"
    )
    if single_shear:
        ax.legend(
            handles=[parity_handle],
            labels=["Perfect parity"],
            loc="lower right",
            fontsize=13,
            edgecolor=C_BORDER,
            borderpad=0.8,
            handlelength=1.8,
        )
    else:
        shear_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=SHEAR_COLORS[sc],
                markersize=8,
                label=SHEAR_LABELS[sc],
            )
            for sc in shear_subset
        ]
        ax.legend(
            handles=[parity_handle] + shear_handles,
            labels=["Perfect parity"] + [SHEAR_LABELS[sc] for sc in shear_subset],
            loc="lower right",
            fontsize=11,
            edgecolor=C_BORDER,
            borderpad=0.8,
            handlelength=1.8,
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=17, pad=14, color=C_TEXT, loc="left")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Ibalizumab parity test: top-10 context vs held-out 24."
    )
    parser.add_argument("--model_dir", default=MODEL_DIR)
    parser.add_argument("--data", default=DATA_CSV)
    parser.add_argument("--top10_csv", default=TOP10_CSV)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--protein_key", default=PROTEIN_KEY)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading formulation data: {args.data}")
    df = prepare_df(pd.read_csv(args.data))
    iba_df = (
        df[df["Protein_type"].str.lower() == args.protein_key.lower()]
        .copy()
        .reset_index(drop=True)
    )
    logger.info(f"  Total {args.protein_key} samples: {len(iba_df)}")

    # ── Load top-10 IDs ───────────────────────────────────────────────────────
    logger.info(f"Loading top-10 selection: {args.top10_csv}")
    top10_meta = prepare_df(pd.read_csv(args.top10_csv))
    rank_col = "cnp_rank" if "cnp_rank" in top10_meta.columns else "rank"
    top10_ids = top10_meta.sort_values(rank_col)["ID"].head(10).tolist()
    logger.info(f"  Context IDs: {top10_ids}")

    missing = [i for i in top10_ids if i not in set(iba_df["ID"])]
    if missing:
        sys.exit(f"ERROR: IDs not found in ibalizumab rows: {missing}")

    # ── Split context / held-out ──────────────────────────────────────────────
    context_df = iba_df[iba_df["ID"].isin(top10_ids)].copy()
    held_out_df = iba_df[~iba_df["ID"].isin(top10_ids)].copy().reset_index(drop=True)
    logger.info(f"  Context  : {len(context_df)} samples")
    logger.info(f"  Held-out : {len(held_out_df)} samples")

    # ── Load model & encode context ───────────────────────────────────────────
    logger.info(f"Loading model: {args.model_dir}")
    predictor = ViscosityPredictorCNP(args.model_dir, verbose=False)
    logger.info(
        f"  static_dim={predictor.static_dim}, "
        f"hidden_dim={predictor.config['hidden_dim']}, "
        f"latent_dim={predictor.config['latent_dim']}"
    )

    logger.info("Encoding context (top-10 samples) ...")
    predictor.memory_vector = None
    predictor.context_t = None
    predictor.learn(context_df)

    # ── Predict held-out ──────────────────────────────────────────────────────
    logger.info(f"Predicting {len(held_out_df)} held-out samples ...")
    results_df = predictor.predict(held_out_df)

    # ── Build long-form results table ─────────────────────────────────────────
    rows = []
    for _, row in results_df.iterrows():
        for sc in SHEAR_COLS:
            act = row.get(sc, np.nan)
            prd = row.get(f"Pred_{sc}", np.nan)
            valid = pd.notna(act) and pd.notna(prd) and act > 0 and prd > 0
            rows.append(
                {
                    "ID": row["ID"],
                    "Protein_conc": row.get("Protein_conc", np.nan),
                    "Buffer_pH": row.get("Buffer_pH", np.nan),
                    "Salt_type": row.get("Salt_type", np.nan),
                    "Salt_conc": row.get("Salt_conc", np.nan),
                    "Stabilizer_type": row.get("Stabilizer_type", np.nan),
                    "Stabilizer_conc": row.get("Stabilizer_conc", np.nan),
                    "Surfactant_type": row.get("Surfactant_type", np.nan),
                    "shear_col": sc,
                    "shear_label": SHEAR_LABELS[sc],
                    "actual_cP": act,
                    "pred_cP": prd,
                    "log10_error": (np.log10(prd) - np.log10(act)) if valid else np.nan,
                    "fold_error": (prd / act) if valid else np.nan,
                    "pct_error": (abs(prd - act) / act * 100) if valid else np.nan,
                }
            )

    long_df = pd.DataFrame(rows)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "ibalizumab_parity_results.csv")
    long_df.to_csv(csv_path, index=False)
    logger.info(f"Parity results saved: {csv_path}")

    # ── Console metrics summary ───────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("PER-SHEAR-RATE SUMMARY  (held-out 24 samples)")
    logger.info("=" * 65)
    logger.info(
        f"{'Shear Rate':>18}  {'N':>4}  {'MAE log10':>10}  "
        f"{'RMSE log10':>10}  {'Bias':>8}  {'<=2x%':>7}"
    )
    logger.info("-" * 65)
    for sc in SHEAR_COLS:
        sub = long_df[(long_df["shear_col"] == sc) & long_df["log10_error"].notna()]
        if sub.empty:
            continue
        m = calc_metrics(sub["actual_cP"], sub["pred_cP"])
        logger.info(
            f"{SHEAR_LABELS[sc]:>18}  {len(sub):>4}  "
            f"{m['log_mae']:>10.4f}  {m['log_rmse']:>10.4f}  "
            f"{m['log_bias']:>+8.4f}  {m['within_2x']:>6.0f}%"
        )
    logger.info("-" * 65)
    valid_all = long_df[(long_df["actual_cP"] > 0) & (long_df["pred_cP"] > 0)].dropna(
        subset=["actual_cP", "pred_cP"]
    )
    m_all = calc_metrics(valid_all["actual_cP"], valid_all["pred_cP"])
    logger.info(
        f"{'All shear rates':>18}  {len(valid_all):>4}  "
        f"{m_all['log_mae']:>10.4f}  {m_all['log_rmse']:>10.4f}  "
        f"{m_all['log_bias']:>+8.4f}  {m_all['within_2x']:>6.0f}%"
    )
    logger.info("=" * 65)

    # ── Parity plots ──────────────────────────────────────────────────────────
    if not PLOT_ENABLED:
        logger.info("Plots suppressed (PLOT_ENABLED=False).")
        return

    logger.info("Generating parity plots ...")

    make_parity_plot(
        long_df,
        shear_subset=SHEAR_COLS,
        title=("Ibalizumab \u2014 All Shear Rates\n" "10 context  |  24 ablated"),
        out_path=os.path.join(args.out_dir, "parity_ibal_all_shears.png"),
        single_shear=False,
    )

    make_parity_plot(
        long_df,
        shear_subset=["Viscosity_1000"],
        title=(
            "Viscosity @ 1 000 s\u207b\u00b9 \u2014 Ibalizumab\n"
            "10 context  |  24 ablated"
        ),
        out_path=os.path.join(args.out_dir, "parity_ibal_1000.png"),
        single_shear=True,
    )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
