"""
predict_conc_sweep.py
=====================
Generates viscosity vs. protein concentration curves (0–330 mg/mL) for
Nivolumab and Adalimumab in 15 mM Histidine buffer (pH 6.0, 25 °C, no salt,
no excipients, no stabilisers) across all five shear rates.

Outputs
-------
  results/
    sweep_raw_data.csv          – all predicted viscosity values (long format)
    nivolumab_viscosity.png     – linear + log y-axis plots for Nivolumab
    adalimumab_viscosity.png    – linear + log y-axis plots for Adalimumab

Usage
-----
  python predict_conc_sweep.py --model_dir <path_to_model_dir> \
                               --data     <path_to_training_csv>

  model_dir   : directory containing best_model.pth, preprocessor.pkl,
                physics_scaler.pkl  (default: models/experiments/o_net_v3)
  data        : training CSV used to supply calibration context
                (default: data/raw/formulation_data_03042026.csv)
"""

import argparse
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Import the predictor from inference_cnp ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
try:
    from inference_o_net import ViscosityPredictorCNP
except ImportError:
    print("ERROR: Could not import ViscosityPredictorCNP from inference_cnp.py")
    print("       Make sure inference_cnp.py is in the same directory.")
    sys.exit(1)


# ============================================================
# Configuration
# ============================================================

PROTEINS = {
    "Nivolumab": {
        "Protein_class_type": "mAb_IgG4",
        "kP": 3.5,
        "MW": 146.0,
        "PI_mean": 8.8,
        "PI_range": 0.3,
        "C_Class": 1.3,
        "HCI": 1.1,
    },
    "Adalimumab": {
        "Protein_class_type": "mAb_IgG1",
        "kP": 3.0,
        "MW": 148.0,
        "PI_mean": 8.7,
        "PI_range": 0.3,
        "C_Class": 1.0,
        "HCI": 1.0,
    },
}

# Fixed formulation conditions for the sweep
BUFFER_CONDITIONS = dict(
    Buffer_type="Histidine",
    Buffer_pH=6.0,
    Buffer_conc=15.0,
    Salt_type="none",
    Salt_conc=0.0,
    Stabilizer_type="none",
    Stabilizer_conc=0.0,
    Surfactant_type="none",
    Surfactant_conc=0.0,
    Excipient_type="none",
    Excipient_conc=0.0,
    Temperature=25.0,
)

# Concentration sweep (mg/mL).
# [FIX-A] Single uniform grid replaces the previous four-segment concatenation.
# The old approach created density seams at 20→25, 100→105, and 200→210 mg/mL:
# every feature that is a nonlinear function of concentration (conc_sq,
# Phi_Protein, log_conc, KD_Asymptote) changes slope rate at each boundary,
# producing decoder kinks that are visible as bumps on the log-scale plot.
# A uniform grid eliminates all seams. 150 points gives ~2.2 mg/mL resolution
# across the full range, which is sufficient for plotting.
CONC_POINTS = np.concatenate(
    [
        np.array([0.0]),  # true zero reference (excluded from log plots)
        np.linspace(1, 330, 150),  # uniform spacing, no density seams
    ]
)
CONC_POINTS = np.unique(CONC_POINTS)

# Shear rates and display labels
SHEAR_COLS = [
    "Viscosity_100",
    "Viscosity_1000",
    "Viscosity_10000",
    "Viscosity_100000",
    "Viscosity_15000000",
]
SHEAR_RATES = [100, 1_000, 10_000, 100_000, 15_000_000]
SHEAR_LABELS = [
    "100 s⁻¹",
    "1,000 s⁻¹",
    "10,000 s⁻¹",
    "100,000 s⁻¹",
    "15,000,000 s⁻¹",
]

SHEAR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
SHEAR_MARKERS = ["o", "s", "^", "D", "v"]

OUT_DIR = "results"


# ============================================================
# Build the query DataFrame
# ============================================================


def build_query_df(protein_name: str, conc_array: np.ndarray) -> pd.DataFrame:
    """Returns a DataFrame with one row per concentration point."""
    meta = PROTEINS[protein_name]
    rows = []
    for i, c in enumerate(conc_array):
        row = {
            "ID": f"{protein_name}_sweep_{i:04d}",
            "Protein_type": protein_name,
            **meta,
            "Protein_conc": float(c),
            **BUFFER_CONDITIONS,
        }
        # Dummy viscosity columns required by the preprocessor (set to 1.0).
        for col in SHEAR_COLS:
            row[col] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Context selection
# ============================================================


def select_context(
    train_df: pd.DataFrame,
    protein_name: str,
    max_context: int = 30,
) -> pd.DataFrame:
    """
    Returns the best context rows for a given protein.

    Preference order:
      1. Histidine buffer, no excipients/stabilisers (cleanest signal)
      2. Any Histidine buffer row
      3. All rows for that protein (fallback)

    Capped at max_context to match the k used during training.
    """
    sub = train_df[train_df["Protein_type"] == protein_name].copy()

    tier1 = sub[
        (sub["Buffer_type"] == "Histidine")
        & (sub["Stabilizer_type"].str.lower() == "none")
        & (sub["Excipient_type"].str.lower() == "none")
    ]
    tier2 = sub[sub["Buffer_type"] == "Histidine"]
    tier3 = sub

    for tier in [tier1, tier2, tier3]:
        if len(tier) >= 2:
            chosen = tier
            break
    else:
        chosen = tier3

    if len(chosen) > max_context:
        chosen = chosen.sample(max_context, random_state=42)

    return chosen.reset_index(drop=True)


# ============================================================
# Sliding-window context for concentration sweep
# ============================================================

# [FIX-B] Window half-width (mg/mL) used to select context rows.
# The CNP's latent vector r is a fixed global summary of whatever context
# rows were passed to learn().  When that summary is built from the full
# training set, r anchors the decoder to the average viscosity at each
# training concentration.  As the query concentration passes through a
# training anchor the decoder is pulled toward that remembered value and
# then pulled away again, creating a local trough (visible as the
# ~20-70% dips in the Nivolumab sweep at 100-117 and 159-183 mg/mL).
#
# Solution: re-encode r for each query block using only training rows
# within +/-WINDOW_HALF_WIDTH of the current query concentration.  r then
# reflects the local viscosity regime rather than a global average, so
# the decoder sees consistent context and the anchor-collapse artefact
# disappears.
#
# 40 mg/mL is wide enough to capture at least one training point for
# every query position across the 1-330 mg/mL range for both proteins,
# and narrow enough that distant anchor points no longer corrupt the
# local prediction.
WINDOW_HALF_WIDTH: float = 40.0

# Minimum number of context rows required before falling back to a wider
# window (or ultimately the full protein context).
WINDOW_MIN_ROWS: int = 2


def select_context_local(
    train_df: pd.DataFrame,
    protein_name: str,
    query_conc: float,
    window: float = WINDOW_HALF_WIDTH,
    min_rows: int = WINDOW_MIN_ROWS,
    max_context: int = 30,
) -> pd.DataFrame:
    """
    Returns context rows near *query_conc* for sliding-window re-encoding.

    Search order (same buffer-quality tiers as select_context):
      1. Histidine, no excipients/stabilisers, within +/-window
      2. Any Histidine row within +/-window
      3. All rows for the protein within +/-window
      4. Progressively wider windows (2x, 4x, full range) if still sparse

    Capped at *max_context* rows to match the k used during training.
    """
    sub = train_df[train_df["Protein_type"] == protein_name].copy()

    def _tier_candidates(pool: pd.DataFrame, w: float) -> pd.DataFrame:
        in_window = pool[pool["Protein_conc"].between(query_conc - w, query_conc + w)]
        tier1 = in_window[
            (in_window["Buffer_type"] == "Histidine")
            & (in_window["Stabilizer_type"].str.lower() == "none")
            & (in_window["Excipient_type"].str.lower() == "none")
        ]
        tier2 = in_window[in_window["Buffer_type"] == "Histidine"]
        for t in [tier1, tier2, in_window]:
            if len(t) >= min_rows:
                return t
        return pd.DataFrame()

    # Try widening windows: 1x, 2x, 4x, then unrestricted
    for multiplier in [1, 2, 4]:
        ctx = _tier_candidates(sub, window * multiplier)
        if len(ctx) >= min_rows:
            break
    else:
        # Final fallback: use select_context (full protein pool)
        return select_context(train_df, protein_name, max_context)

    if len(ctx) > max_context:
        # Prefer rows closest to the query concentration.
        ctx = ctx.copy()
        ctx["_dist"] = (ctx["Protein_conc"] - query_conc).abs()
        ctx = ctx.nsmallest(max_context, "_dist").drop(columns=["_dist"])

    return ctx.reset_index(drop=True)


# ============================================================
# Isolated-sample filter
# ============================================================


def filter_isolated_samples(
    train_df: pd.DataFrame,
    protein_name: str,
    buffer_conc_lo: float = 10.0,
    buffer_conc_hi: float = 20.0,
) -> pd.DataFrame:
    """
    Return rows from *train_df* that match the histidine-15 mM no-additive
    condition:
      • Protein_type   == protein_name  (case-insensitive)
      • Buffer_type    == "Histidine"   (case-insensitive)
      • Buffer_conc    in [buffer_conc_lo, buffer_conc_hi]
      • Salt_conc      == 0
      • Stabilizer_conc == 0
      • Surfactant_conc == 0
      • Excipient_conc  == 0
    """
    df = train_df.copy()
    for col in [
        "Protein_type",
        "Buffer_type",
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
        "Excipient_type",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("none").astype(str)

    for col in [
        "Salt_conc",
        "Stabilizer_conc",
        "Surfactant_conc",
        "Excipient_conc",
        "Buffer_conc",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    mask = (
        (df["Protein_type"].str.lower() == protein_name.lower())
        & (df["Buffer_type"].str.lower() == "histidine")
        & (df["Buffer_conc"].between(buffer_conc_lo, buffer_conc_hi))
        & (df["Salt_conc"] == 0)
        & (df["Stabilizer_conc"] == 0)
        & (df["Surfactant_conc"] == 0)
        & (df["Excipient_conc"] == 0)
    )
    return df[mask].reset_index(drop=True)


# ============================================================
# Plotting
# ============================================================


def _style_ax(
    ax: plt.Axes,
    title: str,
    ylog: bool,
    ylabel: str,
) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Protein Concentration (mg/mL)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0, 335)
    if ylog:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f"{y:g}" if y >= 1 else f"{y:.2f}")
        )
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    if ylog:
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.4)
    ax.tick_params(labelsize=9)


def make_figure(
    protein_name: str,
    conc_array: np.ndarray,
    pred_visc: dict[str, np.ndarray],
    out_path: str,
    actual_pts: pd.DataFrame | None = None,
) -> None:
    """
    Creates a figure with two side-by-side subplots:
      Left  — linear y-axis
      Right — log₁₀ y-axis
    Each subplot has one curve per shear rate, with actual data points
    overlaid as filled scatter markers when *actual_pts* is provided.
    """
    import matplotlib.lines as mlines

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(
        f"{protein_name}  |  Viscosity vs. Concentration\n"
        f"15 mM Histidine, pH 6.0, 25 °C, no salt/excipient/stabiliser",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    has_actual = actual_pts is not None and len(actual_pts) > 0

    # Plot on both axes.
    for ax, ylog in zip(axes, [False, True]):
        for shear_col, shear_label, color, marker in zip(
            SHEAR_COLS, SHEAR_LABELS, SHEAR_COLORS, SHEAR_MARKERS
        ):
            visc = pred_visc[shear_col]
            # Exclude c=0 from curves (undefined / model artefact).
            mask = conc_array > 0
            ax.plot(
                conc_array[mask],
                visc[mask],
                color=color,
                marker=marker,
                markersize=3,
                linewidth=1.6,
                markeredgewidth=0.5,
                markeredgecolor="white",
                label=shear_label,
            )

            # Overlay actual data points (same colour, larger marker, black edge).
            if actual_pts is not None and shear_col in actual_pts.columns:
                pts = actual_pts.dropna(subset=[shear_col, "Protein_conc"])
                pts = pts[pts[shear_col] > 0]
                if len(pts) > 0:
                    ax.scatter(
                        pts["Protein_conc"].values,
                        pts[shear_col].values,
                        color=color,
                        marker=marker,
                        s=80,
                        zorder=5,
                        edgecolors="black",
                        linewidths=0.9,
                    )

        ylabel = "Viscosity (cP)" if not ylog else "Viscosity (cP) — log scale"
        scale_tag = "Linear scale" if not ylog else "Log scale"
        _style_ax(ax, scale_tag, ylog, ylabel)

    # Build shared legend: shear-rate entries + optional measured-data proxy.
    handles, labels = axes[0].get_legend_handles_labels()
    if has_actual:
        measured_proxy = mlines.Line2D(
            [],
            [],
            color="gray",
            marker="o",
            linestyle="None",
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.9,
            label="Measured (His 15 mM, isolated)",
        )
        handles.append(measured_proxy)
        labels.append(measured_proxy.get_label())

    fig.legend(
        handles,
        labels,
        title="Shear Rate",
        title_fontsize=10,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# Raw data export
# ============================================================


def export_raw_data(
    all_records: list[dict],
    out_path: str,
) -> None:
    df = pd.DataFrame(all_records)
    # Sort for readability.
    df = df.sort_values(["Protein", "Shear_Rate_s-1", "Protein_Conc_mgmL"]).reset_index(
        drop=True
    )
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  Saved: {out_path}  ({len(df)} rows)")


# ============================================================
# Main
# ============================================================


def main(model_dir: str, data_path: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load predictor.
    print(f"\nLoading model from: {model_dir}")
    try:
        predictor = ViscosityPredictorCNP(model_dir, verbose=False)
        print("  Model loaded successfully.")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        sys.exit(1)

    # Load training data for context.
    print(f"\nLoading context pool from: {data_path}")
    try:
        train_df = pd.read_csv(data_path)
        # Ensure string columns have consistent type.
        for col in [
            "Protein_type",
            "Buffer_type",
            "Stabilizer_type",
            "Excipient_type",
            "Salt_type",
        ]:
            if col in train_df.columns:
                train_df[col] = train_df[col].fillna("none").astype(str)
        print(f"  Loaded {len(train_df)} training rows.")
    except FileNotFoundError:
        print(f"  ERROR: {data_path} not found.")
        sys.exit(1)

    all_records: list[dict] = []

    for protein_name in PROTEINS:
        print(f"\n{'─'*60}")
        print(f"  Protein: {protein_name}")

        # ── [FIX-B] Sliding-window context: re-encode r for each query block ──
        # Group consecutive concentration points into blocks that share the
        # same nearest training concentration.  Each block gets its own
        # learn() call so that r reflects the local viscosity regime rather
        # than a global average.  This eliminates the anchor-collapse troughs
        # seen when a fixed global r is used across the full sweep range.
        #
        # We also print a compact summary: how many unique context windows
        # were used and the fallback rate.
        query_df_full = build_query_df(protein_name, CONC_POINTS)
        non_zero_mask = CONC_POINTS > 0
        sweep_concs = CONC_POINTS[non_zero_mask]  # excludes the c=0 reference

        # Collect predictions into per-shear arrays (pre-filled with NaN;
        # the c=0 slot will be predicted separately with the full context).
        pred_visc: dict[str, np.ndarray] = {
            sc: np.full(len(CONC_POINTS), np.nan) for sc in SHEAR_COLS
        }

        # --- Predict c=0 with full global context (used only for the linear
        #     reference line; not part of the biological sweep). ---
        global_ctx = select_context(train_df, protein_name)
        predictor.memory_vector = None
        predictor.context_t = None
        predictor.learn(global_ctx, n_draws=20, k=8)
        zero_df = query_df_full.iloc[[0]]  # the c=0 row
        zero_result = predictor.predict(zero_df)
        for sc in SHEAR_COLS:
            pred_col = f"Pred_{sc}"
            pred_visc[sc][0] = (
                zero_result[pred_col].values[0]
                if pred_col in zero_result.columns
                else np.nan
            )

        # --- Group sweep concs into blocks sharing the same local context ---
        # A new block starts whenever select_context_local() would return a
        # materially different set of rows.  We approximate this by detecting
        # when the set of row IDs changes compared to the previous query.
        blocks: list[tuple[list[int], pd.DataFrame]] = []  # (indices, ctx_df)
        current_ids: frozenset = frozenset()
        current_indices: list[int] = []
        fallback_count = 0

        for rel_i, conc in enumerate(sweep_concs):
            abs_i = rel_i + 1  # +1 because index 0 is the c=0 slot
            ctx = select_context_local(train_df, protein_name, conc)

            # Detect whether the fallback (global) context was used
            global_ids = frozenset(global_ctx.index)
            if frozenset(ctx.index) == global_ids:
                fallback_count += 1

            ids = frozenset(ctx.index)
            if ids != current_ids:
                if current_indices:
                    blocks.append((current_indices, _last_ctx))  # noqa: F821
                current_ids = ids
                current_indices = [abs_i]
                _last_ctx = ctx
            else:
                current_indices.append(abs_i)
        if current_indices:
            blocks.append((current_indices, _last_ctx))

        print(
            f"  [FIX-B] Sliding-window encoding: {len(blocks)} unique context "
            f"windows across {len(sweep_concs)} sweep points "
            f"(fallbacks to global: {fallback_count})"
        )

        # --- Predict each block with its own latent r ---
        for block_indices, ctx_df in blocks:
            predictor.memory_vector = None
            predictor.context_t = None
            predictor.learn(ctx_df, n_draws=20, k=min(8, len(ctx_df)))

            block_query = query_df_full.iloc[block_indices]
            block_result = predictor.predict(block_query)

            for sc in SHEAR_COLS:
                pred_col = f"Pred_{sc}"
                vals = (
                    block_result[pred_col].values
                    if pred_col in block_result.columns
                    else np.zeros(len(block_indices))
                )
                for arr_i, df_i in enumerate(block_indices):
                    pred_visc[sc][df_i] = vals[arr_i]

        print(
            f"  Predicted {len(CONC_POINTS)} concentration points "
            f"({CONC_POINTS[0]:.0f}–{CONC_POINTS[-1]:.0f} mg/mL)"
        )

        # ── Collect raw records (one per conc × shear) ──
        for shear_col, shear_rate in zip(SHEAR_COLS, SHEAR_RATES):
            visc = pred_visc[shear_col]
            for i, conc in enumerate(CONC_POINTS):
                all_records.append(
                    {
                        "Protein": protein_name,
                        "Buffer": "Histidine",
                        "Buffer_pH": 6.0,
                        "Buffer_Conc_mM": 15.0,
                        "Temperature_C": 25.0,
                        "Protein_Conc_mgmL": round(float(conc), 4),
                        "Shear_Rate_s-1": shear_rate,
                        "Viscosity_Pred_cP": round(float(visc[i]), 6),
                    }
                )

        # Print preview table.
        print(
            f"\n  {'Conc (mg/mL)':>14}  " + "  ".join(f"{s:>14}" for s in SHEAR_LABELS)
        )
        print("  " + "-" * (16 + 16 * 5))
        for pc in [0, 5, 25, 50, 100, 150, 200, 250, 300, 330]:
            idx = np.argmin(np.abs(CONC_POINTS - pc))
            row_vals = [pred_visc[sc][idx] for sc in SHEAR_COLS]
            print(
                f"  {CONC_POINTS[idx]:>14.1f}  "
                + "  ".join(f"{v:>14.2f}" for v in row_vals)
            )

        # ── Isolated actual data (Histidine 15 mM, no additives) ──
        isolated = filter_isolated_samples(train_df, protein_name)
        print(
            f"  Isolated His-15mM samples found: {len(isolated)}"
            + (
                f"  (conc range: {isolated['Protein_conc'].min():.0f}–"
                f"{isolated['Protein_conc'].max():.0f} mg/mL)"
                if len(isolated) > 0
                else ""
            )
        )

        # ── Plot ──
        out_png = os.path.join(OUT_DIR, f"{protein_name.lower()}_viscosity.png")
        make_figure(
            protein_name,
            CONC_POINTS,
            pred_visc,
            out_png,
            actual_pts=isolated if len(isolated) > 0 else None,
        )

    # ── Save raw data ──
    print(f"\n{'─'*60}")
    raw_path = os.path.join(OUT_DIR, "sweep_raw_data.csv")
    export_raw_data(all_records, raw_path)

    print(f"\n{'='*60}")
    print("Done.  All outputs written to:", os.path.abspath(OUT_DIR))
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Viscosity concentration sweep for Nivolumab & Adalimumab"
    )
    parser.add_argument(
        "--model_dir",
        default="models/experiments/o_net_v3_debug_aug",
        help="Directory containing best_model.pth and preprocessors "
        "(default: models/experiments/o_net_v3)",
    )
    parser.add_argument(
        "--data",
        default="data/raw/formulation_data_03042026.csv",
        help="Training CSV to source calibration context from "
        "(default: data/raw/formulation_data_03042026.csv)",
    )
    args = parser.parse_args()
    main(args.model_dir, args.data)
