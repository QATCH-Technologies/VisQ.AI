"""
test_stabilizer_prior.py
========================
Tests the stabilizer prior (+1) against observed viscosity data for every
protein type and regime (Near-pI / Mixed / Far).

For each protein that has at least one stabilizer observation, generates a
figure styled after the panel-A/B/C analysis:
  - One subplot per regime that has stabilizer data
  - Each point: baseline viscosity (dash) vs observed viscosity (dot)
  - Red = prior wrong (stabilizer reduced η), green = prior correct (raised η)
  - Annotated with stabilizer type and concentration

Also produces a summary heatmap (prior_accuracy_summary.png) showing
% correct across all protein × regime cells.

Outputs (written to ./stabilizer_regime_plots/):
  <protein>_stabilizer_prior.png   — per-protein regime breakdown
  prior_accuracy_summary.png       — heatmap of all cells
  stabilizer_prior_audit.csv       — full row-level audit table
"""

import os
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH = "data/raw/formulation_data_03042026.csv"
OUT_DIR = "stabilizer_regime_plots"
SHEAR_COL = "Viscosity_1000"  # primary shear for comparison
SHEAR_LABEL = "1,000 s⁻¹"
CONC_WINDOW = 30  # ±mg/mL when matching baseline rows

PRIOR_COLOR = "#e74c3c"  # red  — prior wrong
CORRECT_COLOR = "#27ae60"  # green — prior correct
NEUTRAL_COLOR = "#7f8c8d"  # grey — baseline tick

STAB_MARKERS = {"sucrose": "o", "trehalose": "D", "other": "s"}
STAB_LABELS = {"sucrose": "Suc", "trehalose": "Tre", "other": "Stab"}

# Proteins to skip (buffer-only controls, not real protein types)
SKIP_PROTEINS = {"none", "unknown", "nan"}

# ── CCI / Regime helpers ────────────────────────────────────────────────────


def compute_cci_regime(row: pd.Series) -> tuple[float, str]:
    try:
        c = float(row.get("C_Class", 1.0) or 1.0)
        ph = float(row.get("Buffer_pH", 7.0) or 7.0)
        pi = float(row.get("PI_mean", 7.0) or 7.0)
    except (TypeError, ValueError):
        c, ph, pi = 1.0, 7.0, 7.0
    if np.isnan(ph):
        ph = 7.0
    if np.isnan(pi):
        pi = 7.0

    cci = c * np.exp(-abs(ph - pi) / 1.5)
    ptype = str(row.get("Protein_class_type", "")).lower()

    if "mab_igg1" in ptype:
        regime = "Near-pI" if cci >= 0.90 else ("Mixed" if cci >= 0.50 else "Far")
    elif "mab_igg4" in ptype:
        regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.40 else "Far")
    elif any(x in ptype for x in ["fc-fusion", "trispecific"]):
        regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
    elif any(x in ptype for x in ["bispecific", "adc"]):
        regime = "Near-pI" if cci >= 0.80 else ("Mixed" if cci >= 0.45 else "Far")
    elif any(x in ptype for x in ["bsa", "polyclonal"]):
        regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")
    else:
        regime = "Near-pI" if cci >= 0.70 else ("Mixed" if cci >= 0.40 else "Far")

    return cci, regime


def is_stabilizer(s: str) -> bool:
    return any(x in str(s).lower() for x in ["sucrose", "trehalose"])


def stab_key(s: str) -> str:
    s = str(s).lower()
    if "sucrose" in s:
        return "sucrose"
    if "trehalose" in s:
        return "trehalose"
    return "other"


# ── Data loading ────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    str_cols = [
        "Protein_type",
        "Protein_class_type",
        "Buffer_type",
        "Salt_type",
        "Stabilizer_type",
        "Surfactant_type",
        "Excipient_type",
    ]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].fillna("none").astype(str).str.strip().str.lower()

    df["cci"], df["regime"] = zip(*df.apply(compute_cci_regime, axis=1))
    df["has_stabilizer"] = df["Stabilizer_type"].apply(is_stabilizer)
    df["stab_key"] = df["Stabilizer_type"].apply(stab_key)
    return df


# ── Baseline matching ───────────────────────────────────────────────────────


def find_baseline(
    prot_df: pd.DataFrame,
    target_conc: float,
    shear_col: str = SHEAR_COL,
    window: float = CONC_WINDOW,
) -> float | None:
    """
    Returns mean viscosity at shear_col for no-stabilizer rows within
    ±window mg/mL of target_conc.  Falls back to ±2×window, then all
    no-stabilizer rows if still empty.
    """
    no_stab = prot_df[~prot_df["has_stabilizer"]]
    for w in [window, window * 2, np.inf]:
        close = no_stab[np.abs(no_stab["Protein_conc"] - target_conc) <= w]
        vals = close[shear_col].replace(0, np.nan).dropna()
        if len(vals) > 0:
            return float(vals.mean())
    return None


# ── Build audit records ─────────────────────────────────────────────────────


def build_audit(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for prot, pgrp in df.groupby("Protein_type"):
        if prot in SKIP_PROTEINS:
            continue
        stab_rows = pgrp[pgrp["has_stabilizer"]]
        if len(stab_rows) == 0:
            continue

        for _, row in stab_rows.iterrows():
            conc = float(row["Protein_conc"])
            stab_c = float(row.get("Stabilizer_conc", 0) or 0)
            stab_t = row["stab_key"]
            regime = row["regime"]
            cci = float(row["cci"])
            obs_visc = row.get(SHEAR_COL)

            if not pd.notna(obs_visc) or float(obs_visc) <= 0:
                continue
            obs_visc = float(obs_visc)

            base = find_baseline(pgrp, conc)
            if base is None or base <= 0:
                continue

            delta_log10 = np.log10(obs_visc) - np.log10(base)
            prior_correct = delta_log10 > 0  # prior says +1 → expects increase

            records.append(
                {
                    "ID": row.get("ID", ""),
                    "Protein": prot,
                    "Protein_class": row["Protein_class_type"],
                    "Regime": regime,
                    "CCI": round(cci, 3),
                    "Protein_conc": conc,
                    "Stabilizer": stab_t,
                    "Stab_conc_M": stab_c,
                    "Buffer_pH": row.get("Buffer_pH"),
                    "Obs_visc_cP": round(obs_visc, 3),
                    "Base_visc_cP": round(base, 3),
                    "Delta_log10": round(delta_log10, 4),
                    "Prior_correct": prior_correct,
                }
            )

    return pd.DataFrame(records)


# ── Per-protein figure (panels A-C style) ──────────────────────────────────

REGIME_ORDER = ["Near-pI", "Mixed", "Far"]


def make_protein_figure(
    prot: str,
    prot_class: str,
    audit_df: pd.DataFrame,
    prot_raw: pd.DataFrame,
    out_path: str,
) -> dict:
    """
    One figure per protein.  Columns = regimes that have data.
    Returns a dict of {regime: (n_correct, n_total)} for the summary heatmap.
    """
    prot_audit = audit_df[audit_df["Protein"] == prot]
    regimes_present = [r for r in REGIME_ORDER if r in prot_audit["Regime"].values]

    n_cols = len(regimes_present)
    fig_w = max(5 * n_cols, 6)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(fig_w, 5.5), squeeze=False, constrained_layout=True
    )
    fig.patch.set_facecolor("#f8f9fa")

    accuracy_by_regime = {}

    for col_i, regime in enumerate(regimes_present):
        ax = axes[0, col_i]
        rdata = prot_audit[prot_audit["Regime"] == regime].copy()

        n_correct = rdata["Prior_correct"].sum()
        n_total = len(rdata)
        pct = n_correct / n_total * 100 if n_total > 0 else 0
        accuracy_by_regime[regime] = (n_correct, n_total)

        ax.set_facecolor("white")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#cccccc")

        # Plot each observation
        for _, rec in rdata.iterrows():
            x_jitter = rec["Protein_conc"] + np.random.uniform(-2, 2)
            yb = rec["Base_visc_cP"]
            yo = rec["Obs_visc_cP"]
            color = CORRECT_COLOR if rec["Prior_correct"] else PRIOR_COLOR
            marker = STAB_MARKERS.get(rec["Stabilizer"], "o")

            # Vertical connector
            ax.plot(
                [x_jitter, x_jitter],
                [yb, yo],
                color=color,
                alpha=0.55,
                linewidth=1.5,
                zorder=2,
            )
            # Baseline tick
            ax.scatter(
                x_jitter,
                yb,
                marker="_",
                s=90,
                color=NEUTRAL_COLOR,
                linewidths=2.5,
                zorder=3,
            )
            # Observed dot
            ax.scatter(
                x_jitter,
                yo,
                marker=marker,
                s=50,
                color=color,
                alpha=0.88,
                zorder=4,
                edgecolors="white",
                linewidths=0.5,
            )

            # Annotate concentration
            label_txt = (
                f"{STAB_LABELS.get(rec['Stabilizer'], '?')}\n{rec['Stab_conc_M']:.2f}M"
            )
            ax.annotate(
                label_txt,
                (x_jitter, max(yb, yo)),
                textcoords="offset points",
                xytext=(3, 4),
                fontsize=6,
                color="#444444",
                va="bottom",
            )

        # Accuracy badge
        badge_color = CORRECT_COLOR if pct >= 60 else PRIOR_COLOR
        ax.text(
            0.97,
            0.97,
            f"{n_correct}/{n_total} correct\n({pct:.0f}%)",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="top",
            color=badge_color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=badge_color,
                alpha=0.85,
            ),
        )

        # Axis labels / title
        ax.set_title(f"Regime: {regime}", fontsize=11, fontweight="bold", pad=4)
        ax.set_xlabel("Protein conc (mg/mL)", fontsize=9)
        if col_i == 0:
            ax.set_ylabel(f"Viscosity @ {SHEAR_LABEL} (cP)", fontsize=9)
        ax.tick_params(labelsize=8)

    # Legend (shared, placed outside rightmost axis)
    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="_",
            color=NEUTRAL_COLOR,
            markersize=12,
            linewidth=0,
            label="Baseline (no stabilizer)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=PRIOR_COLOR,
            markersize=7,
            linewidth=0,
            label="Reduces η  — prior WRONG",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=CORRECT_COLOR,
            markersize=7,
            linewidth=0,
            label="Increases η — prior correct",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="grey",
            markersize=6,
            linewidth=0,
            label="○ sucrose  ◇ trehalose",
        ),
        Line2D([0], [0], marker="D", color="grey", markersize=6, linewidth=0, label=""),
    ]
    axes[0, -1].legend(
        handles=legend_elems[:4],
        fontsize=8,
        loc="upper left",
        framealpha=0.9,
        bbox_to_anchor=(1.02, 1.0),
    )

    prot_display = prot.title()
    fig.suptitle(
        f"{prot_display}  ({prot_class.upper()})\n"
        f"Stabilizer Prior (+1) vs Observed  |  {SHEAR_LABEL}",
        fontsize=13,
        fontweight="bold",
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return accuracy_by_regime


# ── Summary heatmap ─────────────────────────────────────────────────────────


def make_summary_heatmap(
    accuracy_table: dict,  # {protein: {regime: (n_correct, n_total)}}
    out_path: str,
) -> None:
    proteins = sorted(accuracy_table.keys())
    regimes = REGIME_ORDER

    pct_matrix = np.full((len(proteins), len(regimes)), np.nan)
    count_matrix = np.full((len(proteins), len(regimes)), 0, dtype=int)

    for pi, prot in enumerate(proteins):
        for ri, reg in enumerate(regimes):
            if reg in accuracy_table[prot]:
                nc, nt = accuracy_table[prot][reg]
                if nt > 0:
                    pct_matrix[pi, ri] = nc / nt * 100
                    count_matrix[pi, ri] = nt

    fig, ax = plt.subplots(
        figsize=(8, max(5, len(proteins) * 0.7 + 2.5)), constrained_layout=True
    )
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f0f0f0")

    # Custom colormap: red (0%) → white (50%) → green (100%)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "rwg", [PRIOR_COLOR, "#ffffff", CORRECT_COLOR], N=256
    )

    masked = np.ma.masked_where(np.isnan(pct_matrix), pct_matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Cell annotations
    for pi in range(len(proteins)):
        for ri in range(len(regimes)):
            if not np.isnan(pct_matrix[pi, ri]):
                pct = pct_matrix[pi, ri]
                nt = count_matrix[pi, ri]
                txt_color = "white" if (pct < 25 or pct > 80) else "#333333"
                ax.text(
                    ri,
                    pi,
                    f"{pct:.0f}%\n(n={nt})",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=txt_color,
                    fontweight="bold",
                )
            else:
                ax.text(
                    ri, pi, "—", ha="center", va="center", fontsize=10, color="#aaaaaa"
                )

    # Axes
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels([p.title() for p in proteins], fontsize=10)
    ax.set_xlabel("Charge Interaction Regime", fontsize=11, labelpad=8)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position("top")

    # 50% reference line annotation on colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Prior correct (%)", fontsize=10)
    cbar.ax.axhline(50, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    cbar.ax.text(2.8, 50, "50%\n(random)", va="center", fontsize=8, color="#333333")

    # Grid lines
    for x in np.arange(-0.5, len(regimes), 1):
        ax.axvline(x, color="#cccccc", linewidth=0.8)
    for y in np.arange(-0.5, len(proteins), 1):
        ax.axhline(y, color="#cccccc", linewidth=0.8)

    ax.set_title(
        "Stabilizer Prior (+1) Accuracy by Protein × Regime\n"
        "Green = prior correct (stabilizer raised η)  |  Red = prior wrong (stabilizer reduced η)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary heatmap → {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(42)

    print(f"Loading data from {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    print(f"  {len(df)} rows | {df['Protein_type'].nunique()} protein types")

    print("\nBuilding audit table ...")
    audit = build_audit(df)
    print(f"  {len(audit)} stabilizer observations matched to baselines")

    # Save full audit CSV
    csv_path = os.path.join(OUT_DIR, "stabilizer_prior_audit.csv")
    audit.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Audit table → {csv_path}")

    # Overall accuracy summary in console
    n_correct = audit["Prior_correct"].sum()
    n_total = len(audit)
    print(
        f"\n  Overall prior accuracy: {n_correct}/{n_total} = {n_correct/n_total*100:.1f}%"
    )
    print(f"\n  Per-protein accuracy:")
    for prot, grp in audit.groupby("Protein"):
        nc, nt = grp["Prior_correct"].sum(), len(grp)
        bar = "█" * int(nc / nt * 20) + "░" * (20 - int(nc / nt * 20))
        print(f"    {prot:20s}  {bar}  {nc:3d}/{nt:3d} = {nc/nt*100:5.1f}%")

    print(f"\n  Per-regime accuracy:")
    for regime, grp in audit.groupby("Regime"):
        nc, nt = grp["Prior_correct"].sum(), len(grp)
        print(f"    {regime:10s}  {nc:3d}/{nt:3d} = {nc/nt*100:5.1f}%")

    # Per-protein figures
    accuracy_table = {}
    proteins = sorted([p for p in audit["Protein"].unique() if p not in SKIP_PROTEINS])
    print(f"\nGenerating {len(proteins)} protein figures ...")

    for prot in proteins:
        prot_class = df[df["Protein_type"] == prot]["Protein_class_type"].iloc[0]
        prot_raw = df[df["Protein_type"] == prot]
        out_path = os.path.join(
            OUT_DIR, f"{prot.replace(' ', '_')}_stabilizer_prior.png"
        )

        acc = make_protein_figure(prot, prot_class, audit, prot_raw, out_path)
        accuracy_table[prot] = acc

        regime_strs = [f"{r}:{v[0]}/{v[1]}" for r, v in acc.items()]
        print(
            f"  {prot:20s}  {' | '.join(regime_strs)}  → {os.path.basename(out_path)}"
        )

    # Summary heatmap
    print("\nGenerating summary heatmap ...")
    heatmap_path = os.path.join(OUT_DIR, "prior_accuracy_summary.png")
    make_summary_heatmap(accuracy_table, heatmap_path)

    print(f"\nDone. All outputs in: {os.path.abspath(OUT_DIR)}/")
    print(f"  {len(proteins)} protein figures + 1 summary heatmap + 1 audit CSV")


if __name__ == "__main__":
    main()
