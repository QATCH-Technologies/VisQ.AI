"""
Viscosity vs. Protein Concentration Analysis
=============================================
Filters samples containing ONLY Protein + Buffer (no salt, stabilizer,
surfactant, or excipient), then plots viscosity vs. concentration for each
unique (Protein, Buffer, Buffer_pH, Buffer_conc) combination.
A power-law fit (η = a·C^b) is applied independently for each of the 5 shear
rates and overlaid on the scatter data.
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ── 1.  Load data ─────────────────────────────────────────────────────────────
DATA_FILE = "data/raw/formulation_data_02162026.csv"
OUTPUT_DIR = "viscosity_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

# ── 2.  Filter: protein present + ONLY protein & buffer (no other ingredients) ─
protein_only = df[
    (df["Protein_type"].str.lower() != "none")  # must have protein
    & (df["Salt_conc"].astype(float) == 0)  # no salt
    & (df["Stabilizer_conc"].astype(float) == 0)  # no stabilizer
    & (df["Surfactant_conc"].astype(float) == 0)  # no surfactant
    & (df["Excipient_conc"].astype(float) == 0)  # no excipient
].copy()

protein_only["Protein_conc"] = protein_only["Protein_conc"].astype(float)

print(f"Total rows after filtering: {len(protein_only)}")
print(
    protein_only[
        [
            "ID",
            "Protein_type",
            "Buffer_type",
            "Buffer_pH",
            "Buffer_conc",
            "Protein_conc",
        ]
    ].to_string()
)

# ── 3.  Define shear rates & viscosity columns ─────────────────────────────────
SHEAR_RATES = [100, 1_000, 10_000, 100_000, 15_000_000]
VISC_COLS = [f"Viscosity_{sr}" for sr in SHEAR_RATES]
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]


# ── 4.  Power-law fit function: η = a * C^b ────────────────────────────────────
def power_law(C, a, b):
    return a * np.power(C, b)


def fit_power_law(x, y):
    """Return (a, b, r2, x_fit, y_fit) or None if fit fails."""
    try:
        mask = (np.array(x) > 0) & (np.array(y) > 0)
        xf, yf = np.array(x)[mask], np.array(y)[mask]
        if len(xf) < 3:
            return None
        p0 = [1.0, 1.0]
        popt, _ = curve_fit(
            power_law, xf, yf, p0=p0, bounds=([0, 0], [np.inf, 10]), maxfev=5000
        )
        a, b = popt
        y_pred = power_law(xf, a, b)
        ss_res = np.sum((yf - y_pred) ** 2)
        ss_tot = np.sum((yf - np.mean(yf)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        x_line = np.linspace(xf.min(), xf.max(), 200)
        y_line = power_law(x_line, a, b)
        return a, b, r2, x_line, y_line
    except Exception:
        return None


# ── 5.  Group & plot ───────────────────────────────────────────────────────────
groups = protein_only.groupby(
    ["Protein_type", "Buffer_type", "Buffer_pH", "Buffer_conc"], sort=True
)

print(f"\nFound {len(groups)} unique Protein + Buffer combination(s):\n")

for (protein, buf_type, buf_ph, buf_conc), grp in groups:
    grp = grp.sort_values("Protein_conc")
    concentrations = grp["Protein_conc"].values

    label_str = f"{protein}  |  {buf_type} pH {buf_ph}  " f"({int(buf_conc)} mM)"
    print(
        f"  • {label_str}  →  {len(grp)} sample(s), "
        f"conc range {concentrations.min():.1f}–{concentrations.max():.1f} mg/mL"
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, (sr, vcol, color) in enumerate(zip(SHEAR_RATES, VISC_COLS, COLORS)):
        visc = grp[vcol].values.astype(float)

        # scatter
        ax.scatter(
            concentrations,
            visc,
            color=color,
            s=60,
            zorder=5,
            label=f"$\\dot{{\\gamma}}$ = {sr:,} s⁻¹",
        )

        # power-law fit
        result = fit_power_law(concentrations, visc)
        if result is not None:
            a, b, r2, x_line, y_line = result
            ax.plot(x_line, y_line, color=color, linewidth=1.8, zorder=4)
            # Annotate fit equation near the right end of the line
            x_ann = x_line[-1] * 0.97
            y_ann = y_line[-1]
            ax.annotate(
                f"η={a:.3f}·C^{b:.2f}  R²={r2:.2f}",
                xy=(x_ann, y_ann),
                fontsize=6.5,
                color=color,
                ha="right",
                va="bottom",
            )

    ax.set_xlabel("Protein Concentration (mg/mL)", fontsize=11)
    ax.set_ylabel("Viscosity (cP)", fontsize=11)
    ax.set_title(
        f"Viscosity vs. Concentration\n{label_str}", fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", linestyle="--", alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # safe filename
    safe = f"{protein}_{buf_type}_{int(buf_conc)}mM".replace(" ", "_").replace("/", "-")
    fig_path = os.path.join(OUTPUT_DIR, f"{safe}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {fig_path}")

print("\nDone. All plots saved to:", OUTPUT_DIR)
