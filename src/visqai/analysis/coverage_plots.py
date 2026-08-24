"""
coverage_plots.py
==================
Plot generation (P1-P6, plus v2's P2b/P5'/P6b/P9) for the dataset coverage &
sparsity census in coverage.py. One figure per file, PNG at 200 dpi, scope
tag in the axis label or subtitle, colorblind-safe sequential map for
magnitude, cells below MIN_CELL_FOR_DISPLAY hatched rather than colored so
"thin" and "zero" read as visually distinct (see dataviz skill).

CATEGORICAL COLOR NOTE
-----------------------
The documented categorical palette (dataviz skill, references/palette.md)
only validates all-pairs CVD separation through its first THREE slots; a
4th+ slot in a scatter/all-pairs chart form is a known FAIL (e.g. slot 4
yellow vs slot 2 orange measures CVD dE 4.8 dark, below the 6.0 floor). This
module has real >3-category data (12 proteins, 2 remedy classes is fine,
but individual-protein identity is not) -- resolved per-plot:
  - P2/P2b use ZERO categorical hue for protein identity: every point is
    individually text-labeled (per the plan's own spec), so color carries
    no identity burden at all.
  - P4 needs "one color per protein" to show viscosity concentrated in a
    few clusters; per the skill's own prescribed remedy ("past three, fold
    to Other"), only the top-3 highest-viscosity proteins get a validated
    categorical slot -- the remaining 9 render in a single shared muted
    tone, which is also the plot's actual point (a small number of
    clusters carries the signal).
  - P5'/P9 use frontier-vs-dominated (2 categories) or on_frontier-vs-not --
    both fit the validated zone.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.neighbors import NearestNeighbors

from visqai.analysis import coverage as cov

OUTPUT_DIR = "reports/coverage"

# --- Palette (dataviz skill: references/palette.md) -------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
CHART_SURFACE = "#fcfcfb"

SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("visq_sequential_blue", SEQUENTIAL_BLUE_STEPS)

# Categorical slots 1-3 only (the all-pairs-validated zone; see module docstring).
CAT_SLOT_1 = "#2a78d6"  # blue
CAT_SLOT_2 = "#eb6834"  # orange
CAT_SLOT_3 = "#1baf7a"  # aqua
NEUTRAL_OTHER = "#c3c2b7"  # BASELINE tone -- "everything else," decorative only

# Task 7: hatch pattern for cells below MIN_CELL_FOR_DISPLAY, so "thin" and
# "zero" stay visually distinct from a genuinely well-supported cell.
HATCH_PATTERN = "///"

plt.rcParams.update(
    {
        "figure.facecolor": CHART_SURFACE,
        "axes.facecolor": CHART_SURFACE,
        "savefig.facecolor": CHART_SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
    }
)


def _save(fig, filename: str, out_dir: str) -> None:
    fig.savefig(os.path.join(out_dir, filename), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, filename.replace(".png", ".pdf")), bbox_inches="tight")
    plt.close(fig)


def _label_points_with_collision_avoidance(ax, xs, ys, labels, x_range, y_range) -> None:
    """Direct point labels with radial leader-line fan-out for any group of
    points that sit close together in normalized-axis space -- a plain
    fixed offset silently overlaps labels when points cluster (Task 7:
    "render it and look at it" -- this is what that check catches)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    nx = (xs - x_range[0]) / (x_range[1] - x_range[0])
    ny = (ys - y_range[0]) / (y_range[1] - y_range[0])
    n = len(xs)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    proximity_threshold = 0.07  # normalized-axis-fraction collision radius
    for i in range(n):
        for j in range(i + 1, n):
            if abs(nx[i] - nx[j]) < proximity_threshold and abs(ny[i] - ny[j]) < proximity_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    for members in groups.values():
        if len(members) == 1:
            i = members[0]
            ax.annotate(labels[i], (xs[i], ys[i]), textcoords="offset points", xytext=(8, 8), fontsize=8, color=INK_SECONDARY, zorder=4)
            continue
        k = len(members)
        radius_pts = 30 + 8 * k
        for idx, i in enumerate(members):
            angle = 2 * np.pi * idx / k - np.pi / 2
            dx = radius_pts * np.cos(angle)
            dy = radius_pts * np.sin(angle) + 18
            ax.annotate(
                labels[i],
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8,
                color=INK_SECONDARY,
                zorder=4,
                arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=0.7, shrinkA=0, shrinkB=3),
            )


def _hatch_thin_cells(ax, mat: np.ndarray, extent=None, cell_size=1.0, origin_offset=(0.0, 0.0)):
    """Overlay a hatch rectangle on every cell below MIN_CELL_FOR_DISPLAY
    (Task 7) -- including true zeros, so absence and thinness both read as
    visually distinct from a supported cell."""
    ox, oy = origin_offset
    for (i, j), v in np.ndenumerate(mat):
        if np.isnan(v) or v < cov.MIN_CELL_FOR_DISPLAY:
            ax.add_patch(
                Rectangle(
                    (j - 0.5 + ox, i - 0.5 + oy),
                    cell_size,
                    cell_size,
                    fill=False,
                    hatch=HATCH_PATTERN,
                    edgecolor=BASELINE,
                    linewidth=0.0,
                )
            )


# =============================================================================
# P1 -- LOGO support matrix
# =============================================================================


def plot_p1_logo_support_matrix(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    non_placebo_pt = cov._normalize(df["Protein_type"])
    is_placebo = cov._is_placebo(df["Protein_type"])
    protein_order = sorted(non_placebo_pt[~is_placebo].unique()) + ["none (placebo)"]

    level_cols: list[str] = []
    counts = pd.DataFrame(index=protein_order)
    cluster_marginal = {}
    for axis in cov.CLASS_B_AXES:
        levels = cov.class_b_levels(df, axis)
        norm_level = cov._normalize_level(df[axis])
        pt_label = non_placebo_pt.where(~is_placebo, "none (placebo)")
        for level in sorted(norm_level.unique()):
            col = f"{axis}={level}"
            level_cols.append(col)
            mask = norm_level == level
            counts[col] = pt_label[mask].value_counts().reindex(protein_order, fill_value=0)
            cluster_marginal[col] = int(levels.loc[levels["level"] == level, "n_clusters"].iloc[0])

    mat = counts[level_cols].to_numpy(dtype=float)

    fig = plt.figure(figsize=(max(12.0, 0.55 * len(level_cols)), 8.5))
    ax_bar = fig.add_axes([0.16, 0.80, 0.80, 0.12])
    ax_heat = fig.add_axes([0.16, 0.08, 0.80, 0.70])
    cax = fig.add_axes([0.97, 0.08, 0.015, 0.70])

    required = cov.REQUIRED_CLUSTERS_GENERALIZATION.value
    marginal_vals = [cluster_marginal[c] for c in level_cols]
    bar_colors = [INK_MUTED if v < required else CAT_SLOT_1 for v in marginal_vals]
    ax_bar.bar(range(len(level_cols)), marginal_vals, color=bar_colors, width=0.7)
    ax_bar.axhline(required, color=INK_SECONDARY, linewidth=1.0, linestyle="--")
    ax_bar.text(
        len(level_cols) - 0.5,
        required + 0.3,
        f"required_clusters = {required:g}",
        fontsize=8,
        color=INK_SECONDARY,
        ha="right",
    )
    ax_bar.set_xlim(-0.5, len(level_cols) - 0.5)
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("n_clusters\n(between_protein)", fontsize=8)
    ax_bar.set_title(
        "P1 -- LOGO support matrix (cells: row scope; top bars: cluster support, between_protein scope)",
        fontsize=12,
        fontweight="600",
        loc="left",
    )

    im = ax_heat.imshow(mat, aspect="auto", cmap=SEQUENTIAL_CMAP, vmin=0)
    _hatch_thin_cells(ax_heat, mat)
    ax_heat.set_yticks(range(len(protein_order)))
    ax_heat.set_yticklabels(protein_order, fontsize=8)
    ax_heat.set_xticks(range(len(level_cols)))
    ax_heat.set_xticklabels(level_cols, fontsize=7, rotation=90)
    ax_heat.set_xlabel("ingredient level", fontsize=9)
    ax_heat.set_ylabel("Protein_type (12 clusters + placebo stratum)", fontsize=9)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("n_rows (row scope)", fontsize=8)
    cbar.outline.set_visible(False)

    _save(fig, "p1_logo_support_matrix.png", out_dir)


# =============================================================================
# P2 -- Descriptor space occupancy
# =============================================================================


def plot_p2_descriptor_occupancy(df: pd.DataFrame, axis_x: str = "PI_mean", axis_y: str = "kP", out_dir: str = OUTPUT_DIR) -> None:
    non_placebo = df.loc[~cov._is_placebo(df["Protein_type"])].copy()
    non_placebo["_pt"] = cov._normalize(non_placebo["Protein_type"])
    per_protein = non_placebo.groupby("_pt")[[axis_x, axis_y]].first()

    x_edges = cov.CLASS_A_AXES[axis_x]
    y_edges = cov.CLASS_A_AXES[axis_y]
    pair = cov.class_a_pair(df, axis_x, axis_y)
    occ = pair.set_index(["bin_x", "bin_y"])["n_clusters"]

    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    x_labels = sorted(set(pair["bin_x"]))
    y_labels = sorted(set(pair["bin_y"]))
    for xi, xl in enumerate(x_edges[:-1]):
        for yi, yl in enumerate(y_edges[:-1]):
            bx = f"[{cov._fmt_edge(x_edges[xi])}, {cov._fmt_edge(x_edges[xi + 1])})"
            by = f"[{cov._fmt_edge(y_edges[yi])}, {cov._fmt_edge(y_edges[yi + 1])})"
            n = int(occ.get((bx, by), 0))
            face = "none" if n > 0 else "#f3f2ee"
            ax.add_patch(Rectangle((x_edges[xi], y_edges[yi]), x_edges[xi + 1] - x_edges[xi], y_edges[yi + 1] - y_edges[yi], facecolor=face, edgecolor=GRIDLINE, linewidth=0.8, zorder=1))

    ax.scatter(per_protein[axis_x], per_protein[axis_y], s=90, color=CAT_SLOT_1, edgecolors="white", linewidths=0.9, zorder=3)
    _label_points_with_collision_avoidance(
        ax, per_protein[axis_x].to_numpy(), per_protein[axis_y].to_numpy(), list(per_protein.index),
        x_range=(x_edges[0], x_edges[-1]), y_range=(y_edges[0], y_edges[-1]),
    )

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xlabel(f"{axis_x} (between_protein; n = 12 proteins, not 434 rows)")
    ax.set_ylabel(axis_y)
    ax.set_title("P2 -- Descriptor space occupancy (n = 12 proteins)", fontsize=14, fontweight="600", loc="left")
    ax.grid(False)
    _save(fig, "p2_descriptor_occupancy.png", out_dir)


# =============================================================================
# P3 -- Concentration grid completeness
# =============================================================================


def plot_p3_concentration_grid(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    axis = "Protein_conc"
    grid = cov.class_c_grid(df, axis)
    edges = cov.CLASS_C_AXES[axis]
    bin_labels = [f"[{cov._fmt_edge(edges[i])}, {cov._fmt_edge(edges[i + 1])})" for i in range(len(edges) - 1)]
    proteins = sorted(grid["protein"].unique())

    mat = grid.pivot(index="protein", columns="bin", values="n_rows").reindex(index=proteins, columns=bin_labels).to_numpy(dtype=float)
    _, per_bin = cov.class_c_grid_completeness(grid, axis)
    marginal = per_bin.set_index("bin")["n_clusters_occupying"].reindex(bin_labels, fill_value=0)

    fig = plt.figure(figsize=(9.5, 8.0))
    ax_heat = fig.add_axes([0.18, 0.24, 0.72, 0.68])
    ax_bar = fig.add_axes([0.18, 0.08, 0.72, 0.12])
    cax = fig.add_axes([0.92, 0.24, 0.02, 0.68])

    im = ax_heat.imshow(mat, aspect="auto", cmap=SEQUENTIAL_CMAP, vmin=0)
    _hatch_thin_cells(ax_heat, mat)
    ax_heat.set_yticks(range(len(proteins)))
    ax_heat.set_yticklabels(proteins, fontsize=8)
    ax_heat.set_xticks([])
    ax_heat.set_title(
        "P3 -- Protein_conc grid completeness (cells: within_protein scope)",
        fontsize=13,
        fontweight="600",
        loc="left",
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("n_rows", fontsize=8)
    cbar.outline.set_visible(False)

    ax_bar.bar(range(len(bin_labels)), marginal.to_numpy(), color=CAT_SLOT_1, width=0.7)
    ax_bar.set_xticks(range(len(bin_labels)))
    ax_bar.set_xticklabels(bin_labels, fontsize=8, rotation=20, ha="right")
    ax_bar.set_xlabel("Protein_conc bin (mg/mL)")
    ax_bar.set_ylabel("clusters\noccupying\n(between_protein)", fontsize=7)

    _save(fig, "p3_concentration_grid_completeness.png", out_dir)


# =============================================================================
# P4 -- Response coverage
# =============================================================================


def plot_p4_response_coverage(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    non_placebo = df.loc[~cov._is_placebo(df["Protein_type"])].copy()
    non_placebo["_pt"] = cov._normalize(non_placebo["Protein_type"])
    non_placebo["log_v1000"] = np.log10(non_placebo["Viscosity_1000"].where(non_placebo["Viscosity_1000"] > 0))

    top3 = non_placebo.groupby("_pt")["log_v1000"].max().sort_values(ascending=False).head(3).index.tolist()
    top3_colors = dict(zip(top3, [CAT_SLOT_1, CAT_SLOT_2, CAT_SLOT_3]))

    fig = plt.figure(figsize=(10.0, 8.0))
    ax = fig.add_axes([0.10, 0.10, 0.62, 0.62])
    ax_histx = fig.add_axes([0.10, 0.73, 0.62, 0.18], sharex=ax)
    ax_histy = fig.add_axes([0.74, 0.10, 0.18, 0.62], sharey=ax)

    other = non_placebo.loc[~non_placebo["_pt"].isin(top3)]
    ax.scatter(other["Protein_conc"], other["log_v1000"], s=26, color=NEUTRAL_OTHER, alpha=0.6, edgecolors="none", label=f"Other proteins (n={other['_pt'].nunique()})", zorder=2)
    for pt in top3:
        g = non_placebo.loc[non_placebo["_pt"] == pt]
        ax.scatter(g["Protein_conc"], g["log_v1000"], s=44, color=top3_colors[pt], alpha=0.9, edgecolors="white", linewidths=0.6, label=pt, zorder=3)

    x_right = float(non_placebo["Protein_conc"].max())
    for thr in cov.LOG_VISC_1000_THRESHOLDS:
        ax.axhline(thr, color=INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=1)
        ax.text(x_right * 0.99, thr, f"log10 v1000 > {thr:g} ", fontsize=8, color=INK_SECONDARY, va="bottom", ha="right")

    ax_histx.hist(non_placebo["Protein_conc"], bins=20, color=BASELINE, edgecolor="white")
    ax_histx.tick_params(labelbottom=False)
    ax_histx.set_yticks([])
    ax_histy.hist(non_placebo["log_v1000"], bins=20, orientation="horizontal", color=BASELINE, edgecolor="white")
    ax_histy.tick_params(labelleft=False)
    ax_histy.set_xticks([])

    ax.set_xlabel("Protein_conc (mg/mL)")
    ax.set_ylabel("log10(Viscosity_1000) [row scope]")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax_histx.set_title(
        "P4 -- Response coverage: high-viscosity region's dependence on a small number of clusters\n"
        "(top-3 highest-viscosity proteins colored; remaining 9 folded to a neutral tone -- see module docstring)",
        fontsize=11,
        fontweight="600",
        loc="left",
    )
    _save(fig, "p4_response_coverage.png", out_dir)


# =============================================================================
# P2b -- Whitened protein descriptor space + void map (Task 9, v2 primary;
# replaces P2 as the primary descriptor-space view. P2 (raw-units, 2-axis)
# is kept as a companion.)
# =============================================================================


def plot_p2b_whitened_void_space(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    space = cov.class_a_descriptor_space(df)
    per_protein, scaler, pca = space["per_protein"], space["scaler"], space["pca"]
    eigvals = pca.explained_variance_
    scores2 = pca.transform(scaler.transform(per_protein.to_numpy(dtype=float)))[:, :2]
    whitened2 = scores2 / np.sqrt(eigvals[:2])

    non_placebo = df.loc[~cov._is_placebo(df["Protein_type"])].copy()
    non_placebo["_pt"] = cov._normalize(non_placebo["Protein_type"])
    non_placebo["log_v1000"] = np.log10(non_placebo["Viscosity_1000"].where(non_placebo["Viscosity_1000"] > 0))
    global_min = float(non_placebo["log_v1000"].min())
    global_max = float(non_placebo["log_v1000"].max())
    lev_fn = cov._leverage_fn("range", None, global_min, global_max)
    protein_leverage = np.array(
        [lev_fn(non_placebo.loc[non_placebo["_pt"] == pt, "log_v1000"], "p2b", pt) for pt in per_protein.index]
    )

    pad = 0.6
    xs = np.linspace(whitened2[:, 0].min() - pad, whitened2[:, 0].max() + pad, 120)
    ys = np.linspace(whitened2[:, 1].min() - pad, whitened2[:, 1].max() + pad, 120)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    nn = NearestNeighbors(n_neighbors=1).fit(whitened2)
    dist, idx = nn.kneighbors(grid)
    void_surface = (dist.ravel() * protein_leverage[idx.ravel()]).reshape(gx.shape)

    void = cov.class_a_void_regions(df)

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    cf = ax.contourf(gx, gy, void_surface, levels=14, cmap=SEQUENTIAL_CMAP, alpha=0.85, zorder=1)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("void_score in the PC1/PC2 plane (2D proxy -- ranking uses all retained components)", fontsize=8)
    cbar.outline.set_visible(False)

    ax.scatter(whitened2[:, 0], whitened2[:, 1], s=100, color=CAT_SLOT_1, edgecolors="white", linewidths=1.0, zorder=3)
    _label_points_with_collision_avoidance(
        ax, whitened2[:, 0], whitened2[:, 1], list(per_protein.index),
        x_range=(xs.min(), xs.max()), y_range=(ys.min(), ys.max()),
    )
    # Each void region's rank, marked near its nearest_protein's point --
    # the region itself lives in the full retained-PCA subspace (9.3), so
    # this PC1/PC2 marker is a locator, not the region's true extent.
    for _, row in void.iterrows():
        wp = whitened2[list(per_protein.index).index(row["nearest_protein"])]
        ax.annotate(
            f"#{int(row['rank'])}", wp, textcoords="offset points", xytext=(-14, 14),
            fontsize=9, fontweight="600", color=CAT_SLOT_2, zorder=5,
            bbox=dict(boxstyle="circle,pad=0.15", facecolor="white", edgecolor=CAT_SLOT_2, linewidth=1.0),
        )

    ax.set_xlabel(f"PC1 (whitened; {space['eigenspectrum'].loc[0, 'explained_variance_ratio']:.1%} of variance)")
    ax.set_ylabel(f"PC2 (whitened; {space['eigenspectrum'].loc[1, 'explained_variance_ratio']:.1%} of variance)")
    subtitle = (
        f"retained-to-90% components: {space['n_components_90']} of 6  ·  "
        f"participation ratio: {space['participation_ratio']:.2f}  ·  "
        f"PC1+PC2 cumulative variance: {space['eigenspectrum'].loc[1, 'cumulative_variance_ratio']:.1%}"
    )
    ax.set_title("P2b -- Whitened protein descriptor space (n = 12 proteins)", fontsize=14, fontweight="600", loc="left", pad=18)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_MUTED)
    ax.grid(False)
    _save(fig, "p2b_whitened_void_space.png", out_dir)


# =============================================================================
# P5' -- Gap report: two remedy panels (never combined into one scored list,
# Task 11) + a short class-A void-region panel (Task 9)
# =============================================================================


def _plot_remedy_panel(ax, table: pd.DataFrame, title: str) -> None:
    if len(table) == 0:
        ax.text(0.5, 0.5, "no gaps above threshold", ha="center", va="center", fontsize=10, color=INK_MUTED, transform=ax.transAxes)
        ax.set_title(title, fontsize=12, fontweight="600", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    top = table.iloc[::-1]  # frontier-first / deficit-desc order already set by _finalize_remedy_table; reverse for barh
    colors = [CAT_SLOT_1 if f else BASELINE for f in top["on_frontier"]]
    labels = [f"{a} = {b}" for a, b in zip(top["axis"], top["bin"])]

    ax.barh(range(len(top)), top["deficit"], color=colors, height=0.7, zorder=2)
    for i, (d, lev) in enumerate(zip(top["deficit"], top["leverage"])):
        ax.text(d + max(top["deficit"]) * 0.02, i, f"leverage={lev:.2f}", fontsize=7, color=INK_SECONDARY, va="center")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("deficit (color: on Pareto frontier of (deficit, leverage))", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="600", loc="left")


def plot_p5_gap_report(report: dict, top_n: int = 20, out_dir: str = OUTPUT_DIR) -> None:
    """Task 14 P5' (replaces P5): acquire_cluster and acquire_rows_within_cluster
    ranked SEPARATELY (Task 11 -- they draw on different acquisition
    budgets, never combined into one scored list), plus a short third panel
    for the class-A void regions (Task 9), labeled by descriptor range
    rather than a bin name."""
    acquire_cluster = report["acquire_cluster"].head(top_n)
    acquire_rows = report["acquire_rows_within_cluster"].sort_values(
        ["on_frontier", "deficit", "leverage"], ascending=[False, False, False]
    ).head(top_n)
    void = report["void_regions"]

    fig = plt.figure(figsize=(11.5, 14.5))
    ax_cluster = fig.add_axes([0.32, 0.68, 0.62, 0.23])
    ax_rows = fig.add_axes([0.32, 0.29, 0.62, 0.34])
    ax_void = fig.add_axes([0.06, 0.05, 0.88, 0.17])

    _plot_remedy_panel(ax_cluster, acquire_cluster, "acquire_cluster -- needs a protein not currently in the dataset")
    _plot_remedy_panel(ax_rows, acquire_rows, "acquire_rows_within_cluster -- new formulations of an existing protein")

    if len(void) == 0:
        ax_void.text(0.5, 0.5, "no void regions", ha="center", va="center", transform=ax_void.transAxes)
    else:
        rep_cols = [c[:-4] for c in void.columns if c.endswith("_min")]
        void_labels = []
        for _, row in void.iterrows():
            parts = [f"{a} [{row[f'{a}_min']:.2g}, {row[f'{a}_max']:.2g})" for a in rep_cols]
            void_labels.append(f"near {row['nearest_protein']}: " + ", ".join(parts))
        top_void = void.iloc[::-1]
        void_labels = void_labels[::-1]
        ax_void.barh(range(len(top_void)), top_void["void_score"], color=CAT_SLOT_2, height=0.6)
        ax_void.set_yticks(range(len(top_void)))
        ax_void.set_yticklabels(void_labels, fontsize=7.5)
        ax_void.set_xlabel("void_score = nearest_neighbour_distance x response_leverage (between_protein scope)", fontsize=8)
    ax_void.set_title("class-A void regions (Task 9 -- descriptor-space acquisition targets, not bin gaps)", fontsize=12, fontweight="600", loc="left")

    leverage_mode = report["acquire_cluster"].attrs.get("leverage_mode", "?")
    fig.suptitle(f"P5' -- Gap report by remedy (leverage_mode={leverage_mode})", fontsize=15, fontweight="600", x=0.06, y=0.99, ha="left", va="top")
    _save(fig, "p5_gap_report.png", out_dir)


# =============================================================================
# P6b -- "Rows lie, but n_eff isn't cross-bin comparable either" (Task 12.3
# / C-13; replaces P6)
# =============================================================================


def plot_p6b_rows_lie(df: pd.DataFrame, out_dir: str = OUTPUT_DIR) -> None:
    """n_rows vs n_eff on log-log, marker area proportional to Kish balance
    (Task 12.3) -- the vertical distance from the identity line is the
    amount by which a raw row-count census overstates a bin, and marker
    size shows why n_eff alone isn't cross-bin comparable. The all-12 vs
    excl-poly-hIgG pair is drawn as its own connected, annotated pair so the
    non-monotonicity (fewer rows, HIGHER n_eff, because balance improves) is
    the plot's visual subject, not a trap a reader has to compute."""
    points = []
    for axis in list(cov.CLASS_A_AXES) + [cov.CLASS_A_CATEGORICAL_AXIS]:
        m = cov.class_a_marginal(df, axis)
        points.append(m[["n_rows", "n_eff", "balance"]])
    for axis in cov.CLASS_B_AXES:
        b = cov.class_b_levels(df, axis)
        points.append(b[["n_rows", "n_eff", "balance"]])
    pts = pd.concat(points, ignore_index=True)
    pts = pts.loc[(pts["n_rows"] > 0) & (pts["n_eff"] > 0) & pts["balance"].notna()]

    non_placebo = df.loc[~cov._is_placebo(df["Protein_type"])].copy()
    bc_all = cov.count_bin(non_placebo, pd.Series(True, index=non_placebo.index), scope=cov.SCOPE_BETWEEN_PROTEIN, icc=cov.ICC_BETWEEN_PROTEIN)
    excl = non_placebo.loc[cov._normalize(non_placebo["Protein_type"]) != "poly-higg"]
    bc_excl = cov.count_bin(excl, pd.Series(True, index=excl.index), scope=cov.SCOPE_BETWEEN_PROTEIN, icc=cov.ICC_BETWEEN_PROTEIN)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    lims = [0.8, float(max(pts["n_rows"].max(), pts["n_eff"].max(), bc_all.n_rows) * 1.3)]
    ax.plot(lims, lims, color=INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=1, label="identity (n_eff = n_rows)")

    sizes = 24 + 140 * pts["balance"].clip(0, 1)
    ax.scatter(pts["n_rows"], pts["n_eff"], s=sizes, color=CAT_SLOT_1, alpha=0.6, edgecolors="white", linewidths=0.6, zorder=2, label="class-A/B bins (size ~ balance)")

    ax.plot([bc_all.n_rows, bc_excl.n_rows], [bc_all.n_eff, bc_excl.n_eff], color=CAT_SLOT_2, linewidth=1.4, zorder=3, marker="o", markersize=9, markeredgecolor="white")
    ax.annotate(
        f"all 12 clusters\nn_rows={bc_all.n_rows}, n_eff={bc_all.n_eff:.2f}\nbalance={bc_all.balance:.2f}",
        (bc_all.n_rows, bc_all.n_eff), textcoords="offset points", xytext=(10, -28), fontsize=8, color=INK_PRIMARY, zorder=4,
    )
    ax.annotate(
        f"excl. poly-hIgG (11 clusters)\nn_rows={bc_excl.n_rows}, n_eff={bc_excl.n_eff:.2f}\nbalance={bc_excl.balance:.2f}",
        (bc_excl.n_rows, bc_excl.n_eff), textcoords="offset points", xytext=(-140, 14), fontsize=8, color=INK_PRIMARY, zorder=4,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("n_rows (row scope)")
    ax.set_ylabel("n_eff (between_protein scope, ICC-corrected)")
    ax.set_title('P6b -- n_eff is not monotone in n_rows without balance alongside it', fontsize=14, fontweight="600", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    _save(fig, "p6b_rows_lie.png", out_dir)


# =============================================================================
# P9 -- Deficit/leverage frontier view (Task 14; adapted from the plan's
# "leverage ECDF with the LEVERAGE_MIN_IQR band" -- that band no longer
# exists (Task 10/16 replaced the IQR guard with a rank-based one that only
# applies to a scalar score, and the default is now a Pareto frontier with
# no scalar at all). The (deficit, leverage) scatter is the more direct
# "is leverage doing discriminating work, at a glance" view for a frontier:
# a saturated leverage collapses to a near-vertical smear; genuine
# discrimination fans the points out across the leverage axis at every
# deficit level.
# =============================================================================


def plot_p9_deficit_leverage_frontier(report: dict, out_dir: str = OUTPUT_DIR) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))
    panels = [
        ("acquire_cluster", report["acquire_cluster"]),
        ("acquire_rows_within_cluster", report["acquire_rows_within_cluster"]),
    ]
    for ax, (name, table) in zip(axes, panels):
        if len(table) == 0 or "on_frontier" not in table.columns:
            ax.text(0.5, 0.5, "no scalar leverage guard applies\n(logo_residual mode)" if len(table) else "no gaps", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name, fontsize=11, fontweight="600", loc="left")
            continue
        frontier = table.loc[table["on_frontier"]]
        rest = table.loc[~table["on_frontier"]]
        ax.scatter(rest["deficit"], rest["leverage"], s=40, color=BASELINE, alpha=0.7, edgecolors="none", label="dominated", zorder=2)
        ax.scatter(frontier["deficit"], frontier["leverage"], s=70, color=CAT_SLOT_1, edgecolors="white", linewidths=0.8, label="Pareto frontier", zorder=3)
        ax.set_xlabel("deficit")
        ax.set_ylabel("leverage")
        ax.set_title(f"{name} (n={len(table)}, frontier={len(frontier)})", fontsize=11, fontweight="600", loc="left")
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.grid(True, linestyle="-", alpha=1.0, zorder=0)

    fig.suptitle(
        "P9 -- deficit x leverage frontier (replaces the retired IQR-band ECDF; see coverage_plots.py module note)",
        fontsize=13, fontweight="600", x=0.02, ha="left",
    )
    _save(fig, "p9_deficit_leverage_frontier.png", out_dir)


def generate_all(df: pd.DataFrame, report: dict, out_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plot_p1_logo_support_matrix(df, out_dir)
    plot_p2_descriptor_occupancy(df, out_dir=out_dir)
    plot_p2b_whitened_void_space(df, out_dir=out_dir)
    plot_p3_concentration_grid(df, out_dir)
    plot_p4_response_coverage(df, out_dir)
    plot_p5_gap_report(report, out_dir=out_dir)
    plot_p6b_rows_lie(df, out_dir)
    plot_p9_deficit_leverage_frontier(report, out_dir=out_dir)
