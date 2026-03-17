"""
draw_cnp_arch.py
================
Generates a publication-quality architecture diagram for CrossSampleCNP,
styled after the "Attention is All You Need" transformer figures.

Outputs: viz_outputs/cnp_architecture_diagram.png  (300 dpi)
         viz_outputs/cnp_architecture_diagram.pdf   (vector)

Usage:
    python draw_cnp_arch.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

os.makedirs("viz_outputs", exist_ok=True)

# ─── Palette ──────────────────────────────────────────────────────────────────
BG = "#F7F8FC"
C_INPUT = ("#E8EDF5", "#4A6FA5")  # fill, border
C_ENC = ("#DCF0FF", "#2378C3")
C_POOL = ("#D6F5E8", "#1A9E6A")
C_LATENT = ("#FFF8DC", "#C8960C")
C_DEC = ("#EDE0FF", "#7A3FC0")
C_OUT = ("#FFF0D6", "#D4820A")
C_ARROW = "#555E6E"
C_ANNOT = "#7A8499"
C_LANE = "#E4E8F0"
C_LANETXT = "#A0AABF"

FONT = "DejaVu Sans"

# ─── Canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis("off")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def rbox(
    ax,
    x,
    y,
    w,
    h,
    label,
    sublabel="",
    fill=C_INPUT,
    radius=0.25,
    fontsize=10.5,
    subfontsize=8.5,
    bold=True,
):
    """Draw a rounded-rectangle block."""
    fc, ec = fill
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
        zorder=3,
    )
    ax.add_patch(box)
    # subtle inner shadow line at top
    ax.plot(
        [x - w / 2 + radius, x + w / 2 - radius],
        [y + h / 2 - 0.045, y + h / 2 - 0.045],
        color=ec,
        alpha=0.25,
        linewidth=1.2,
        zorder=4,
    )

    lbl_y = y + (0.13 if sublabel else 0)
    weight = "bold" if bold else "normal"
    ax.text(
        x,
        lbl_y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        fontfamily=FONT,
        color="#1A202C",
        zorder=5,
    )
    if sublabel:
        ax.text(
            x,
            y - 0.22,
            sublabel,
            ha="center",
            va="center",
            fontsize=subfontsize,
            fontfamily=FONT,
            color=ec,
            alpha=0.85,
            zorder=5,
        )


def mini_layers(ax, x, y, w, h, n, fill, radius=0.12):
    """Draw stacked mini-layer indicators inside a block."""
    fc, ec = fill
    step = (h * 0.55) / max(n, 1)
    for i in range(n):
        ly = y - h * 0.25 + i * step
        mini = FancyBboxPatch(
            (x - w * 0.35, ly),
            w * 0.70,
            step * 0.72,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=0.8,
            edgecolor=ec,
            facecolor=fc,
            alpha=0.45,
            zorder=4,
        )
        ax.add_patch(mini)


def arrow(
    ax, x0, y0, x1, y1, label="", color=C_ARROW, connectionstyle="arc3,rad=0.0", lw=1.6
):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        zorder=2,
    )
    ax.add_patch(arr)
    if label:
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        ax.text(
            mx + 0.08,
            my,
            label,
            ha="left",
            va="center",
            fontsize=7.5,
            fontfamily=FONT,
            color=C_ANNOT,
            fontstyle="italic",
            zorder=6,
        )


def curved_arrow(ax, x0, y0, x1, y1, rad=0.3, label="", color=C_ARROW, lw=1.6):
    cs = f"arc3,rad={rad}"
    arrow(ax, x0, y0, x1, y1, label=label, color=color, connectionstyle=cs, lw=lw)


def phase_label(ax, x, y1, y2, label, color=C_LANETXT):
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y2),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.0, linestyle="dashed"),
    )
    ax.text(
        x - 0.12,
        (y1 + y2) / 2,
        label,
        ha="right",
        va="center",
        fontsize=8,
        fontfamily=FONT,
        color=color,
        rotation=90,
        fontweight="bold",
    )


def dim_tag(ax, x, y, text, color=C_ANNOT):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        fontfamily=FONT,
        color=color,
        fontstyle="italic",
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor=color,
            linewidth=0.8,
            alpha=0.85,
        ),
        zorder=6,
    )


# ─── Phase swim-lane backgrounds ──────────────────────────────────────────────
lanes = [
    (9.85, 10.95, "ENCODING"),
    (6.85, 9.75, "AGGREGATION"),
    (1.05, 6.75, "DECODING"),
]
for y_lo, y_hi, lbl in lanes:
    ax.fill_between(
        [0.45, 13.55], [y_lo, y_lo], [y_hi, y_hi], color=C_LANE, alpha=0.38, zorder=0
    )
    ax.text(
        13.45,
        (y_lo + y_hi) / 2,
        lbl,
        ha="right",
        va="center",
        fontsize=7.5,
        fontfamily=FONT,
        color=C_LANETXT,
        fontweight="bold",
        alpha=0.7,
    )

# ─── Title ────────────────────────────────────────────────────────────────────
ax.text(
    7,
    10.72,
    "CrossSampleCNP",
    ha="center",
    va="center",
    fontsize=17,
    fontweight="bold",
    fontfamily=FONT,
    color="#1A202C",
)
ax.text(
    7,
    10.48,
    "Conditional Neural Process  ·  Protein Viscosity Predictor",
    ha="center",
    va="center",
    fontsize=9.5,
    fontfamily=FONT,
    color=C_ANNOT,
)

# ─── Column headers ───────────────────────────────────────────────────────────
for cx, lbl in [(3.5, "CONTEXT  PATH"), (10.5, "QUERY  PATH")]:
    ax.text(
        cx,
        10.18,
        lbl,
        ha="center",
        va="center",
        fontsize=8.2,
        fontfamily=FONT,
        color=C_ANNOT,
        fontweight="bold",
    )
    ax.plot(
        [cx - 1.6, cx + 1.6], [10.04, 10.04], color=C_ANNOT, linewidth=0.8, alpha=0.5
    )

# ─── Vertical divider ─────────────────────────────────────────────────────────
ax.plot(
    [7, 7],
    [1.3, 9.95],
    color=C_LANE,
    linewidth=1.2,
    linestyle="--",
    alpha=0.7,
    zorder=1,
)

# ══════════════════════════════════════════════════════════════════════════════
# ENCODING PHASE — both inputs
# ══════════════════════════════════════════════════════════════════════════════

# Context Input
rbox(
    ax,
    3.5,
    9.45,
    3.8,
    0.62,
    "Context Input",
    sublabel="concentration  ·  log η  ·  static features  [B × N_ctx × (2 + D)]",
    fill=C_INPUT,
    fontsize=10,
)

# Query Input
rbox(
    ax,
    10.5,
    9.45,
    3.8,
    0.62,
    "Query Input",
    sublabel="shear rate  ·  static features  [B × N_q × (1 + D)]",
    fill=C_INPUT,
    fontsize=10,
)

# ── Encoder MLP ───────────────────────────────────────────────────────────────
rbox(
    ax,
    3.5,
    8.05,
    3.8,
    1.52,
    "Encoder MLP",
    sublabel="Linear(2+D → H)  ·  ReLU  ·  Dropout",
    fill=C_ENC,
    fontsize=11,
)
mini_layers(ax, 3.5, 8.05, 3.8, 1.52, 3, C_ENC)

# sub-annotations inside encoder
for i, (ly, txt) in enumerate(
    [
        (8.52, "Linear  (2+D, H)  →  ReLU"),
        (8.08, "Linear  (H, H)    →  ReLU"),
        (7.64, "Linear  (H, latent_dim)"),
    ]
):
    ax.text(
        3.5,
        ly,
        txt,
        ha="center",
        va="center",
        fontsize=7.2,
        fontfamily=FONT,
        color=C_ENC[1],
        alpha=0.9,
        zorder=5,
    )

# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION PHASE
# ══════════════════════════════════════════════════════════════════════════════

# Attention Pooler
rbox(
    ax,
    3.5,
    6.18,
    3.8,
    2.12,
    "Attention Pooler",
    sublabel="Learnable query  ·  n_heads = 4",
    fill=C_POOL,
    fontsize=11,
)

# internal structure
for ly, txt in [
    (6.72, "Learned Query  q ∈ ℝ^latent"),
    (6.38, "MultiheadAttention  (Q, K, V)"),
    (6.04, "LayerNorm"),
]:
    ax.text(
        3.5,
        ly,
        txt,
        ha="center",
        va="center",
        fontsize=7.5,
        fontfamily=FONT,
        color=C_POOL[1],
        alpha=0.95,
        zorder=5,
    )

# MHA attention illustration — small dot grid
for dx in np.linspace(-0.55, 0.55, 6):
    for dy in np.linspace(-0.08, 0.08, 2):
        ax.plot(
            3.5 + dx,
            5.7 + dy,
            "o",
            markersize=2.2,
            color=C_POOL[1],
            alpha=0.35,
            zorder=5,
        )

# Latent vector r
rbox(
    ax,
    3.5,
    4.80,
    3.8,
    0.68,
    "Latent Context  r",
    sublabel="[B × latent_dim]   ·   protein-level summary",
    fill=C_LATENT,
    fontsize=11,
    bold=True,
)

# r broadcast annotation
ax.annotate(
    "broadcast\nacross queries",
    xy=(5.42, 4.80),
    xytext=(6.4, 4.80),
    ha="left",
    va="center",
    fontsize=7.2,
    fontfamily=FONT,
    color=C_ANNOT,
    fontstyle="italic",
    arrowprops=dict(
        arrowstyle="->", color=C_ANNOT, lw=0.9, connectionstyle="arc3,rad=-0.2"
    ),
)

# Query path just holds position (no transform needed)
rbox(
    ax,
    10.5,
    6.80,
    3.8,
    0.62,
    "Query Features  (unchanged)",
    sublabel="shear rate  ·  static  [B × N_q × (1+D)]",
    fill=C_INPUT,
    fontsize=9.5,
    bold=False,
)

# ══════════════════════════════════════════════════════════════════════════════
# DECODING PHASE  — merge + decoder
# ══════════════════════════════════════════════════════════════════════════════

# Concatenate node
rbox(
    ax,
    7,
    3.62,
    4.0,
    0.72,
    "Concatenate",
    sublabel="shear  ‖  static  ‖  r_expanded       [B × N_q × (1+D+latent_dim)]",
    fill=C_DEC,
    fontsize=10.5,
    radius=0.18,
)

# Decoder MLP
rbox(
    ax,
    7,
    2.22,
    4.0,
    1.52,
    "Decoder MLP",
    sublabel="Linear(1+D+L → H)  ·  ReLU  ·  Dropout",
    fill=C_DEC,
    fontsize=11,
)

for ly, txt in [
    (2.68, "Linear  (1+D+L, H)  →  ReLU"),
    (2.24, "Linear  (H, H)      →  ReLU"),
    (1.80, "Linear  (H, 1)"),
]:
    ax.text(
        7,
        ly,
        txt,
        ha="center",
        va="center",
        fontsize=7.2,
        fontfamily=FONT,
        color=C_DEC[1],
        alpha=0.9,
        zorder=5,
    )

# Output
rbox(
    ax,
    7,
    1.18,
    4.0,
    0.68,
    "log η̂  (predicted log-viscosity)",
    sublabel="[B × N_q × 1]   one value per shear rate per query",
    fill=C_OUT,
    fontsize=11,
    bold=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS
# ══════════════════════════════════════════════════════════════════════════════

# Context Input → Encoder
arrow(ax, 3.5, 9.14, 3.5, 8.82)
dim_tag(ax, 4.52, 8.98, "[B × N_ctx × (2+D)]")

# Encoder → Pooler
arrow(ax, 3.5, 7.29, 3.5, 7.25)
dim_tag(ax, 4.52, 7.27, "[B × N_ctx × L]")

# Pooler → r
arrow(ax, 3.5, 5.12, 3.5, 5.14)
dim_tag(ax, 4.52, 5.13, "[B × L]")

# r → Concat (curved left-to-center)
curved_arrow(ax, 3.5, 4.46, 5.0, 3.98, rad=0.25, color=C_LATENT[1], lw=2.0)

# Query Input → Query features passthrough
arrow(ax, 10.5, 9.14, 10.5, 7.11)
dim_tag(ax, 11.52, 8.12, "[B × N_q × (1+D)]")

# Query features → Concat (curved right-to-center)
curved_arrow(ax, 10.5, 6.49, 9.0, 3.98, rad=-0.25, color=C_INPUT[1], lw=1.8)

# Concat → Decoder
arrow(ax, 7, 3.26, 7, 2.99)

# Decoder → Output
arrow(ax, 7, 1.46, 7, 1.52)

# ── Key dimension labels on merge arrows ──────────────────────────────────────
ax.text(
    5.3,
    3.72,
    "r_expanded\n[B×N_q×L]",
    ha="center",
    va="center",
    fontsize=7,
    fontfamily=FONT,
    color=C_LATENT[1],
    fontstyle="italic",
    bbox=dict(
        boxstyle="round,pad=0.15",
        facecolor="#FFFFF0",
        edgecolor=C_LATENT[1],
        linewidth=0.8,
        alpha=0.9,
    ),
    zorder=7,
)

ax.text(
    8.9,
    3.72,
    "query\n[B×N_q×(1+D)]",
    ha="center",
    va="center",
    fontsize=7,
    fontfamily=FONT,
    color=C_INPUT[1],
    fontstyle="italic",
    bbox=dict(
        boxstyle="round,pad=0.15",
        facecolor="white",
        edgecolor=C_INPUT[1],
        linewidth=0.8,
        alpha=0.9,
    ),
    zorder=7,
)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_INPUT, "Input / passthrough"),
    (C_ENC, "Encoder MLP"),
    (C_POOL, "Attention Pooler"),
    (C_LATENT, "Latent representation  r"),
    (C_DEC, "Decoder MLP"),
    (C_OUT, "Output"),
]
lx, ly = 0.62, 3.5
ax.text(
    lx,
    ly + 0.28,
    "LEGEND",
    fontsize=7.5,
    fontfamily=FONT,
    color=C_ANNOT,
    fontweight="bold",
)
for i, (fill, label) in enumerate(legend_items):
    fc, ec = fill
    by = ly - 0.38 * i
    patch = FancyBboxPatch(
        (lx, by - 0.1),
        0.32,
        0.22,
        boxstyle="round,pad=0,rounding_size=0.06",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
        zorder=5,
    )
    ax.add_patch(patch)
    ax.text(
        lx + 0.42,
        by + 0.01,
        label,
        ha="left",
        va="center",
        fontsize=7.8,
        fontfamily=FONT,
        color="#2D3748",
        zorder=6,
    )

# ── Hyperparameter box ────────────────────────────────────────────────────────
hp_x, hp_y = 0.62, 0.98
ax.text(
    hp_x,
    hp_y,
    "D = static_dim = 85    H = hidden_dim = 128    L = latent_dim = 128",
    ha="left",
    va="center",
    fontsize=7.8,
    fontfamily=FONT,
    color=C_ANNOT,
    fontstyle="italic",
    bbox=dict(
        boxstyle="round,pad=0.28", facecolor="white", edgecolor=C_LANE, linewidth=1.0
    ),
)

# ── Footnote ──────────────────────────────────────────────────────────────────
ax.text(
    7,
    0.32,
    "N_ctx = number of context observations  ·  N_q = number of query shear rates  ·  B = batch size",
    ha="center",
    va="center",
    fontsize=7.5,
    fontfamily=FONT,
    color=C_ANNOT,
)

# ─── Save ─────────────────────────────────────────────────────────────────────
for fmt, dpi in [("png", 300), ("pdf", None)]:
    path = f"viz_outputs/cnp_architecture_diagram.{fmt}"
    kw = dict(bbox_inches="tight", facecolor=BG)
    if dpi:
        kw["dpi"] = dpi
    plt.savefig(path, **kw)
    print(f"Saved → {path}")

plt.close()
