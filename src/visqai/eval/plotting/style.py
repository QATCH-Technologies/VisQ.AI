"""
style.py
========
One house style, merged from two independently-maintained divergent copies:
ibal_parity_test.py's _mpl()/_apply_style() and learning_curve_ibal.py's
apply_base_style()/_style_axis(). learning_curve_ibal.py's palette is used as
canonical (it's the richer superset -- more accent colors, FONT_MAIN, and a
more complete rcParams set including legend/text colors). This IS a visible
change from ibal_parity_test.py's exact shades (e.g. its C_DEEP_BLUE was
"#1a85ad" vs this module's "#2596be") -- cosmetic only, not a correctness
change, but noting it since a side-by-side of old vs new PNGs will show it.
"""

from __future__ import annotations

import logging

# VisQ brand palette (canonical -- from learning_curve_ibal.py, the superset).
C_DEEP_BLUE = "#2596be"
C_BRIGHT_BLUE = "#13B5F0"
C_CYAN_MED = "#4EC4EB"
C_CYAN_PALE = "#8DD9F7"
C_GREEN = "#4caf50"
C_ORANGE = "#ff9800"
C_TEXT = "#24292f"
C_MUTED = "#6b7280"
C_BORDER = "#d1d5db"
C_BORDER_LT = "#e5e7eb"
C_BG_LIGHT = "#f3f4f6"
C_BG_LIGHTEST = "#f9fafb"
C_WHITE = "#ffffff"
C_PURPLE = "#9b59b6"
C_RED_SOFT = "#e74c3c"
C_ACCENT = "#d95f3b"  # from ibal_parity_test.py, used for predicted-curve accent
C_CONTEXT = "#f39c12"  # from ibal_parity_test.py, marks context points in parity plots

FONT_MAIN = "DejaVu Sans"


def mpl():
    """Lazy matplotlib import (Agg backend) -- keeps selection-only code paths
    fast by not importing matplotlib unless a plot is actually rendered."""
    import matplotlib

    matplotlib.use("Agg")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    return plt, ticker, Line2D


def apply_style(plt):
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
            "grid.alpha": 1.0,
            "text.color": C_TEXT,
            "legend.facecolor": C_WHITE,
            "legend.edgecolor": C_BORDER,
            "legend.labelcolor": C_TEXT,
            "font.family": FONT_MAIN,
            "font.size": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axis(ax, ticker_mod, spine_color=C_BORDER):
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(0.9)
    ax.tick_params(length=3, width=0.8)
    ax.grid(True, which="major", axis="both", zorder=0)
    ax.yaxis.set_minor_locator(ticker_mod.AutoMinorLocator(2))
    ax.set_facecolor(C_BG_LIGHTEST)
