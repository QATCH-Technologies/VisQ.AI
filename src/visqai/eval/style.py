from __future__ import annotations

import logging

from visqai.constants import (
    C_WHITE,
    C_BG_LIGHTEST,
    C_BORDER,
    C_MUTED,
    C_BORDER_LT,
    C_TEXT,
    FONT_MAIN,
)


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
