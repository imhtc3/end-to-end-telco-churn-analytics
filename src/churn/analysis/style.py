"""Shared plotting style so every figure in reports/ looks like it belongs together."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#d9d8d3"

BLUE = "#2a78d6"     # retained / primary series
ORANGE = "#eb6834"   # churned
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, "#008300", "#4a3aa7", "#e34948"]

CHURN_COLORS = {0: BLUE, 1: ORANGE, "No": BLUE, "Yes": ORANGE, "Retained": BLUE, "Churned": ORANGE}

SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc2ef", BLUE, "#153e70"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ["#fdf0ea", "#f5b394", ORANGE, "#8a3212"])
DIVERGING = LinearSegmentedColormap.from_list("div_bo", [BLUE, "#f1f0ed", ORANGE])


def apply() -> None:
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def titled(ax, title: str, sub: str | None = None) -> None:
    ax.set_title(title, pad=24 if sub else 12)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")
