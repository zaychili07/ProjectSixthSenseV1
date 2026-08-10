"""
BIRE OS Architecture Utilities

Shared rendering helpers for:
- lifecycle maps
- governance maps
- subsystem diagrams
- circulation maps
- transition maps

These utilities standardize the visual language of
BIRE OS architecture cartography.
"""

import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch


# =========================================================
# Figure Setup
# =========================================================

def setup_figure(
    figsize=(18, 12),
    xlim=(0, 16),
    ylim=(0, 12),
):
    """
    Create standardized architecture figure.
    """

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.axis("off")

    return fig, ax


# =========================================================
# Draw Box
# =========================================================

def draw_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    fontsize=10,
    linewidth=1.6,
):
    """
    Draw standardized architecture box.
    """

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=linewidth,
        edgecolor="black",
        facecolor="white",
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        wrap=True,
    )


# =========================================================
# Draw Arrow
# =========================================================

def draw_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    text=None,
    curve=0.0,
    linewidth=1.4,
):
    """
    Draw standardized architecture arrow.
    """

    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=linewidth,
        color="black",
        connectionstyle=f"arc3,rad={curve}",
    )

    ax.add_patch(arr)

    if text:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.15,
            text,
            ha="center",
            va="center",
            fontsize=8,
            wrap=True,
        )


# =========================================================
# Add Title
# =========================================================

def add_title(
    ax,
    title,
    subtitle=None,
    title_y=11.4,
    subtitle_y=11.0,
):
    """
    Add standardized architecture title.
    """

    ax.text(
        8,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )

    if subtitle:
        ax.text(
            8,
            subtitle_y,
            subtitle,
            ha="center",
            va="center",
            fontsize=10,
        )


# =========================================================
# Add Footer
# =========================================================

def add_footer(
    ax,
    text,
    y=0.2,
    fontsize=10,
):
    """
    Add standardized architecture doctrine footer.
    """

    ax.text(
        8,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        wrap=True,
    )


# =========================================================
# Add Section Label
# =========================================================

def add_section_label(
    ax,
    x,
    y,
    text,
    fontsize=12,
):
    """
    Add architecture section label.
    """

    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
    )


# =========================================================
# Draw Boundary Line
# =========================================================

def draw_boundary_line(
    ax,
    x,
    y1,
    y2,
    label=None,
):
    """
    Draw lifecycle or governance boundary.
    """

    ax.plot(
        [x, x],
        [y1, y2],
        linestyle="--",
        linewidth=1.2,
        color="black",
    )

    if label:
        ax.text(
            x + 0.1,
            y2 + 0.15,
            label,
            fontsize=9,
            fontweight="bold",
            ha="left",
        )

# =========================================================
# Figure Export
# =========================================================

def save_architecture_figure(
    fig,
    path,
    dpi=300,
):
    """
    Save architecture figure artifact.

    Used for:
    - governance documentation
    - architecture memory
    - notebook exports
    - operational cartography
    """

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )