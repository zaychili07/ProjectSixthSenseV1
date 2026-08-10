"""
BIRE OS Architecture — Circulation Maps

Contains architecture diagrams focused on:
- operational queue circulation
- continuous intelligence circulation
"""

from bire.architecture.utils import (
    setup_figure,
    draw_box,
    draw_arrow,
    add_title,
    add_footer,
)


# =========================================
# 52.3 — Operational Queue Orchestration Flow Map
# =========================================

def plot_operational_queue_flow(figsize=(16, 9)):
    """
    Plot BIRE OS operational queue orchestration flow.

    Visualizes how patients circulate between:
    - background surveillance
    - PSR prioritization
    - active queue
    - clinician review
    - reintegration / de-escalation
    - re-escalation
    """

    fig, ax = setup_figure(
        figsize=figsize,
        xlim=(0, 16),
        ylim=(0, 10),
    )

    add_title(
        ax,
        title="BIRE OS — Operational Queue Orchestration Flow",
        subtitle="Queues continuously rebalance attention while preserving full patient visibility.",
        title_y=9.4,
        subtitle_y=8.8,
    )

    # Core queue states
    draw_box(
        ax, 0.8, 6.2, 3.0, 1.1,
        "BACKGROUND_SURVEILLANCE\nAll patients remain visible\nLower-intensity monitoring",
    )

    draw_box(
        ax, 4.7, 6.2, 3.0, 1.1,
        "PSR PRIORITIZATION\nRanks attention need\nNot patient value",
    )

    draw_box(
        ax, 8.6, 6.2, 3.0, 1.1,
        "ACTIVE_QUEUE\nSuggested closer review\nHigh operational attention",
    )

    draw_box(
        ax, 12.5, 6.2, 3.0, 1.1,
        "CLINICIAN VIEW\nClear movement summary\nNo internal pressure score",
    )

    draw_arrow(ax, 3.8, 6.75, 4.7, 6.75)
    draw_arrow(ax, 7.7, 6.75, 8.6, 6.75)
    draw_arrow(ax, 11.6, 6.75, 12.5, 6.75)

    # Escalation / re-escalation logic
    draw_box(
        ax, 4.7, 3.9, 3.0, 1.1,
        "ESCALATION SIGNALS\nWorsening trajectory\nRising PSR / instability",
    )

    draw_box(
        ax, 8.6, 3.9, 3.0, 1.1,
        "QUEUE ENTRY / REFRESH\nOpen slots filled\nHigher PSR patients surfaced",
    )

    draw_arrow(ax, 6.2, 5.0, 6.2, 6.2)
    draw_arrow(ax, 7.7, 4.45, 8.6, 4.45)
    draw_arrow(ax, 10.1, 5.0, 10.1, 6.2)

    # Reintegration / decompression logic
    draw_box(
        ax, 4.7, 1.6, 3.0, 1.1,
        "REINTEGRATION LOGIC\nIBPIP + BIRE-FI agreement\nStable / no rising trend",
    )

    draw_box(
        ax, 8.6, 1.6, 3.0, 1.1,
        "DECOMPRESSION\nWATCH / SUPPRESSED_WATCH\nReduced queue burden",
    )

    draw_arrow(ax, 10.1, 6.2, 10.1, 2.7)
    draw_arrow(ax, 8.6, 2.15, 7.7, 2.15)
    draw_arrow(ax, 4.7, 2.15, 2.3, 6.2)

    # Audit and governance
    draw_box(
        ax, 12.5, 1.6, 3.0, 1.1,
        "AUDIT LOG\nEvery movement recorded\nReason + evidence preserved",
    )

    draw_arrow(ax, 13.9, 6.2, 13.9, 2.7)

    # Feedback loop
    draw_arrow(ax, 14.0, 1.6, 14.0, 0.8)
    draw_arrow(ax, 14.0, 0.8, 2.3, 0.8)
    draw_arrow(ax, 2.3, 0.8, 2.3, 6.2)

    add_footer(
        ax,
        text=(
            "Queue orchestration keeps patients visible, surfaces higher-priority patients, "
            "reintegrates stabilizing patients, and logs movement for trust and traceability."
        ),
        y=0.35,
        fontsize=10,
    )

    fig.tight_layout()
    return fig, ax


# =========================================
# 52.5 — Operational Intelligence Circulation Map
# =========================================

def plot_operational_intelligence_circulation(figsize=(18, 12)):
    """
    Chapter 52.5 — Operational Intelligence Circulation Map

    Visualizes BIRE OS as a continuously circulating intelligence ecosystem
    rather than a one-time prediction or alerting pipeline.
    """

    fig, ax = setup_figure(
        figsize=figsize,
        xlim=(0, 14),
        ylim=(0, 10),
    )

    # Manual title because this map uses x-center 7 instead of 8
    ax.text(
        7,
        9.6,
        "BIRE OS — Operational Intelligence Circulation Map",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    ax.text(
        7,
        9.25,
        "Continuous surveillance, reassessment, routing, reintegration, and clinician-facing awareness",
        ha="center",
        va="center",
        fontsize=10,
    )

    # Main circulation ring
    draw_box(ax, 0.8, 7.3, 2.3, 0.9, "Patient Data Stream\nVitals / Labs / Context")
    draw_box(ax, 3.6, 7.3, 2.3, 0.9, "Temporal Feature Layer\nLag / Delta / Rolling")
    draw_box(ax, 6.4, 7.3, 2.3, 0.9, "Risk Intelligence\nMulti-Horizon Scores")
    draw_box(ax, 9.2, 7.3, 2.3, 0.9, "Trajectory Intelligence\nBIRE-FI")
    draw_box(ax, 11.8, 5.6, 1.8, 0.9, "Baseline Review\nIBPIP")

    draw_box(ax, 9.2, 3.8, 2.3, 0.9, "Suppression Review\nGSS")
    draw_box(ax, 6.4, 3.8, 2.3, 0.9, "Operational Routing\nDecision Core")
    draw_box(ax, 3.6, 3.8, 2.3, 0.9, "Queue Rebalance\nActive / Background")
    draw_box(ax, 0.8, 3.8, 2.3, 0.9, "Clinician View\nCurated Awareness")

    # Ring arrows
    draw_arrow(ax, 3.1, 7.75, 3.6, 7.75)
    draw_arrow(ax, 5.9, 7.75, 6.4, 7.75)
    draw_arrow(ax, 8.7, 7.75, 9.2, 7.75)
    draw_arrow(ax, 11.5, 7.45, 12.2, 6.5, "baseline check", curve=-0.15)
    draw_arrow(ax, 12.2, 5.6, 10.7, 4.7, "recovery status", curve=-0.15)
    draw_arrow(ax, 9.2, 4.25, 8.7, 4.25)
    draw_arrow(ax, 6.4, 4.25, 5.9, 4.25)
    draw_arrow(ax, 3.6, 4.25, 3.1, 4.25)
    draw_arrow(ax, 1.9, 4.7, 1.9, 7.3, "new observations", curve=-0.25)

    # Central doctrine box
    draw_box(
        ax,
        4.8,
        5.45,
        4.4,
        1.0,
        "Continuous Intelligence Loop\nSurveil → Reassess → Route → Monitor → Reintegrate",
        fontsize=11,
    )

    # Internal influence arrows
    draw_arrow(ax, 6.9, 7.3, 6.9, 6.45, "risk refresh")
    draw_arrow(ax, 7.0, 5.45, 7.0, 4.7, "routing update")
    draw_arrow(ax, 4.75, 5.95, 3.0, 4.7, "queue pressure", curve=0.15)
    draw_arrow(ax, 9.25, 5.95, 10.2, 4.7, "suppression logic", curve=-0.15)
    draw_arrow(ax, 9.25, 5.95, 12.0, 6.0, "baseline agreement", curve=0.15)

    # State circulation layer
    draw_box(ax, 1.2, 1.7, 2.2, 0.8, "SUPPRESSED_WATCH\nBackground Loop", fontsize=9)
    draw_box(ax, 4.1, 1.7, 1.8, 0.8, "WATCH\nEarly Loop", fontsize=9)
    draw_box(ax, 6.8, 1.7, 1.8, 0.8, "ESCALATE\nConcern Loop", fontsize=9)
    draw_box(ax, 9.5, 1.7, 1.8, 0.8, "URGENT\nPriority Loop", fontsize=9)
    draw_box(ax, 11.8, 1.7, 1.7, 0.8, "Audit Trail\nMemory", fontsize=9)

    draw_arrow(ax, 4.7, 3.8, 2.3, 2.5, "background")
    draw_arrow(ax, 4.7, 3.8, 5.0, 2.5, "watch")
    draw_arrow(ax, 7.5, 3.8, 7.7, 2.5, "escalate")
    draw_arrow(ax, 7.5, 3.8, 10.4, 2.5, "urgent")
    draw_arrow(ax, 10.4, 2.5, 12.2, 2.1, "log")

    # Reversibility arrows
    draw_arrow(ax, 10.4, 1.7, 7.7, 1.7, "de-escalate", curve=-0.05)
    draw_arrow(ax, 7.7, 1.7, 5.0, 1.7, "de-escalate", curve=-0.05)
    draw_arrow(ax, 5.0, 1.7, 2.3, 1.7, "reintegrate", curve=-0.05)
    draw_arrow(ax, 5.0, 2.5, 7.7, 2.5, "re-escalate", curve=0.05)
    draw_arrow(ax, 7.7, 2.5, 10.4, 2.5, "worsen", curve=0.05)

    ax.text(
        7,
        0.65,
        "Doctrine: BIRE OS is a living operational intelligence loop — not a one-time alert generator.",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        7,
        0.35,
        (
            "Every cycle can refresh risk, suppress noise, escalate concern, "
            "de-escalate recovery, reintegrate stability, and preserve audit memory."
        ),
        ha="center",
        va="center",
        fontsize=9,
    )

    fig.tight_layout()
    return fig, ax