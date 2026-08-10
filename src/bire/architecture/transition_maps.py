"""
BIRE OS Architecture — Transition Maps

Contains architecture diagrams focused on:
- event transition governance
- pre-event to post-event handoff
"""

from bire.architecture.utils import (
    setup_figure,
    draw_box,
    draw_arrow,
    draw_boundary_line,
)


# =========================================
# 52.6 — Event Transition Governance Map
# =========================================

def plot_event_transition_governance(figsize=(18, 12)):
    """
    Chapter 52.6 — Event Transition Governance Map

    Visualizes how BIRE OS transitions from pre-event surveillance
    into post-event operational management.
    """

    fig, ax = setup_figure(
        figsize=figsize,
        xlim=(0, 14),
        ylim=(0, 10),
    )

    # Title
    ax.text(
        7,
        9.6,
        "BIRE OS — Event Transition Governance Map",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    ax.text(
        7,
        9.25,
        "How BIRE OS crosses from pre-event surveillance into post-event operational management",
        ha="center",
        va="center",
        fontsize=10,
    )

    # Pre-event side
    draw_box(ax, 0.7, 7.4, 2.3, 0.9, "Pre-Event Surveillance\nContinuous Monitoring")
    draw_box(ax, 3.5, 7.4, 2.1, 0.9, "WATCH\nEarly Awareness")
    draw_box(ax, 6.2, 7.4, 2.1, 0.9, "ESCALATE\nRising Concern")
    draw_box(ax, 8.9, 7.4, 2.1, 0.9, "URGENT\nHigh Concern")

    draw_arrow(ax, 3.0, 7.85, 3.5, 7.85)
    draw_arrow(ax, 5.6, 7.85, 6.2, 7.85)
    draw_arrow(ax, 8.3, 7.85, 8.9, 7.85)

    # Event gate
    draw_box(ax, 11.7, 7.1, 1.8, 1.3, "EVENT\nTRANSITION\nGATE", fontsize=11)
    draw_arrow(ax, 11.0, 7.85, 11.7, 7.85, "event_now = 1")

    # Transition governance core
    draw_box(
        ax,
        4.8,
        5.4,
        4.4,
        1.1,
        "Transition Governance Core\nConfirm → Handoff → Activate Post-Event Logic",
        fontsize=11,
    )

    draw_arrow(ax, 12.5, 7.1, 9.2, 6.1, "confirmed event", curve=-0.15)
    draw_arrow(ax, 9.2, 5.95, 9.9, 4.8, "post-event activation")

    # Governance inputs
    draw_box(ax, 0.7, 5.3, 2.3, 0.9, "IBPIP\nBaseline Context")
    draw_box(ax, 0.7, 4.0, 2.3, 0.9, "BIRE-FI\nTrajectory Context")
    draw_box(ax, 0.7, 2.7, 2.3, 0.9, "BMS\nCare Mode Context")
    draw_box(ax, 0.7, 1.4, 2.3, 0.9, "GSS / Policy Registry\nSuppression + Rules")

    draw_arrow(ax, 3.0, 5.75, 4.8, 5.95, "patient context")
    draw_arrow(ax, 3.0, 4.45, 4.8, 5.75, "trajectory")
    draw_arrow(ax, 3.0, 3.15, 4.8, 5.55, "mode")
    draw_arrow(ax, 3.0, 1.85, 4.8, 5.45, "policy")

    # Post-event side
    draw_box(ax, 9.8, 4.0, 2.4, 0.9, "POST-EVENT MONITOR\nStabilization Watch")
    draw_box(ax, 9.8, 2.8, 2.4, 0.9, "VOLATILE MONITOR\nUnstable Recovery")
    draw_box(ax, 9.8, 1.6, 2.4, 0.9, "RE-ESCALATE\nRenewed Worsening")
    draw_box(ax, 9.8, 0.4, 2.4, 0.9, "CRITICAL\nSevere Post-Event Concern")

    draw_arrow(ax, 11.0, 4.0, 11.0, 3.7, "unstable")
    draw_arrow(ax, 11.0, 2.8, 11.0, 2.5, "worsening")
    draw_arrow(ax, 11.0, 1.6, 11.0, 1.3, "severe")

    # Recovery / reintegration path
    draw_box(ax, 5.0, 2.8, 2.8, 0.9, "Recovery Evaluation\nIBPIP + BIRE-FI Agreement")
    draw_box(ax, 5.0, 1.4, 2.8, 0.9, "Reintegration Eligibility\nReturn Toward Pre-Event Surveillance")

    draw_arrow(ax, 9.8, 4.45, 7.8, 3.25, "stabilizing", curve=0.1)
    draw_arrow(ax, 6.4, 2.8, 6.4, 2.3, "recovery confirmed")
    draw_arrow(ax, 5.0, 1.85, 2.0, 7.4, "reintegrate to surveillance", curve=0.35)

    # Clinician + audit layer
    draw_box(ax, 12.4, 4.0, 1.3, 0.9, "Clinician\nView", fontsize=9)
    draw_box(ax, 12.4, 2.2, 1.3, 0.9, "Audit\nLog", fontsize=9)

    draw_arrow(ax, 12.2, 4.45, 12.4, 4.45, "curated")
    draw_arrow(ax, 12.2, 2.05, 12.4, 2.65, "trace")

    draw_arrow(ax, 9.2, 5.4, 12.4, 4.45, "handoff summary", curve=-0.1)
    draw_arrow(ax, 9.2, 5.4, 12.4, 2.65, "transition record", curve=0.1)

    # Boundary label
    draw_boundary_line(
        ax,
        x=9.45,
        y1=0.3,
        y2=8.7,
        label="Event Boundary",
    )

    # Footer
    ax.text(
        7,
        0.05,
        "Doctrine: An event is not the end of intelligence — it is the bridge from prediction into post-event lifecycle management.",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    fig.tight_layout()
    return fig, ax