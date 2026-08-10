"""
BIRE OS Architecture — Lifecycle Maps

Contains architecture diagrams focused on:
- pre-event operational lifecycle
- post-event lifecycle intelligence
- unified pre/post lifecycle intelligence
"""

from bire.architecture.utils import (
    setup_figure,
    draw_box,
    draw_arrow,
    add_title,
    add_footer,
    add_section_label,
    draw_boundary_line,
)


# =========================================
# 52.2 — Operational State Lifecycle Map
# =========================================

def plot_operational_state_lifecycle(figsize=(16, 9)):
    """
    Chapter 52.2 — Operational State Lifecycle Map

    Visualizes the BIRE OS pre-event operational lifecycle.

    Separates:
    - operational surveillance states
    - trajectory context states
    """

    fig, ax = setup_figure(figsize=figsize, xlim=(0, 16), ylim=(0, 10))

    add_title(
        ax,
        title="BIRE OS — Pre-Event Operational Lifecycle Fluidity",
        subtitle=(
            "Operational State = where attention is placed | "
            "Trajectory Context = where the patient appears to be moving"
        ),
        title_y=9.3,
        subtitle_y=8.6,
    )

    # Operational state layer
    draw_box(ax, 0.8, 6.5, 2.4, 1.0, "SUPPRESSED_WATCH\nLow-Intensity\nSurveillance")
    draw_box(ax, 4.2, 6.5, 2.4, 1.0, "WATCH\nEarly Monitoring\nPre-Event Concern")
    draw_box(ax, 7.6, 6.5, 2.4, 1.0, "ESCALATE\nActive Review\nRising Concern")
    draw_box(ax, 11.0, 6.5, 2.4, 1.0, "URGENT\nHigh Priority\nStill Pre-Event")

    draw_arrow(ax, 3.2, 7.15, 4.2, 7.15)
    draw_arrow(ax, 6.6, 7.15, 7.6, 7.15)
    draw_arrow(ax, 10.0, 7.15, 11.0, 7.15)

    draw_arrow(ax, 4.2, 6.85, 3.2, 6.85)
    draw_arrow(ax, 7.6, 6.85, 6.6, 6.85)
    draw_arrow(ax, 11.0, 6.85, 10.0, 6.85)

    draw_arrow(ax, 5.4, 6.3, 8.8, 6.3)
    draw_arrow(ax, 8.8, 7.7, 5.4, 7.7)

    ax.text(
        8,
        5.95,
        (
            "Operational states remain reversible through IBPIP agreement, "
            "BIRE-FI trajectory agreement, PSR stabilization, and reintegration logic."
        ),
        ha="center",
        fontsize=9,
        style="italic",
    )

    # Trajectory context layer
    add_section_label(ax, 8, 5.25, "Trajectory Context Layer", fontsize=14)

    draw_box(ax, 1.2, 3.8, 2.8, 1.0, "IMPROVING_STABILIZING\nRecovery / Decompression\nCandidate")
    draw_box(ax, 6.6, 3.8, 2.8, 1.0, "UNSTABLE_VOLATILE\nVariable / Uncertain\nNeeds Watching")
    draw_box(ax, 12.0, 3.8, 2.8, 1.0, "WORSENING_ACCELERATING\nRising Risk\nEscalation Candidate")

    draw_arrow(ax, 2.6, 4.8, 5.4, 6.5)
    draw_arrow(ax, 8.0, 4.8, 8.8, 6.5)
    draw_arrow(ax, 13.4, 4.8, 12.2, 6.5)

    draw_arrow(ax, 5.4, 6.5, 2.6, 4.8)
    draw_arrow(ax, 8.8, 6.5, 8.0, 4.8)
    draw_arrow(ax, 12.2, 6.5, 13.4, 4.8)

    # Agreement / authority layer
    draw_box(ax, 3.0, 1.5, 3.0, 1.0, "IBPIP Agreement\nPatient Near / Returning\nToward Baseline")
    draw_box(ax, 6.5, 1.5, 3.0, 1.0, "PSR Authority\nAttention Priority\nRemains Anchored")
    draw_box(ax, 10.0, 1.5, 3.0, 1.0, "BIRE-FI Agreement\nNo Rising Trajectory\nor Worsening Trend")

    draw_arrow(ax, 4.5, 2.5, 2.6, 3.8)
    draw_arrow(ax, 8.0, 2.5, 8.0, 3.8)
    draw_arrow(ax, 11.5, 2.5, 13.4, 3.8)

    add_footer(
        ax,
        text=(
            "Pre-event lifecycle is fluid: patients may escalate, stabilize, de-escalate, "
            "or re-escalate based on trajectory behavior, IBPIP baseline context, "
            "PSR authority, and reintegration logic."
        ),
        y=0.55,
        fontsize=10,
    )

    fig.tight_layout()
    return fig, ax


# =========================================
# 52.7 — Post-Event Lifecycle Intelligence Map
# =========================================

def plot_post_event_lifecycle_intelligence(figsize=(18, 12)):
    """
    Chapter 52.7 — Post-Event Lifecycle Intelligence Map
    """

    fig, ax = setup_figure(figsize=figsize, xlim=(0, 14), ylim=(0, 10))

    ax.text(7, 9.6, "BIRE OS — Post-Event Lifecycle Intelligence Map",
            ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(7, 9.25,
            "Stabilization, volatility tracking, re-escalation, recovery evaluation, and reintegration after an event",
            ha="center", va="center", fontsize=10)

    draw_box(ax, 0.7, 8.0, 2.2, 0.9, "EVENT OCCURS\nTransition Gate")
    draw_box(ax, 3.5, 8.0, 2.5, 0.9, "Initial Post-Event\nTreatment Context")
    draw_box(ax, 6.7, 8.0, 2.5, 0.9, "POST-EVENT MONITOR\nStabilization Watch")
    draw_box(ax, 10.0, 8.0, 2.5, 0.9, "Clinician View\nPost-Event Awareness")

    draw_arrow(ax, 2.9, 8.45, 3.5, 8.45, "handoff")
    draw_arrow(ax, 6.0, 8.45, 6.7, 8.45, "activate")
    draw_arrow(ax, 9.2, 8.45, 10.0, 8.45, "curated status")

    draw_box(ax, 0.7, 6.3, 2.3, 0.85, "IBPIP\nRecovery Baseline")
    draw_box(ax, 0.7, 5.1, 2.3, 0.85, "BIRE-FI\nPost-Event Trajectory")
    draw_box(ax, 0.7, 3.9, 2.3, 0.85, "BMS\nCare Mode Context")
    draw_box(ax, 0.7, 2.7, 2.3, 0.85, "GSS\nNoise / Suppression Control")
    draw_box(ax, 0.7, 1.5, 2.3, 0.85, "PSR\nAttention Priority")

    draw_box(ax, 4.2, 5.0, 3.3, 1.0,
             "Post-Event Assessment Core\nStability / Volatility / Decline", fontsize=11)

    draw_arrow(ax, 3.0, 6.7, 4.2, 5.8, "baseline")
    draw_arrow(ax, 3.0, 5.5, 4.2, 5.65, "trajectory")
    draw_arrow(ax, 3.0, 4.3, 4.2, 5.45, "mode")
    draw_arrow(ax, 3.0, 3.1, 4.2, 5.25, "suppression")
    draw_arrow(ax, 3.0, 1.9, 4.2, 5.05, "priority")
    draw_arrow(ax, 7.9, 8.0, 6.0, 6.0, "monitor data", curve=0.1)

    draw_box(ax, 8.6, 6.2, 2.5, 0.85, "STABLE_MONITOR\nImproving / Controlled")
    draw_box(ax, 8.6, 5.0, 2.5, 0.85, "VOLATILE_MONITOR\nUnstable Recovery")
    draw_box(ax, 8.6, 3.8, 2.5, 0.85, "DECLINING_MONITOR\nRecovery Failing")
    draw_box(ax, 8.6, 2.6, 2.5, 0.85, "RE-ESCALATE\nRenewed Worsening")
    draw_box(ax, 8.6, 1.4, 2.5, 0.85, "CRITICAL\nSevere Concern")

    draw_arrow(ax, 7.5, 5.5, 8.6, 6.6, "stable")
    draw_arrow(ax, 7.5, 5.5, 8.6, 5.4, "volatile")
    draw_arrow(ax, 7.5, 5.5, 8.6, 4.2, "declining")

    draw_arrow(ax, 9.85, 5.0, 9.85, 4.65, "worsens")
    draw_arrow(ax, 9.85, 3.8, 9.85, 3.45, "fails")
    draw_arrow(ax, 9.85, 2.6, 9.85, 2.25, "severe")

    draw_box(ax, 4.4, 2.2, 3.0, 0.9, "Recovery Evaluation\nIBPIP + BIRE-FI Agreement")
    draw_box(ax, 4.4, 0.9, 3.0, 0.9, "Reintegration Path\nReturn to Surveillance")

    draw_arrow(ax, 8.6, 6.6, 7.4, 2.75, "improving", curve=0.25)
    draw_arrow(ax, 5.9, 2.2, 5.9, 1.8, "recovery trusted")
    draw_arrow(ax, 4.4, 1.35, 2.0, 8.0, "pre-event surveillance", curve=0.35)

    draw_box(ax, 11.9, 5.7, 1.6, 0.9, "Clinician\nOverride", fontsize=9)
    draw_box(ax, 11.9, 4.2, 1.6, 0.9, "Audit Log\nTraceability", fontsize=9)
    draw_box(ax, 11.9, 2.7, 1.6, 0.9, "Policy\nReview", fontsize=9)

    draw_arrow(ax, 11.1, 6.6, 11.9, 6.15, "review")
    draw_arrow(ax, 11.1, 5.4, 11.9, 6.0, "review")
    draw_arrow(ax, 11.1, 4.2, 11.9, 4.65, "record")
    draw_arrow(ax, 11.1, 3.0, 11.9, 4.55, "record")
    draw_arrow(ax, 11.1, 2.0, 11.9, 4.45, "record")
    draw_arrow(ax, 11.9, 3.15, 11.1, 2.95, "rule update", curve=-0.1)
    draw_arrow(ax, 10.8, 8.0, 12.5, 6.6, "human judgment", curve=-0.1)
    draw_arrow(ax, 12.7, 5.7, 10.8, 2.95, "override path", curve=-0.2)

    ax.text(7, 0.35,
            "Doctrine: Post-event intelligence determines whether recovery is stable, unstable, failing, or ready for reintegration.",
            ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(7, 0.12,
            "BIRE OS continues monitoring after the event because stabilization, volatility, and re-escalation are part of the same lifecycle.",
            ha="center", va="center", fontsize=8)

    fig.tight_layout()
    return fig, ax


# =========================================
# 52.8 — Unified Pre/Post Lifecycle Intelligence Map
# =========================================

def plot_unified_lifecycle_intelligence(figsize=(20, 13)):
    """
    Chapter 52.8 — Unified Pre/Post Lifecycle Intelligence Map
    """

    fig, ax = setup_figure(figsize=figsize, xlim=(0, 16), ylim=(0, 12))

    add_title(
        ax,
        title="BIRE OS — Unified Pre/Post Lifecycle Intelligence Map",
        subtitle="Continuous operational intelligence from surveillance through recovery and reintegration",
        title_y=11.4,
        subtitle_y=11.0,
    )

    add_section_label(ax, 3.8, 9.9, "PRE-EVENT LIFECYCLE")

    draw_box(ax, 0.6, 8.8, 2.3, 0.9, "SURVEILLANCE\nContinuous Monitoring")
    draw_box(ax, 3.3, 8.8, 1.9, 0.9, "WATCH\nEarly Awareness")
    draw_box(ax, 5.7, 8.8, 2.0, 0.9, "ESCALATE\nRising Concern")
    draw_box(ax, 8.2, 8.8, 2.0, 0.9, "URGENT\nHigh Concern")

    draw_arrow(ax, 2.9, 9.25, 3.3, 9.25)
    draw_arrow(ax, 5.2, 9.25, 5.7, 9.25)
    draw_arrow(ax, 7.7, 9.25, 8.2, 9.25)

    draw_box(ax, 11.2, 8.45, 2.3, 1.4, "EVENT\nTRANSITION\nGOVERNANCE", fontsize=11)
    draw_arrow(ax, 10.2, 9.25, 11.2, 9.25, "event_now = 1")

    add_section_label(ax, 12.7, 6.9, "POST-EVENT LIFECYCLE")

    draw_box(ax, 11.0, 5.8, 2.5, 0.9, "POST-EVENT MONITOR\nStabilization Watch")
    draw_box(ax, 11.0, 4.5, 2.5, 0.9, "VOLATILE_MONITOR\nUnstable Recovery")
    draw_box(ax, 11.0, 3.2, 2.5, 0.9, "DECLINING_MONITOR\nRecovery Failing")
    draw_box(ax, 11.0, 1.9, 2.5, 0.9, "RE-ESCALATE\nRenewed Worsening")
    draw_box(ax, 11.0, 0.6, 2.5, 0.9, "CRITICAL\nSevere Concern")

    draw_arrow(ax, 12.25, 5.8, 12.25, 5.4, "volatile")
    draw_arrow(ax, 12.25, 4.5, 12.25, 4.1, "declining")
    draw_arrow(ax, 12.25, 3.2, 12.25, 2.8, "worsens")
    draw_arrow(ax, 12.25, 1.9, 12.25, 1.5, "critical")

    draw_box(ax, 5.6, 4.2, 3.2, 1.0, "Recovery Evaluation\nIBPIP + BIRE-FI Agreement", fontsize=11)
    draw_box(ax, 5.6, 2.4, 3.2, 1.0, "Reintegration Governance\nReturn Toward Surveillance", fontsize=11)

    draw_arrow(ax, 11.0, 6.2, 8.8, 4.9, "stabilizing", curve=0.15)
    draw_arrow(ax, 7.2, 4.2, 7.2, 3.4, "recovery trusted")
    draw_arrow(ax, 5.6, 2.9, 2.0, 8.8, "return to surveillance", curve=0.35)

    draw_box(
        ax,
        4.6,
        6.1,
        4.5,
        1.2,
        "Longitudinal Intelligence Core\nContinuous Reassessment Across Entire Lifecycle",
        fontsize=11,
    )

    draw_arrow(ax, 5.5, 8.8, 6.2, 7.3, "pre-event intelligence", curve=-0.15)
    draw_arrow(ax, 9.0, 6.7, 11.0, 6.2, "post-event intelligence", curve=-0.1)
    draw_arrow(ax, 6.8, 6.1, 7.0, 5.2, "recovery reassessment")

    draw_box(ax, 14.1, 7.4, 1.5, 0.9, "Clinician\nOverride", fontsize=9)
    draw_box(ax, 14.1, 5.8, 1.5, 0.9, "Policy\nGovernance", fontsize=9)
    draw_box(ax, 14.1, 4.2, 1.5, 0.9, "Audit\nTraceability", fontsize=9)

    draw_arrow(ax, 13.5, 6.2, 14.1, 7.85, "human review", curve=-0.15)
    draw_arrow(ax, 13.5, 4.9, 14.1, 6.25, "policy", curve=-0.1)
    draw_arrow(ax, 13.5, 3.6, 14.1, 4.65, "logging", curve=-0.1)

    draw_boundary_line(ax, x=10.7, y1=0.2, y2=10.2, label="Event Boundary")

    add_footer(
        ax,
        text="Doctrine: BIRE OS governs a continuous lifecycle — from anticipation to recovery to reintegration.",
        y=-0.1,
        fontsize=11,
    )

    fig.tight_layout()
    return fig, ax