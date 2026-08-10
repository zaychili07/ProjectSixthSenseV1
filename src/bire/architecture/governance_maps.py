"""
BIRE OS Architecture — Governance Maps

Contains architecture diagrams focused on:
- governance
- clinician override
- audit intelligence
- accountability infrastructure
"""

from bire.architecture.utils import (
    setup_figure,
    draw_box,
    draw_arrow,
    add_title,
    add_footer,
)


# =========================================
# 52.9 — Governance, Override, and Audit Intelligence Map
# =========================================

def plot_governance_override_audit_map(figsize=(20, 12)):
    """
    Chapter 52.9 — Governance, Override, and Audit Intelligence Map

    Visualizes how BIRE OS preserves:
    - clinician authority
    - policy governance
    - override control
    - audit traceability
    - longitudinal accountability
    """

    fig, ax = setup_figure(
        figsize=figsize,
        xlim=(0, 16),
        ylim=(0, 11),
    )

    add_title(
        ax,
        title="BIRE OS — Governance, Override, and Audit Intelligence Map",
        subtitle="Clinician authority, policy control, subsystem governance, override handling, and traceability",
        title_y=10.45,
        subtitle_y=10.05,
    )

    # =====================================================
    # Operational intelligence layer
    # =====================================================

    draw_box(ax, 0.7, 8.4, 2.3, 0.9, "Operational States\nWATCH / ESCALATE / URGENT")

    draw_box(ax, 3.6, 8.4, 2.3, 0.9, "Post-Event States\nMONITOR / RE-ESCALATE / CRITICAL")

    draw_box(ax, 6.5, 8.4, 2.3, 0.9, "Subsystem Outputs\nIBPIP / BIRE-FI / GSS / PSR")

    draw_box(ax, 9.4, 8.4, 2.3, 0.9, "Operational Decision Core\nRouting + Suppression")

    draw_box(ax, 12.4, 8.4, 2.4, 0.9, "Clinician View\nCurated Awareness")

    draw_arrow(ax, 3.0, 8.85, 3.6, 8.85)
    draw_arrow(ax, 5.9, 8.85, 6.5, 8.85)
    draw_arrow(ax, 8.8, 8.85, 9.4, 8.85)
    draw_arrow(ax, 11.7, 8.85, 12.4, 8.85)

    # =====================================================
    # Governance core
    # =====================================================

    draw_box(
        ax,
        5.6,
        6.0,
        4.8,
        1.2,
        "Governance Intelligence Core\nPolicy Validation / Override Review / Accountability",
        fontsize=12,
    )

    draw_arrow(ax, 10.55, 8.4, 8.0, 7.2, "decision metadata", curve=0.08)
    draw_arrow(ax, 13.6, 8.4, 8.4, 7.2, "clinician feedback", curve=0.12)
    draw_arrow(ax, 7.65, 8.4, 7.4, 7.2, "subsystem evidence")

    # =====================================================
    # Policy layer
    # =====================================================

    draw_box(ax, 0.8, 5.8, 2.6, 0.9, "Policy Registry\nThresholds / Rules / Modes")

    draw_box(ax, 0.8, 4.4, 2.6, 0.9, "BMS Governance\nCare Mode Constraints")

    draw_box(ax, 0.8, 3.0, 2.6, 0.9, "Suppression Governance\nGSS Review")

    draw_box(ax, 0.8, 1.6, 2.6, 0.9, "Reintegration Governance\nRecovery Trust Rules")

    draw_arrow(ax, 3.4, 6.25, 5.6, 6.65, "rules")
    draw_arrow(ax, 3.4, 4.85, 5.6, 6.45, "mode logic")
    draw_arrow(ax, 3.4, 3.45, 5.6, 6.25, "suppression logic")
    draw_arrow(ax, 3.4, 2.05, 5.6, 6.05, "recovery criteria")

    # =====================================================
    # Override layer
    # =====================================================

    draw_box(ax, 12.2, 6.0, 2.7, 0.9, "Clinician Override\nHuman Authority")

    draw_box(ax, 12.2, 4.6, 2.7, 0.9, "Override Classification\nAccept / Reject / Modify")

    draw_box(ax, 12.2, 3.2, 2.7, 0.9, "Override Reason Capture\nClinical Rationale")

    draw_box(ax, 12.2, 1.8, 2.7, 0.9, "Review Queue\nGovernance Follow-Up")

    draw_arrow(ax, 12.4, 8.4, 13.55, 6.9, "review")
    draw_arrow(ax, 13.55, 6.0, 13.55, 5.5, "classify")
    draw_arrow(ax, 13.55, 4.6, 13.55, 4.1, "document")
    draw_arrow(ax, 13.55, 3.2, 13.55, 2.7, "follow-up")

    draw_arrow(ax, 12.2, 6.45, 10.4, 6.65, "override signal")
    draw_arrow(ax, 10.4, 6.25, 12.2, 4.95, "governance response", curve=-0.1)

    # =====================================================
    # Audit / memory layer
    # =====================================================

    draw_box(ax, 5.1, 3.8, 2.4, 0.9, "Audit Log\nDecision Trace")

    draw_box(ax, 8.4, 3.8, 2.4, 0.9, "Event Memory\nLongitudinal Record")

    draw_box(ax, 6.75, 2.2, 2.8, 0.9, "Quality Review\nDrift / Conflict / Failure Modes")

    draw_box(ax, 6.75, 0.9, 2.8, 0.9, "System Learning Queue\nFuture Policy Refinement")

    draw_arrow(ax, 7.0, 6.0, 6.3, 4.7, "trace")
    draw_arrow(ax, 8.8, 6.0, 9.6, 4.7, "memory")
    draw_arrow(ax, 6.3, 3.8, 8.1, 3.1, "review")
    draw_arrow(ax, 9.6, 3.8, 8.3, 3.1, "review")
    draw_arrow(ax, 8.15, 2.2, 8.15, 1.8, "refine")

    # =====================================================
    # Feedback loops
    # =====================================================

    draw_arrow(ax, 8.15, 0.9, 2.1, 5.8, "policy update candidate", curve=0.35)

    draw_arrow(ax, 8.15, 0.9, 7.65, 8.4, "subsystem tuning candidate", curve=-0.35)

    draw_arrow(ax, 8.15, 0.9, 10.55, 8.4, "decision logic review", curve=-0.25)

    # =====================================================
    # Safety doctrine layer
    # =====================================================

    draw_box(
        ax,
        4.8,
        7.55,
        5.6,
        0.45,
        "Safety Doctrine: Support clinical judgment — never replace it.",
        fontsize=9,
    )

    add_footer(
        ax,
        text=(
            "Doctrine: Governance converts intelligence into accountable clinical decision support "
            "through policy, override handling, audit memory, and review."
        ),
        y=0.25,
        fontsize=10,
    )

    fig.tight_layout()
    return fig, ax