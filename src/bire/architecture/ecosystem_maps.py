"""
BIRE OS Architecture — Ecosystem Maps

Contains architecture diagrams focused on:
- full ecosystem architecture
- subsystem interaction and authority flow
"""

from bire.architecture.utils import (
    setup_figure,
    draw_box,
    draw_arrow,
    add_footer,
)


# =========================================
# 52.1 — Full BIRE OS Ecosystem Architecture Map
# =========================================

def plot_bire_ecosystem_architecture(figsize=(18, 10)):
    """
    Chapter 52.1 — Full BIRE OS Ecosystem Architecture Map.

    Visualizes the major BIRE OS components and how they connect across:
    - data ingestion
    - feature engineering
    - risk modeling
    - alert logic
    - intelligence layers
    - queue orchestration
    - clinician view
    - security / audit / governance infrastructure
    """

    fig, ax = setup_figure(
        figsize=figsize,
        xlim=(0, 18),
        ylim=(0, 10),
    )

    # Title
    ax.text(
        9,
        9.4,
        "BIRE OS — Full Ecosystem Architecture Map",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    # Main pipeline
    draw_box(ax, 0.5, 7.5, 2.2, 1.0, "Data Sources\nVitals / Labs / EHR\nFuture Multimodal Inputs")
    draw_box(ax, 3.2, 7.5, 2.2, 1.0, "BIL\nBIRE Ingestion Layer\nValidation + Alignment")
    draw_box(ax, 5.9, 7.5, 2.2, 1.0, "Feature Engineering\nTemporal + Baseline\nStress Features")
    draw_box(ax, 8.6, 7.5, 2.2, 1.0, "Risk Models\nXGBoost\nMulti-Horizon Risk")
    draw_box(ax, 11.3, 7.5, 2.2, 1.0, "Alert Logic\nEpisodes\nPersistence + Cooldown")
    draw_box(ax, 14.0, 7.5, 2.2, 1.0, "GSS\nSuppression\nEscalation Control")

    draw_arrow(ax, 2.7, 8.0, 3.2, 8.0)
    draw_arrow(ax, 5.4, 8.0, 5.9, 8.0)
    draw_arrow(ax, 8.1, 8.0, 8.6, 8.0)
    draw_arrow(ax, 10.8, 8.0, 11.3, 8.0)
    draw_arrow(ax, 13.5, 8.0, 14.0, 8.0)

    # Intelligence layer
    draw_box(ax, 3.2, 5.5, 2.2, 1.0, "BMS\nCare Mode Context\nICU / ER_ESI / Inpatient")
    draw_box(ax, 5.9, 5.5, 2.2, 1.0, "IBPIP\nPatient-Specific\nBaseline Intelligence")
    draw_box(ax, 8.6, 5.5, 2.2, 1.0, "BIRE-FI\nTrajectory Intelligence\nWorsening / Stabilizing")
    draw_box(ax, 11.3, 5.5, 2.2, 1.0, "PSR\nPatient Surveillance Ranking\nAttention Guidance")
    draw_box(ax, 14.0, 5.5, 2.2, 1.0, "Queue Orchestration\nActive Queue\nBackground Surveillance")

    draw_arrow(ax, 4.3, 7.5, 4.3, 6.5)
    draw_arrow(ax, 7.0, 7.5, 7.0, 6.5)
    draw_arrow(ax, 9.7, 7.5, 9.7, 6.5)
    draw_arrow(ax, 12.4, 7.5, 12.4, 6.5)
    draw_arrow(ax, 15.1, 7.5, 15.1, 6.5)

    draw_arrow(ax, 5.4, 6.0, 5.9, 6.0)
    draw_arrow(ax, 8.1, 6.0, 8.6, 6.0)
    draw_arrow(ax, 10.8, 6.0, 11.3, 6.0)
    draw_arrow(ax, 13.5, 6.0, 14.0, 6.0)

    # Temporal operational intelligence
    draw_box(ax, 5.9, 3.4, 2.4, 1.1, "Temporal Operational\nIntelligence\nState Evolution")
    draw_box(ax, 8.8, 3.4, 2.4, 1.1, "WATCH Reintegration\nDe-escalation\nSUPPRESSED_WATCH")
    draw_box(ax, 11.7, 3.4, 2.4, 1.1, "Continuous Simulation\n5-Min Refresh Cycles\nOperational Movement")
    draw_box(ax, 14.6, 3.4, 2.4, 1.1, "Clinician View\nLive Simulation\nSafe Interpretation")

    draw_arrow(ax, 15.1, 5.5, 7.1, 4.5)
    draw_arrow(ax, 8.3, 4.0, 8.8, 4.0)
    draw_arrow(ax, 11.2, 4.0, 11.7, 4.0)
    draw_arrow(ax, 14.1, 4.0, 14.6, 4.0)

    # Security + audit + governance
    draw_box(ax, 0.8, 2.0, 2.5, 1.0, "Security Layer\nEncryption\nSecure Artifacts")
    draw_box(ax, 4.0, 2.0, 2.5, 1.0, "Audit + Logs\nMovement History\nTraceability")
    draw_box(ax, 7.2, 2.0, 2.5, 1.0, "Policy / Registry\nOperational Rules\nGovernance")
    draw_box(ax, 10.4, 2.0, 2.5, 1.0, "Config System\nFuture Tuning\nThreshold Control")

    draw_arrow(ax, 2.0, 7.5, 2.0, 3.0)
    draw_arrow(ax, 15.8, 3.4, 5.3, 3.0)
    draw_arrow(ax, 8.4, 3.4, 8.4, 3.0)
    draw_arrow(ax, 11.6, 3.4, 11.6, 3.0)

    # Feedback loop
    draw_arrow(ax, 15.8, 3.4, 15.8, 1.2)
    draw_arrow(ax, 15.8, 1.2, 1.6, 1.2)
    draw_arrow(ax, 1.6, 1.2, 1.6, 7.5)

    add_footer(
        ax,
        text=(
            "Core flow: Data → Intelligence → Orchestration → Continuous Simulation → Clinician View | "
            "Security, Audit, Policy, and Config support the ecosystem."
        ),
        y=0.45,
        fontsize=10,
    )

    fig.tight_layout()
    return fig, ax


# =========================================
# 52.4 — Subsystem Interaction & Authority Flow Map
# =========================================

def plot_subsystem_authority_flow(figsize=(18, 12)):
    """
    Chapter 52.4 — Subsystem Interaction & Authority Flow Map.

    Visualizes how BIRE OS subsystems communicate, influence operational state,
    and support governed clinical surveillance decisions.
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
        "BIRE OS — Subsystem Interaction & Authority Flow Map",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    ax.text(
        7,
        9.25,
        "How intelligence subsystems cooperate, influence decisions, and support governed operational routing",
        ha="center",
        va="center",
        fontsize=10,
    )

    # Input layer
    draw_box(ax, 0.5, 7.8, 2.4, 0.8, "Patient Data Stream\nVitals / Labs / Context")
    draw_box(ax, 3.4, 7.8, 2.4, 0.8, "Feature Engineering\nTemporal + Rolling Signals")
    draw_box(ax, 6.3, 7.8, 2.4, 0.8, "Risk Model Layer\nMulti-Horizon Prediction")
    draw_box(ax, 9.2, 7.8, 2.4, 0.8, "BMS\nMode-Aware Thresholding")
    draw_box(ax, 12.0, 7.8, 1.5, 0.8, "Policy\nRegistry")

    draw_arrow(ax, 2.9, 8.2, 3.4, 8.2)
    draw_arrow(ax, 5.8, 8.2, 6.3, 8.2)
    draw_arrow(ax, 8.7, 8.2, 9.2, 8.2)
    draw_arrow(ax, 12.0, 8.2, 11.6, 8.2, "rules")

    # Intelligence layer
    draw_box(ax, 1.0, 5.7, 2.5, 0.9, "IBPIP\nIndividual Baseline Intelligence")
    draw_box(ax, 4.2, 5.7, 2.5, 0.9, "BIRE-FI\nForecasting Intelligence")
    draw_box(ax, 7.4, 5.7, 2.5, 0.9, "GSS\nGateway Suppression System")
    draw_box(ax, 10.6, 5.7, 2.5, 0.9, "PSR\nPatient Scoring Report")

    draw_arrow(ax, 7.5, 7.8, 2.25, 6.6, "risk + vitals")
    draw_arrow(ax, 7.5, 7.8, 5.45, 6.6, "trajectory")
    draw_arrow(ax, 10.4, 7.8, 8.65, 6.6, "mode-aware alert logic")
    draw_arrow(ax, 10.4, 7.8, 11.85, 6.6, "priority evidence")

    draw_arrow(ax, 3.5, 6.15, 4.2, 6.15, "baseline status")
    draw_arrow(ax, 6.7, 6.15, 7.4, 6.15, "forecast state")
    draw_arrow(ax, 9.9, 6.15, 10.6, 6.15, "suppression status")

    # Governance layer
    draw_box(
        ax,
        3.0,
        3.8,
        3.0,
        0.9,
        "Operational Decision Core\nEscalate / De-escalate / Suppress / Reinstate",
    )

    draw_box(
        ax,
        7.8,
        3.8,
        3.0,
        0.9,
        "Queue Orchestration\nActive Queue / Background Surveillance",
    )

    draw_box(ax, 11.3, 3.8, 2.0, 0.9, "Audit Log\nTraceability")

    draw_arrow(ax, 2.25, 5.7, 4.5, 4.7, "recovery evidence")
    draw_arrow(ax, 5.45, 5.7, 4.8, 4.7, "trajectory evidence")
    draw_arrow(ax, 8.65, 5.7, 5.2, 4.7, "suppression gate")
    draw_arrow(ax, 11.85, 5.7, 9.3, 4.7, "priority score")

    draw_arrow(ax, 6.0, 4.25, 7.8, 4.25, "routing decision")
    draw_arrow(ax, 10.8, 4.25, 11.3, 4.25, "decision record")

    # Output layer
    draw_box(ax, 1.0, 1.8, 2.4, 0.9, "SUPPRESSED_WATCH\nLow Attention Load")
    draw_box(ax, 4.0, 1.8, 2.0, 0.9, "WATCH\nEarly Signal")
    draw_box(ax, 6.7, 1.8, 2.0, 0.9, "ESCALATE\nRising Concern")
    draw_box(ax, 9.4, 1.8, 2.0, 0.9, "URGENT\nHigh Concern")

    draw_box(ax, 6.15, 0.55, 2.4, 0.9, "Clinician View\nCurated Operational Awareness")

    draw_arrow(ax, 8.7, 3.8, 2.2, 2.7)
    draw_arrow(ax, 8.7, 3.8, 5.0, 2.7)
    draw_arrow(ax, 8.7, 3.8, 7.7, 2.7)
    draw_arrow(ax, 8.7, 3.8, 10.4, 2.7)

    # Clinician-facing visibility
    draw_arrow(ax, 5.0, 1.8, 6.8, 1.45, "early awareness")
    draw_arrow(ax, 7.7, 1.8, 7.35, 1.45, "elevated concern")
    draw_arrow(ax, 10.4, 1.8, 7.95, 1.45, "high concern")

    # Reintegration loop
    draw_arrow(ax, 7.7, 1.8, 5.0, 1.8, "de-escalation")
    draw_arrow(ax, 5.0, 1.8, 2.2, 1.8, "reintegration")
    draw_arrow(ax, 5.0, 2.7, 7.7, 2.7, "re-escalation")

    ax.text(
        7,
        0.25,
        "Doctrine: BIRE OS uses coordinated intelligence — no single subsystem owns the whole decision.",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        7,
        0.08,
        (
            "IBPIP + BIRE-FI + GSS + BMS + PSR cooperate through policy governance "
            "to support reversible operational routing."
        ),
        ha="center",
        va="center",
        fontsize=9,
    )

    fig.tight_layout()
    return fig, ax