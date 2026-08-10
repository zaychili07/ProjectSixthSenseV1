# ============================================================
# BIRE OS — Advanced Trajectory Intelligence (ATI)
# File: src/bire/ati/governance.py
# Chapter: 64.1
#
# Created: 2026-06-13
# Updated: 2026-06-13
#
# CHANGED:
# - Added ATI governance boundary helpers.
# - Added ATI ownership boundary helpers.
# - Added ATI input policy helpers.
# - Added ATI architecture boundary helpers.
# - Added ATI consumer boundary helpers.
# - Added ATI output policy helpers.
#
# Purpose:
# Defines ATI ownership and governance boundaries before
# trajectory interpretation logic is implemented.
# ============================================================

import pandas as pd

# This function builds the ATI governance table.
def build_ati_governance_table():
    """
    Build ATI governance table.

    ATI consumes trajectory-aware evidence and interprets trajectory behavior.
    """

    return pd.DataFrame({
        "governance_rule": [
            "Evidence Before Interpretation",
            "Measurement Boundary",
            "No Forecast Ownership",
            "No Memory Ownership",
            "No Intervention Ownership",
            "No Recovery Ownership",
            "Patient-Centered Consumption",
        ],
        "description": [
            "ATI must consume existing evidence before generating trajectory interpretation.",
            "ATI does not create raw physiologic measurements or hidden vital calculations.",
            "ATI does not forecast future outcomes.",
            "ATI does not store or recall longitudinal patient history.",
            "ATI does not determine intervention effectiveness.",
            "ATI does not determine whether recovery is authentic or durable.",
            "ATI outputs must remain usable by IBPIP for patient-centered interpretation.",
        ],
    })
# This function builds the ATI ownership table.
def build_ati_ownership_table():
    """
    Build ATI ownership review table.

    ATI owns trajectory interpretation.
    ATI does not own measurement generation,
    memory, forecasting, recovery interpretation,
    or intervention evaluation.
    """


    return pd.DataFrame({
        "capability": [
            "Trajectory Direction Interpretation",
            "Trajectory Persistence Interpretation",
            "Trajectory Acceleration Interpretation",
            "Trajectory Reversal Interpretation",
            "Trajectory Continuity Interpretation",
            "Trajectory Stability Interpretation",
            "Physiologic Measurement Creation",
            "Forecast Generation",
            "Longitudinal Memory",
            "Recovery Authenticity Determination",
            "Intervention Effect Evaluation",
            "Clinician Communication",
        ],

        "owned_by_ati": [
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "No",
            "No",
            "No",
            "No",
            "No",
            "No",
        ],
    })

# This function builds the ATI consumer table.
def build_ati_consumer_table():
    """
    Build ATI consumer review table.

    ATI provides trajectory interpretation.
    Consuming domains determine how those
    interpretations are used.
    """


    return pd.DataFrame({
        "consumer": [
            "IBPIP",
            "BIRE-FI",
            "TARI",
            "RSS",
            "PSRv2",
            "Future Intelligence Domains",
        ],

        "purpose": [
            "Patient-centered trajectory interpretation",
            "Trajectory-informed forecasting",
            "Recovery trajectory interpretation",
            "Replay and historical trajectory review",
            "Trajectory-aware clinician communication",
            "Future trajectory-aware intelligence workflows",
        ],
    })

# This function builds the ATI architecture table.
def build_ati_architecture_table():

    return pd.DataFrame({
        "component": [
            "HVI",
            "ATI",
            "IBPIP",
            "BIRE-FI",
            "TARI",
            "RSS",
            "PSRv2",
        ],

        "relationship_to_ati": [
            "Provides trajectory-aware evidence",
            "Interprets trajectory behavior",
            "Consumes patient-centered trajectory interpretation",
            "Consumes trajectory interpretation for forecasting",
            "Consumes trajectory interpretation for recovery evaluation",
            "Consumes trajectory interpretation for replay analysis",
            "Consumes trajectory interpretation for clinician communication",
        ],
    })

# This function builds the ATI input policy table.
def build_ati_input_policy_table():
    """
    Build ATI input policy review table.

    ATI consumes trajectory-relevant evidence
    but does not generate measurements itself.
    """

    return pd.DataFrame({
        "evidence_source": [
            "HVI Burden Signals",
            "HVI Compensation Signals",
            "HVI Instability Signals",
            "HVI Recovery Measurements",
            "Forecast Outputs",
            "Longitudinal Memory",
            "Intervention Effect Evaluations",
            "Clinician Summaries",
        ],

        "eligible_input": [
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "No",
            "No",
            "No",
            "No",
        ],

        "notes": [
            "Trajectory-relevant burden evidence.",
            "Trajectory-relevant compensation evidence.",
            "Trajectory-relevant instability evidence.",
            "Trajectory-relevant recovery measurements.",
            "Forecasting remains owned by BIRE-FI.",
            "Memory remains owned by LPMR.",
            "Intervention evaluation remains owned by TTI.",
            "Clinician communication remains owned by PSRv2.",
        ],
    })

# This function builds the ATI output policy table.
def build_ati_output_policy_table():
    """
    Build ATI output policy review table.

    ATI provides trajectory interpretation outputs.
    ATI does not forecast, evaluate interventions,
    determine recovery authenticity, or communicate
    directly with clinicians.
    """

    return pd.DataFrame({
        "output": [
            "Trajectory Direction",
            "Trajectory Persistence",
            "Trajectory Acceleration Interpretation",
            "Trajectory Stability Interpretation",
            "Trajectory Reversal Interpretation",
            "Trajectory Pattern Interpretation",
            "Forecast Output",
            "Recovery Authenticity Determination",
            "Intervention Effect Evaluation",
            "Clinician Summary",
        ],

        "allowed_output": [
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "No",
            "No",
            "No",
            "No",
        ],

        "notes": [
            "ATI may interpret trajectory direction.",
            "ATI may interpret trajectory persistence.",
            "ATI may interpret trajectory acceleration behavior.",
            "ATI may interpret trajectory stability behavior.",
            "ATI may interpret trajectory reversals.",
            "ATI may interpret higher-level trajectory patterns.",
            "Forecasting remains owned by BIRE-FI.",
            "Recovery authenticity remains owned by TARI.",
            "Intervention evaluation remains owned by TTI.",
            "Clinician communication remains owned by PSRv2.",
        ],
    })

# This function builds the ATI governance rules table.
def build_ati_governance_rules_table():
    """
    Build ATI governance rules review table.

    ATI interprets trajectories.
    ATI does not forecast, evaluate,
    remember, or communicate.
    """

    return pd.DataFrame({
        "rule": [
            "Evidence Before Interpretation",
            "Interpretation Before Prediction",
            "Trajectory Focus",
            "Ownership Boundaries",
            "Patient-Centered Interpretation",
            "Pattern Recognition",
            "No Forecast Ownership",
            "No Recovery Ownership",
            "No Intervention Ownership",
            "No Memory Ownership",
        ],

        "description": [
            "ATI must consume evidence before producing trajectory interpretation.",
            "ATI interprets trajectory behavior but does not forecast outcomes.",
            "ATI remains focused on trajectory behavior and trajectory patterns.",
            "ATI may not absorb responsibilities belonging to other domains.",
            "ATI interpretations should remain usable within patient-centered workflows.",
            "ATI may interpret recurring trajectory patterns across available evidence.",
            "Forecasting remains owned by BIRE-FI.",
            "Recovery authenticity remains owned by TARI.",
            "Intervention evaluation remains owned by TTI.",
            "Longitudinal memory remains owned by LPMR.",
        ],
    })