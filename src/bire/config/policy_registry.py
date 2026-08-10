# src/bire/config/policy_registry.py

"""
BIRE Centralized Policy Registry

This module contains centralized architectural policy definitions
used throughout the BIRE operational intelligence system.

Purpose:
- improve policy traceability
- reduce duplicated governance logic
- support scalable architecture management
- improve subsystem interoperability
- create standardized operational documentation
"""

from __future__ import annotations

import pandas as pd


# =========================================================
# POLICY REGISTRY
# =========================================================

POLICY_REGISTRY = {

    "GSS": {
        "name": "Gateway Suppression System",
        "purpose": "Reduces unnecessary alert burden while preserving operational escalation sensitivity.",
        "category": "Governance",
        "states": [
            "SUPPRESS",
            "WATCH",
            "ESCALATE",
            "URGENT",
        ],
    },

    "BMS": {
        "name": "BIRE Mode Selector",
        "purpose": "Applies mode-aware operational thresholds across care environments.",
        "category": "Operational Intelligence",
        "states": [
            "ICU",
            "ER",
            "INPATIENT",
            "OUTPATIENT",
        ],
    },

    "IBPIP": {
        "name": "Individual Baseline Intelligence Policy",
        "purpose": "Learns patient-specific baseline behavior for personalized intelligence interpretation.",
        "category": "Personalization",
        "states": [
            "BASELINE_STABLE",
            "BASELINE_DRIFT",
            "BASELINE_UNSTABLE",
        ],
    },

    "BIRE_FI": {
        "name": "BIRE Forecasting Intelligence",
        "purpose": "Interprets deterioration trajectory direction and operational instability behavior.",
        "category": "Forecasting",
        "states": [
            "IMPROVING",
            "UNSTABLE",
            "WORSENING",
            "VOLATILE",
        ],
    },

    "CONFIDENCE": {
        "name": "Confidence & Uncertainty Intelligence",
        "purpose": "Evaluates how strongly BIRE trusts operational interpretations.",
        "category": "Interpretation",
        "states": [
            "LOW_CONFIDENCE",
            "MODERATE_CONFIDENCE",
            "HIGH_CONFIDENCE",
        ],
    },

    "LIFECYCLE": {
        "name": "Continuous Lifecycle Intelligence",
        "purpose": "Manages operational patient progression across pre-event and post-event states.",
        "category": "Lifecycle",
        "states": [
            "WATCH",
            "ESCALATE",
            "URGENT",
            "MONITOR",
            "RE-ESCALATE",
            "CRITICAL",
        ],
    },
}


# =========================================================
# REGISTRY HELPERS
# =========================================================

def get_policy_registry():
    """
    Return full policy registry.
    """
    return POLICY_REGISTRY


def get_policy(policy_name: str):
    """
    Return a single policy definition.
    """
    return POLICY_REGISTRY.get(policy_name.upper())


def list_registered_policies():
    """
    Return all registered policy names.
    """
    return list(POLICY_REGISTRY.keys())


def policy_registry_to_dataframe():
    """
    Convert policy registry into a dataframe for inspection/export.
    """

    rows = []

    for key, value in POLICY_REGISTRY.items():

        rows.append({
            "policy_key": key,
            "policy_name": value["name"],
            "category": value["category"],
            "purpose": value["purpose"],
            "states": ", ".join(value["states"]),
        })

    return pd.DataFrame(rows)