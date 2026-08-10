# src/bire/config/state_registry.py

"""
BIRE Lifecycle State Registry

This module contains centralized lifecycle and operational state
definitions used throughout the BIRE intelligence system.

Purpose:
- standardize lifecycle state naming
- reduce state drift across subsystems
- support lifecycle orchestration
- improve architecture traceability
- support future visualization/testing systems
"""

from __future__ import annotations

import pandas as pd


# =========================================================
# STATE REGISTRY
# =========================================================

STATE_REGISTRY = {

    "SUPPRESS": {
        "category": "Governance",
        "phase": "Pre-Event",
        "description": (
            "Alert condition suppressed to reduce unnecessary "
            "operational burden."
        ),
    },

    "WATCH": {
        "category": "Escalation",
        "phase": "Pre-Event",
        "description": (
            "Early deterioration pressure detected with lower "
            "operational severity."
        ),
    },

    "ESCALATE": {
        "category": "Escalation",
        "phase": "Pre-Event",
        "description": (
            "Meaningful deterioration pressure detected requiring "
            "increased operational attention."
        ),
    },

    "URGENT": {
        "category": "Escalation",
        "phase": "Pre-Event",
        "description": (
            "High operational deterioration pressure requiring "
            "immediate escalation awareness."
        ),
    },

    "MONITOR": {
        "category": "Post-Event",
        "phase": "Post-Event",
        "description": (
            "Patient remains under post-event surveillance for "
            "stability and recovery interpretation."
        ),
    },

    "RE-ESCALATE": {
        "category": "Post-Event",
        "phase": "Post-Event",
        "description": (
            "Patient demonstrates renewed deterioration pressure "
            "following a monitored recovery period."
        ),
    },

    "CRITICAL": {
        "category": "Critical",
        "phase": "Post-Event",
        "description": (
            "Patient exhibits severe operational instability or "
            "critical deterioration behavior."
        ),
    },

    "STABLE_MONITOR": {
        "category": "Monitor Substate",
        "phase": "Post-Event",
        "description": (
            "Post-event trajectory appears stable with reduced "
            "volatility and improving operational consistency."
        ),
    },

    "VOLATILE_MONITOR": {
        "category": "Monitor Substate",
        "phase": "Post-Event",
        "description": (
            "Post-event trajectory remains unstable or operationally "
            "volatile."
        ),
    },

    "DECLINING_MONITOR": {
        "category": "Monitor Substate",
        "phase": "Post-Event",
        "description": (
            "Post-event trajectory demonstrates continued operational "
            "decline or worsening behavior."
        ),
    },

    "TRANSITION": {
        "category": "Lifecycle",
        "phase": "Future Lifecycle",
        "description": (
            "Future lifecycle recycling state used to transition "
            "patients back into pre-event surveillance."
        ),
    },
}


# =========================================================
# REGISTRY HELPERS
# =========================================================

def get_state_registry():
    """
    Return full lifecycle state registry.
    """
    return STATE_REGISTRY


def get_state(state_name: str):
    """
    Return a single state definition.
    """
    return STATE_REGISTRY.get(state_name.upper())


def list_registered_states():
    """
    Return all registered state names.
    """
    return list(STATE_REGISTRY.keys())


def state_registry_to_dataframe():
    """
    Convert lifecycle state registry into dataframe.
    """

    rows = []

    for state, meta in STATE_REGISTRY.items():

        rows.append({
            "state": state,
            "category": meta["category"],
            "phase": meta["phase"],
            "description": meta["description"],
        })

    return pd.DataFrame(rows)