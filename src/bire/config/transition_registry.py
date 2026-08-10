# src/bire/config/transition_registry.py

from __future__ import annotations

import pandas as pd


TRANSITION_REGISTRY = [
    {
        "from_state": "SUPPRESS",
        "to_state": "WATCH",
        "phase": "Pre-Event",
        "transition_type": "emergence",
        "description": "Suppressed signal becomes meaningful enough for early instability awareness.",
    },
    {
        "from_state": "WATCH",
        "to_state": "ESCALATE",
        "phase": "Pre-Event",
        "transition_type": "worsening",
        "description": "Early instability strengthens into meaningful deterioration pressure.",
    },
    {
        "from_state": "ESCALATE",
        "to_state": "URGENT",
        "phase": "Pre-Event",
        "transition_type": "worsening",
        "description": "Deterioration pressure becomes high enough for urgent operational awareness.",
    },
    {
        "from_state": "URGENT",
        "to_state": "MONITOR",
        "phase": "Post-Event",
        "transition_type": "post_event_entry",
        "description": "Patient enters post-event surveillance after high-severity escalation/event behavior.",
    },
    {
        "from_state": "MONITOR",
        "to_state": "RE-ESCALATE",
        "phase": "Post-Event",
        "transition_type": "renewed_worsening",
        "description": "Patient demonstrates renewed deterioration pressure while under monitoring.",
    },
    {
        "from_state": "RE-ESCALATE",
        "to_state": "CRITICAL",
        "phase": "Post-Event",
        "transition_type": "critical_worsening",
        "description": "Re-escalation progresses into severe critical instability.",
    },
    {
        "from_state": "CRITICAL",
        "to_state": "MONITOR",
        "phase": "Post-Event",
        "transition_type": "stabilization",
        "description": "Patient moves from critical instability back into post-event surveillance.",
    },
    {
        "from_state": "MONITOR",
        "to_state": "TRANSITION",
        "phase": "Lifecycle Bridge",
        "transition_type": "recovery_bridge",
        "description": "Patient demonstrates enough stability to begin recycling toward pre-event surveillance.",
    },
    {
        "from_state": "TRANSITION",
        "to_state": "WATCH",
        "phase": "Lifecycle Bridge",
        "transition_type": "recycle",
        "description": "Patient is recycled back into pre-event surveillance rather than leaving the intelligence loop.",
    },
]


def get_transition_registry():
    return TRANSITION_REGISTRY


def transition_registry_to_dataframe():
    return pd.DataFrame(TRANSITION_REGISTRY)


def list_allowed_transitions():
    return [(row["from_state"], row["to_state"]) for row in TRANSITION_REGISTRY]


def get_next_states(from_state: str):
    state = from_state.upper()
    return [
        row["to_state"]
        for row in TRANSITION_REGISTRY
        if row["from_state"] == state
    ]