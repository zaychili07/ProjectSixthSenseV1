from __future__ import annotations

import pandas as pd


OPERATIONAL_REGISTRY = {
    "PSR": {
        "name": "Patient Surveillance Ranking",
        "purpose": "Estimate operational attention burden for multi-patient surveillance.",
        "authority": "Primary operational queue authority",
        "base_weights": {
            "state_priority": 20,
            "max_risk_60min": 50,
            "avg_risk_60min": 25,
            "max_uncertainty": 20,
            "rows_observed_scaled": 10,
            "avg_confidence_penalty": -10,
        },
    },

    "GLOBAL_QUEUE_V1": {
        "name": "Global Queue Version 1",
        "purpose": "Separate active operational queue visibility from background surveillance.",
        "authority": "Queue visibility management",
        "thresholds": {
            "active_queue": 150.0,
        },
        "states": [
            "ACTIVE_QUEUE",
            "BACKGROUND_SURVEILLANCE",
        ],
    },

    "GLOBAL_QUEUE_V2": {
        "name": "Global Queue Version 2",
        "purpose": "Future BMS, IBPIP, BIRE-FI, confidence, and uncertainty-aware queue orchestration.",
        "authority": "Future operational orchestration",
        "status": "planned",
        "future_inputs": [
            "BMS care mode",
            "IBPIP baseline deviation",
            "BIRE-FI trajectory context",
            "confidence context",
            "uncertainty burden",
            "PSR attention burden",
            "lifecycle state",
        ],
    },
}


def get_operational_registry():
    return OPERATIONAL_REGISTRY


def get_operational_policy(policy_name: str):
    return OPERATIONAL_REGISTRY.get(policy_name.upper())


def operational_registry_to_dataframe():
    rows = []

    for key, meta in OPERATIONAL_REGISTRY.items():
        rows.append({
            "operational_key": key,
            "name": meta["name"],
            "purpose": meta["purpose"],
            "authority": meta["authority"],
            "status": meta.get("status", "active"),
        })

    return pd.DataFrame(rows)