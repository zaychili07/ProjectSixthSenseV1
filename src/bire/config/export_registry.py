# src/bire/config/export_registry.py

from __future__ import annotations

import pandas as pd


EXPORT_REGISTRY = {

    "RAW_DATA": {
        "category": "Data",
        "description": "Initial ingested patient monitoring data.",
        "artifact_type": "csv",
    },

    "CYCLE1_FEATURES": {
        "category": "Features",
        "description": "Cycle 1 engineered temporal features.",
        "artifact_type": "csv",
    },

    "V2_MULTI_HORIZON": {
        "category": "Forecasting",
        "description": "Multi-horizon forecasting intelligence outputs.",
        "artifact_type": "csv",
    },

    "GSS_OUTPUT": {
        "category": "Governance",
        "description": "Gateway Suppression System operational outputs.",
        "artifact_type": "csv",
    },

    "IBPIP_OUTPUT": {
        "category": "Personalization",
        "description": "Individual baseline intelligence outputs.",
        "artifact_type": "csv",
    },

    "BIRE_FI_OUTPUT": {
        "category": "Forecasting",
        "description": "Forecasting intelligence trajectory outputs.",
        "artifact_type": "csv",
    },

    "CONFIDENCE_OUTPUT": {
        "category": "Confidence",
        "description": "Confidence and uncertainty intelligence outputs.",
        "artifact_type": "csv",
    },

    "EPISODE_OUTPUT": {
        "category": "Lifecycle",
        "description": "Episode-level operational intelligence outputs.",
        "artifact_type": "csv",
    },

    "AUDIT_LOGS": {
        "category": "Audits",
        "description": "Operational governance and audit exports.",
        "artifact_type": "log",
    },

    "RESEARCH_ARTIFACTS": {
        "category": "Research",
        "description": "Research-grade experiment and artifact outputs.",
        "artifact_type": "mixed",
    },
}


def get_export_registry():
    return EXPORT_REGISTRY


def export_registry_to_dataframe():

    rows = []

    for key, meta in EXPORT_REGISTRY.items():

        rows.append({
            "export_key": key,
            "category": meta["category"],
            "artifact_type": meta["artifact_type"],
            "description": meta["description"],
        })

    return pd.DataFrame(rows)