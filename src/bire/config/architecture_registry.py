from __future__ import annotations

import pandas as pd


ARCHITECTURE_REGISTRY = [
    {
        "layer_order": 1,
        "layer": "INGESTION",
        "primary_modules": "bire.ingestion, bire.data",
        "purpose": "Load, normalize, validate, and prepare patient monitoring data.",
        "connects_to": "FEATURE_ENGINEERING",
    },
    {
        "layer_order": 2,
        "layer": "FEATURE_ENGINEERING",
        "primary_modules": "bire.features",
        "purpose": "Create leakage-safe temporal features, rolling windows, deltas, and patient trajectory signals.",
        "connects_to": "FORECASTING",
    },
    {
        "layer_order": 3,
        "layer": "FORECASTING",
        "primary_modules": "bire.pipeline, XGBoost",
        "purpose": "Generate deterioration risk predictions and multi-horizon forecasting outputs.",
        "connects_to": "GOVERNANCE, BIRE_FI, CONFIDENCE",
    },
    {
        "layer_order": 4,
        "layer": "BMS",
        "primary_modules": "bire.evaluation.alerts, bire.config.mode_registry",
        "purpose": "Apply care-mode-aware operational interpretation across ICU, ER, inpatient, and outpatient settings.",
        "connects_to": "GOVERNANCE",
    },
    {
        "layer_order": 5,
        "layer": "GOVERNANCE",
        "primary_modules": "bire.evaluation.alerts, GSS, GSS-VE",
        "purpose": "Control suppression, escalation, alert burden, and operational escalation behavior.",
        "connects_to": "LIFECYCLE, CONFIDENCE, EXPORTS",
    },
    {
        "layer_order": 6,
        "layer": "IBPIP",
        "primary_modules": "bire.intelligence.ibpip",
        "purpose": "Provide individualized baseline intelligence and patient-specific physiological context.",
        "connects_to": "GOVERNANCE, BIRE_FI, CONFIDENCE",
    },
    {
        "layer_order": 7,
        "layer": "BIRE_FI",
        "primary_modules": "bire.intelligence.bire_fi",
        "purpose": "Interpret trajectory direction, instability behavior, worsening, improvement, and volatility.",
        "connects_to": "LIFECYCLE, CONFIDENCE",
    },
    {
        "layer_order": 8,
        "layer": "LIFECYCLE",
        "primary_modules": "bire.evaluation.monitor, bire.config.state_registry, bire.config.transition_registry",
        "purpose": "Manage WATCH, ESCALATE, URGENT, MONITOR, RE-ESCALATE, CRITICAL, and future TRANSITION behavior.",
        "connects_to": "CONFIDENCE, EXPORTS, VISUALIZATION",
    },
    {
        "layer_order": 9,
        "layer": "CONFIDENCE",
        "primary_modules": "bire.intelligence.confidence",
        "purpose": "Evaluate how strongly BIRE trusts operational interpretations without modifying governance.",
        "connects_to": "LIFECYCLE, EXPORTS, CLINICIAN_FACING_INTELLIGENCE",
    },
    {
        "layer_order": 10,
        "layer": "EXPORTS",
        "primary_modules": "bire.io.exports, bire.config.export_registry",
        "purpose": "Persist CSV outputs, logs, manifests, audits, and research artifacts.",
        "connects_to": "DASHBOARDS, AUDITS, RESEARCH_ARTIFACTS",
    },
    {
        "layer_order": 11,
        "layer": "VISUALIZATION",
        "primary_modules": "bire.evaluation.plots",
        "purpose": "Generate architecture, trajectory, heatmap, lifecycle, and dashboard visualizations.",
        "connects_to": "CLINICIAN_FACING_INTELLIGENCE",
    },
    {
        "layer_order": 12,
        "layer": "CLINICIAN_FACING_INTELLIGENCE",
        "primary_modules": "bire.explanations, dashboards, future UI",
        "purpose": "Translate BIRE outputs into calm, interpretable, urgency-aware clinical communication.",
        "connects_to": "HUMAN_REVIEW",
    },
]

SUBSYSTEM_ROLE_REGISTRY = [
    {
        "subsystem": "PSR",
        "primary_role": "Operational prioritization authority",
        "role_type": "Primary Authority",
        "supports": "Queue orchestration, surveillance board",
    },
    {
        "subsystem": "IBPIP",
        "primary_role": "Patient-specific baseline context",
        "role_type": "Context Authority",
        "supports": "PSR, queue interpretation, lifecycle intelligence",
    },
    {
        "subsystem": "BIRE-FI",
        "primary_role": "Trajectory direction interpretation",
        "role_type": "Context Authority",
        "supports": "Queue interpretation, PSR, lifecycle intelligence",
    },
    {
        "subsystem": "Confidence",
        "primary_role": "Interpretation trust context",
        "role_type": "Reliability Context",
        "supports": "PSR, queue interpretation, clinician-facing summaries",
    },
    {
        "subsystem": "Uncertainty",
        "primary_role": "Ambiguity and instability burden context",
        "role_type": "Reliability Context",
        "supports": "PSR, queue interpretation, clinician-facing summaries",
    },
    {
        "subsystem": "Queue Orchestration",
        "primary_role": "Operational visibility organization",
        "role_type": "Orchestration Layer",
        "supports": "Command center, surveillance board, operational flow",
    },
    {
        "subsystem": "BMS",
        "primary_role": "Care-environment context",
        "role_type": "Operational Context",
        "supports": "PSR, queue thresholds, mode-aware surveillance",
    },
]


def subsystem_role_registry_to_dataframe():
    return pd.DataFrame(SUBSYSTEM_ROLE_REGISTRY)

def get_architecture_registry():
    return ARCHITECTURE_REGISTRY


def architecture_registry_to_dataframe():
    return pd.DataFrame(ARCHITECTURE_REGISTRY)


def list_architecture_layers():
    return [row["layer"] for row in ARCHITECTURE_REGISTRY]


def get_architecture_layer(layer_name: str):
    layer_name = layer_name.upper()

    for row in ARCHITECTURE_REGISTRY:
        if row["layer"].upper() == layer_name:
            return row

    return None