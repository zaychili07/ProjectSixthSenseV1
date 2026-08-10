# ============================================================
# BIRE OS — Ecosystem Signal Flow Architecture
# File: src/bire/architecture/ecosystem_flow.py
# Purpose:
# Backend helper for Chapter 60 ecosystem signal-flow mapping.
# ============================================================

import pandas as pd


def build_ecosystem_signal_flow() -> pd.DataFrame:
    """
    Build the foundational BIRE OS ecosystem signal-flow map.

    This function defines the proposed high-level order of future
    BIRE OS subsystem cooperation, from ingestion through operational
    output.

    Returns
    -------
    pd.DataFrame
        Ecosystem signal-flow architecture table.
    """

    return pd.DataFrame(
        {
            "layer_order": [
                1, 2, 3, 4, 5,
                6, 7, 8, 9, 10,
            ],
            "ecosystem_layer": [
                "BIL",
                "NCL",
                "IBPIP / LPMR",
                "HVI",
                "ATI",
                "TTI",
                "BMS / HIT",
                "BIRE-FI",
                "PSR-v2",
                "Operational Output",
            ],
            "primary_role": [
                "Data ingestion and normalization",
                "Noise qualification and signal governance",
                "Patient-specific personalization and longitudinal memory",
                "Hidden physiology and compensation interpretation",
                "Trajectory interpretation and instability pathway analysis",
                "Therapeutic and intervention-aware intelligence",
                "Operational care-environment adaptation",
                "Forecasting orchestration and confidence-aware forecasting",
                "Readable patient prioritization and operational synthesis",
                "Clinician-facing operational interpretation",
            ],
            "core_output": [
                "Normalized patient signal stream",
                "Cleaned and reliability-qualified signal layer",
                "Personalized longitudinal patient context",
                "Hidden instability interpretation",
                "Trajectory-aware operational interpretation",
                "Therapy-adjusted trajectory interpretation",
                "Operationally adapted intelligence context",
                "Forecast-ready trajectory interpretation",
                "Readable operational patient summary",
                "Final operational guidance output",
            ],
        }
    )