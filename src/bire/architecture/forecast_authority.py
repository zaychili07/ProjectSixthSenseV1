# ============================================================
# BIRE OS — Forecasting Authority Hierarchy
# File: src/bire/architecture/forecast_authority.py
# Purpose:
# Backend helper for Chapter 60 forecasting authority
# hierarchy and orchestration governance.
# ============================================================

import pandas as pd


def build_forecasting_authority_hierarchy() -> pd.DataFrame:
    """
    Build the forecasting authority hierarchy table.

    This table defines which BIRE OS subsystems contribute
    specialized forecasting authority and ecosystem influence.

    Returns
    -------
    pd.DataFrame
        Forecasting authority hierarchy table.
    """

    return pd.DataFrame(
        {
            "subsystem": [
                "NCL",
                "IBPIP / LPMR",
                "HVI",
                "ATI",
                "TTI",
                "BMS / HIT",
                "BIRE-FI",
                "PSR-v2",
            ],

            "forecasting_authority": [

                "Signal reliability and noise qualification authority",

                "Longitudinal memory and stabilization history authority",

                "Hidden physiology and compensation authority",

                "Trajectory interpretation and instability pathway authority",

                "Intervention-aware deterioration authority",

                "Operational environment and care-context authority",

                "Forecast orchestration and synthesis authority",

                "Readable operational interpretation authority"
            ],

            "primary_forecast_contribution": [

                "Determine signal trustworthiness before forecasting",

                "Provide historical trajectory continuity and rebound memory",

                "Detect concealed deterioration and hidden burden",

                "Interpret instability cycling and trajectory movement",

                "Interpret therapeutic influence on longitudinal deterioration",

                "Adapt forecasting interpretation to care setting",

                "Coordinate multi-layer forecasting synthesis",

                "Translate forecasting into concise operational summaries"
            ],

            "should_not_control": [

                "Final trajectory interpretation alone",

                "Independent escalation authority",

                "Full operational forecasting independently",

                "Hidden physiology independently",

                "Clinical treatment decisions",

                "Longitudinal memory interpretation",

                "Raw clinician overload output",

                "Subsystem-level raw forecasting logic"
            ]
        }
    )