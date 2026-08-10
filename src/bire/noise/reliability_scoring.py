# ============================================================
# BIRE OS — NCL Operational Reliability Scoring
# File: src/bire/noise/reliability_scoring.py
# Chapter: 61.7
#
# Created: 2026-05-26
# Updated: 2026-05-26
#
# Purpose:
# Backend helper for operational reliability scoring,
# uncertainty-aware signal weighting,
# and NCL ecosystem propagation mechanics.
# ============================================================

import pandas as pd


def build_reliability_scoring_framework() -> pd.DataFrame:
    """
    Build the NCL operational reliability scoring framework.

    This table defines operational reliability scoring
    behavior and downstream ecosystem propagation.

    Returns
    -------
    pd.DataFrame
        Reliability scoring governance table.
    """

    return pd.DataFrame(
        {
            "reliability_component": [

                "signal_quality_score",

                "continuity_score",

                "fragmentation_score",

                "volatility_score",

                "missingness_score",

                "confidence_modifier",

                "forecasting_modifier",

                "escalation_modifier"
            ],

            "primary_role": [

                "Evaluate direct signal trustworthiness",

                "Evaluate longitudinal continuity stability",

                "Evaluate continuity fragmentation severity",

                "Evaluate signal instability and inconsistency",

                "Evaluate operational missingness burden",

                "Adjust downstream ecosystem confidence",

                "Adjust forecasting synchronization confidence",

                "Adjust escalation suppression sensitivity"
            ],

            "downstream_impact": [

                "Impacts overall ecosystem reliability interpretation",

                "Impacts replay continuity confidence",

                "Impacts fragmentation-aware forecasting behavior",

                "Impacts trajectory interpretation stability",

                "Impacts uncertainty propagation weighting",

                "Moderates ecosystem confidence synchronization",

                "Moderates BIRE-FI forecasting behavior",

                "Moderates GSS escalation suppression behavior"
            ],

            "operational_behavior": [

                "Higher scores preserve stronger trustworthiness",

                "Higher scores preserve replay continuity",

                "Higher fragmentation reduces continuity confidence",

                "Higher volatility increases uncertainty propagation",

                "Higher missingness increases operational caution",

                "Lower confidence modifiers increase uncertainty visibility",

                "Lower forecasting modifiers reduce forecasting certainty",

                "Lower escalation modifiers increase suppression sensitivity"
            ],

            "core_reliability_principle": [

                "Signals should remain reliability-aware",

                "Continuity strengthens operational interpretation",

                "Fragmentation weakens replay confidence",

                "Volatility requires adaptive interpretation",

                "Missingness increases uncertainty awareness",

                "Confidence should dynamically adapt to reliability",

                "Forecasting should remain uncertainty-aware",

                "Escalation should remain meaningful under degradation"
            ]
        }
    )