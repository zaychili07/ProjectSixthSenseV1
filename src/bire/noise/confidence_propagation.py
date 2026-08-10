# ============================================================
# BIRE OS — Signal Quality Confidence Propagation
# File: src/bire/noise/confidence_propagation.py
# Chapter: 61.4
#
# Created: 2026-05-26
# Updated: 2026-05-26
#
# Purpose:
# Backend helper for reliability-aware confidence propagation,
# uncertainty synchronization,
# and ecosystem-wide confidence moderation doctrine.
# ============================================================

import pandas as pd


def build_confidence_propagation_framework() -> pd.DataFrame:
    """
    Build the signal quality confidence propagation framework.

    This table defines how signal reliability states
    influence downstream ecosystem confidence behavior.

    Returns
    -------
    pd.DataFrame
        Confidence propagation governance table.
    """

    return pd.DataFrame(
        {
            "confidence_state": [

                "HIGH_CONFIDENCE_PROPAGATION",

                "MODERATE_CONFIDENCE_PROPAGATION",

                "REDUCED_CONFIDENCE_PROPAGATION",

                "FRAGMENTED_CONFIDENCE_PROPAGATION",

                "VOLATILE_CONFIDENCE_PROPAGATION",

                "UNCERTAIN_CONFIDENCE_PROPAGATION",

                "CRITICAL_CONFIDENCE_DEGRADATION"
            ],

            "confidence_state_description": [

                "Reliable signals preserving strong ecosystem confidence.",

                "Partial reliability reducing confidence stability.",

                "Degraded signals weakening forecasting reliability.",

                "Disrupted continuity weakening replay synchronization.",

                "Unstable signals increasing interpretation variability.",

                "Unclear reliability producing uncertainty amplification.",

                "Severe signal instability requiring cautious operational synthesis."
            ],

            "forecasting_behavior": [

                "Forecasting confidence remains stable.",

                "Forecasting confidence slightly moderated.",

                "Forecasting confidence reduced.",

                "Replay continuity confidence weakened.",

                "Trajectory interpretation variability increased.",

                "Forecast uncertainty amplified.",

                "Forecasting confidence heavily restricted."
            ],

            "escalation_behavior": [

                "Escalation behavior remains reliable.",

                "Escalation confidence moderately adjusted.",

                "Escalation confidence reduced.",

                "Escalation continuity becomes fragmented.",

                "Escalation volatility caution required.",

                "Escalation uncertainty increases significantly.",

                "Escalation suppression or strong moderation likely required."
            ],

            "core_confidence_principle": [

                "Reliable signals preserve stable operational confidence",

                "Moderate uncertainty remains operationally usable",

                "Reduced reliability moderates confidence",

                "Fragmentation weakens continuity confidence",

                "Volatility requires adaptive confidence reasoning",

                "Uncertainty should remain operationally visible",

                "Critical degradation requires cautious operational synthesis"
            ]
        }
    )