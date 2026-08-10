# ============================================================
# BIRE OS — Noise-Aware Forecasting Coordination
# File: src/bire/noise/noise_aware_forecasting.py
# Chapter: 61.5
#
# Created: 2026-05-26
# Updated: 2026-05-26
#
# Purpose:
# Backend helper for reliability-aware forecasting coordination,
# uncertainty-preserving forecast synchronization,
# and NCL + BIRE-FI operational forecasting doctrine.
# ============================================================

import pandas as pd


def build_noise_aware_forecasting_framework() -> pd.DataFrame:
    """
    Build the noise-aware forecasting coordination framework.

    This table defines how future BIRE OS forecasting
    behavior adapts to signal reliability degradation
    and uncertainty propagation.

    Returns
    -------
    pd.DataFrame
        Noise-aware forecasting governance table.
    """

    return pd.DataFrame(
        {
            "forecasting_state": [

                "STABLE_FORECASTING",

                "MODERATED_FORECASTING",

                "FRAGMENTED_FORECASTING",

                "VOLATILE_FORECASTING",

                "UNCERTAIN_FORECASTING",

                "RESTRICTED_FORECASTING"
            ],

            "forecasting_state_description": [

                "Reliable forecasting continuity preserved.",

                "Partial reliability reducing forecasting confidence.",

                "Disrupted continuity weakening replay synchronization.",

                "Unstable trajectory interpretation increasing variability.",

                "Unclear signal reliability amplifying uncertainty.",

                "Severe instability requiring cautious forecasting synthesis."
            ],

            "forecasting_behavior": [

                "Forecasting confidence remains stable.",

                "Forecasting confidence moderately adjusted.",

                "Replay continuity confidence weakened.",

                "Trajectory interpretation variability increased.",

                "Forecast uncertainty amplified.",

                "Forecasting confidence heavily restricted."
            ],

            "escalation_behavior": [

                "Escalation behavior remains operationally stable.",

                "Escalation confidence moderately reduced.",

                "Escalation continuity becomes fragmented.",

                "Escalation volatility caution required.",

                "Escalation uncertainty significantly increased.",

                "Escalation suppression or strong moderation likely required."
            ],

            "core_forecasting_principle": [

                "Reliable signals preserve stable forecasting",

                "Moderate uncertainty remains operationally usable",

                "Fragmentation weakens replay continuity",

                "Volatility requires adaptive trajectory forecasting",

                "Uncertainty should remain operationally visible",

                "Severe instability requires cautious forecasting synthesis"
            ]
        }
    )