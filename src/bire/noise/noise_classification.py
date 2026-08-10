# ============================================================
# BIRE OS — Noise Classification Governance
# File: src/bire/noise/noise_classification.py
# Chapter: 61.2
#
# Created: 2026-05-26
# Updated: 2026-05-26
#
# Purpose:
# Backend helper for operational noise classification,
# signal reliability state governance,
# and NCL ecosystem propagation doctrine.
# ============================================================

import pandas as pd


def build_noise_classification_framework() -> pd.DataFrame:
    """
    Build the operational noise classification framework.

    This table defines future NCL signal reliability states,
    interpretation behavior,
    and downstream ecosystem influence.

    Returns
    -------
    pd.DataFrame
        Noise classification governance table.
    """

    return pd.DataFrame(
        {
            "signal_state": [

                "RELIABLE_SIGNAL",

                "PARTIAL_SIGNAL",

                "NOISY_SIGNAL",

                "FRAGMENTED_SIGNAL",

                "UNSTABLE_SIGNAL",

                "SENSOR_DRIFT_SUSPECTED",

                "LOW_CONFIDENCE_SIGNAL"
            ],

            "signal_state_description": [

                "Stable trustworthy signal quality.",

                "Usable signal with moderate reliability concerns.",

                "Degraded signal quality requiring confidence moderation.",

                "Disrupted longitudinal continuity and incomplete signal behavior.",

                "Inconsistent or highly volatile measurements.",

                "Probable measurement drift or physiologic inconsistency detected.",

                "Unreliable interpretation state requiring strong caution."
            ],

            "forecasting_impact": [

                "Minimal reliability restriction.",

                "Moderate forecasting confidence moderation.",

                "Reduced forecasting confidence stability.",

                "Reduced longitudinal replay reliability.",

                "Trajectory instability interpretation caution.",

                "Potential hidden burden interpretation instability.",

                "Strong forecasting uncertainty propagation."
            ],

            "escalation_impact": [

                "Escalation behavior remains stable.",

                "Escalation confidence slightly moderated.",

                "Escalation confidence reduced.",

                "Escalation continuity becomes fragmented.",

                "Escalation volatility awareness required.",

                "Escalation reliability caution required.",

                "Escalation suppression or moderation likely required."
            ],

            "core_governance_principle": [

                "Trustworthy signals preserve stable interpretation",

                "Moderate uncertainty remains operationally usable",

                "Noise reduces confidence without erasing interpretation",

                "Continuity fragmentation weakens replay interpretation",

                "Volatility requires adaptive trajectory reasoning",

                "Drift suspicion requires reliability awareness",

                "Low confidence requires cautious operational synthesis"
            ]
        }
    )