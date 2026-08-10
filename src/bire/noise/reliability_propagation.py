# ============================================================
# BIRE OS — Reliability Modifier Propagation Logic
# File: src/bire/noise/reliability_propagation.py
# Chapter: 61.8
#
# Created: 2026-05-26
# Updated: 2026-05-26
#
# Purpose:
# Backend helper for operational reliability modifier
# propagation, uncertainty synchronization,
# and ecosystem-wide reliability-aware coordination.
# ============================================================

import pandas as pd


def build_reliability_propagation_framework() -> pd.DataFrame:
    """
    Build the reliability modifier propagation framework.

    This table defines how NCL reliability modifiers
    influence downstream ecosystem interpretation.

    Returns
    -------
    pd.DataFrame
        Reliability propagation governance table.
    """

    return pd.DataFrame(
        {
            "modifier_component": [

                "confidence_modifier",

                "forecasting_modifier",

                "escalation_modifier",

                "trajectory_modifier",

                "replay_modifier",

                "hidden_burden_modifier",

                "visibility_modifier"
            ],

            "primary_role": [

                "Adjust ecosystem-wide confidence weighting",

                "Adjust BIRE-FI forecasting certainty",

                "Adjust GSS suppression sensitivity",

                "Adjust ATI trajectory interpretation confidence",

                "Adjust LPMR replay continuity weighting",

                "Adjust HVI hidden burden amplification sensitivity",

                "Adjust PSR-v2 uncertainty communication visibility"
            ],

            "downstream_behavior": [

                "Higher confidence preserves stronger operational certainty",

                "Lower forecasting modifiers reduce forecasting confidence",

                "Lower escalation modifiers increase suppression moderation",

                "Lower trajectory modifiers weaken instability interpretation certainty",

                "Lower replay modifiers weaken replay continuity confidence",

                "Lower hidden burden modifiers reduce aggressive hidden burden amplification",

                "Higher visibility modifiers preserve uncertainty readability"
            ],

            "ecosystem_impact": [

                "Impacts ecosystem-wide confidence synchronization",

                "Impacts forecasting stability and replay certainty",

                "Impacts escalation moderation and alert burden",

                "Impacts trajectory continuity interpretation",

                "Impacts longitudinal continuity reasoning",

                "Impacts hidden deterioration interpretation sensitivity",

                "Impacts operational readability and clinician interpretation"
            ],

            "core_propagation_principle": [

                "Confidence should dynamically adapt to reliability",

                "Forecasting should remain uncertainty-aware",

                "Escalation should remain meaningful under degradation",

                "Trajectory interpretation should remain adaptive",

                "Replay continuity should remain reliability-aware",

                "Hidden burden amplification should remain cautious",

                "Operational uncertainty should remain visible"
            ]
        }
    )