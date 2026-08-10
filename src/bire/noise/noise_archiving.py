# ============================================================
# BIRE OS — NCL Noise Archiving Philosophy
# File: src/bire/noise/noise_archiving.py
# Chapter: 62.3
#
# Created: 2026-05-26
# Updated: 2026-05-26
#
# Purpose:
# Backend helper for NCL noise archiving governance,
# reliability traceability,
# and uncertainty-preserving audit behavior.
# ============================================================

import pandas as pd


def build_noise_archiving_framework() -> pd.DataFrame:
    """
    Build the NCL noise archiving governance framework.

    Returns
    -------
    pd.DataFrame
        Noise archiving governance table.
    """

    return pd.DataFrame(
        {
            "noise_state": [

                "LOW_RISK_NOISE",

                "MODERATE_NOISE",

                "HIGH_NOISE",

                "ARCHIVED_NOISE",

                "RECOVERABLE_NOISE",

                "PERSISTENT_NOISE"
            ],

            "noise_state_description": [

                "Minor instability preserved with low operational concern.",

                "Reliability concerns preserved with uncertainty propagation.",

                "Strong reliability concerns requiring downstream moderation.",

                "Removed from active interpretation but preserved for auditability.",

                "Temporarily unreliable signals eligible for reintegration.",

                "Repeated instability patterns preserved for reliability monitoring."
            ],

            "operational_behavior": [

                "Preserve active interpretation with minimal moderation.",

                "Preserve interpretation with confidence adjustment.",

                "Strongly moderate downstream interpretation confidence.",

                "Exclude from active operational reasoning while preserving traceability.",

                "Allow future recovery if reliability improves.",

                "Track repeated instability for longitudinal reliability analysis."
            ],

            "ecosystem_impact": [

                "Minimal operational disruption.",

                "Moderate uncertainty propagation.",

                "Reduced ecosystem confidence synchronization.",

                "Preserved auditability without active contamination.",

                "Supports future replay-aware reintegration.",

                "Supports longitudinal instability memory."
            ],

            "core_archiving_principle": [

                "Minor noise should remain observable",

                "Moderate noise should preserve uncertainty visibility",

                "High noise requires cautious operational moderation",

                "Archived noise should remain traceable",

                "Recoverable noise should preserve reintegration opportunity",

                "Persistent noise should preserve longitudinal memory"
            ]
        }
    )