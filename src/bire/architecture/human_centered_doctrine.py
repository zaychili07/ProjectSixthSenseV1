# ============================================================
# BIRE OS — Human-Centered Operational Intelligence Doctrine
# File: src/bire/architecture/human_centered_doctrine.py
# Chapter: 60.10
#
# Created: 2026-05-25
# Updated: 2026-05-25
#
# Purpose:
# Backend helper for human-centered operational intelligence
# doctrine, clinician-facing ecosystem coordination,
# and operational usability governance.
# ============================================================

import pandas as pd


def build_human_centered_operational_doctrine() -> pd.DataFrame:
    """
    Build the human-centered operational doctrine table.

    This table defines how future BIRE OS subsystems
    should preserve operational readability,
    usability, and meaningful human interpretation.

    Returns
    -------
    pd.DataFrame
        Human-centered operational doctrine table.
    """

    return pd.DataFrame(
        {
            "ecosystem_component": [
                "NCL",
                "LPMR",
                "HVI",
                "ATI",
                "TTI",
                "BMS / HIT",
                "BIRE-FI",
                "GSS / GSS-VE",
                "PSR-v2",
            ],

            "human_centered_role": [

                "Preserve trustworthy signal interpretation",

                "Preserve longitudinal continuity understanding",

                "Preserve hidden deterioration visibility",

                "Preserve readable trajectory interpretation",

                "Preserve intervention-awareness readability",

                "Preserve operational adaptability",

                "Preserve coordinated forecasting interpretation",

                "Preserve meaningful escalation governance",

                "Preserve concise operational communication"
            ],

            "human_operational_goal": [

                "Reduce false confidence from noisy signals",

                "Improve continuity awareness across encounters",

                "Prevent concealed deterioration from being overlooked",

                "Improve understanding of instability progression",

                "Improve understanding of treatment-related trajectory behavior",

                "Adapt interpretation to operational workflow",

                "Coordinate ecosystem-wide forecasting readability",

                "Reduce alert fatigue while preserving meaningful escalation",

                "Present ecosystem intelligence clearly and concisely"
            ],

            "human_risk_if_unmanaged": [

                "Trust erosion from unreliable interpretation",

                "Loss of patient trajectory continuity awareness",

                "Hidden deterioration blindness",

                "Trajectory confusion and instability misinterpretation",

                "Treatment-effect misunderstanding",

                "Operational workflow mismatch",

                "Forecasting interpretation overload",

                "Alert fatigue and escalation desensitization",

                "Clinician cognitive overload"
            ],

            "core_human_centered_principle": [

                "Trustworthy signals build trust",

                "Continuity improves operational understanding",

                "Hidden deterioration should remain visible",

                "Trajectory interpretation should remain understandable",

                "Intervention interpretation should remain contextualized",

                "Operational adaptation should remain flexible",

                "Forecasting should remain confidence-aware",

                "Escalation should remain meaningful",

                "Operational communication should remain human-readable"
            ]
        }
    )