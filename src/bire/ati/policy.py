# ============================================================
# BIRE OS — Advanced Trajectory Intelligence (ATI)
# File: src/bire/ati/policy.py
# Chapter: 64.1
#
# Created: 2026-06-13
# Updated: 2026-06-13
#
# CHANGED:
# - Added ATI policy definition helpers.
#
# Purpose:
# Defines ATI policy boundaries before trajectory interpretation
# logic is implemented.
# ============================================================

import pandas as pd


def build_ati_policy_table():
    """
    Build ATI policy table.

    ATI interprets trajectory behavior.
    ATI does not create physiologic measurements, forecast outcomes,
    evaluate interventions, store memory, or classify recovery.
    """

    return pd.DataFrame({
        "policy_domain": [
            "Trajectory Interpretation",
            "Physiologic Measurement Creation",
            "Forecast Generation",
            "Longitudinal Memory",
            "Recovery Classification",
            "Intervention Evaluation",
            "Clinician Display",
        ],
        "ati_policy": [
            "Allowed",
            "Not Allowed",
            "Not Allowed",
            "Not Allowed",
            "Not Allowed",
            "Not Allowed",
            "Not Allowed",
        ],
        "notes": [
            "ATI may interpret how signals move, persist, accelerate, stabilize, or reverse over time.",
            "HVI owns physiologic measurement creation.",
            "BIRE-FI owns forecasting.",
            "LPMR owns longitudinal memory and recall.",
            "TARI owns recovery authenticity and durability interpretation.",
            "TTI owns intervention effect interpretation.",
            "PSR-v2 owns clinician-facing communication.",
        ],
    })